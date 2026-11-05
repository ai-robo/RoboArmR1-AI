#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CONTINUOUS RoboArm R1: continuous FOUR-color detection with relaxed ArUco checks.

LIVE edition: main thread continues capturing and displaying camera frames
while the Serial worker runs. ESC stops the loop (not an emergency stop).

After each placement wait until drop zone AND old pickup spot have been
visibly cleared. This avoids repeatedly grabbing the deposited or failed cube.
Motion sequence, XYZ, timing, and physical safety gates unchanged; WHITE detection disabled.
Default without --execute is a no-motion, continuous preview.

Uses tested earlier modules: roboarm_move_to_cube.py,
roboarm_pickup_cube.py, and roboarm_pick_and_place.py (RED zone).
Requires the *user's own* board_calibration.npz and
robot_alignment_far_DIAGNOSTIC.npz from this camera and robot.

IMPORTANT: firmware `state` is programmed/commanded state, NOT physical feedback.
No collision sensor, force feedback or stop control is provided by this code.
Never leave the robot unattended; keep a physical power-off accessible.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import roboarm_move_to_cube as base
import roboarm_pickup_cube as grip
import roboarm_pick_and_place as zone

WINDOW = 'RoboArm - AUTO pick -> center between ArUco 2 and 3'
# Fixed physical target, independent of image resolution and ArUco occlusion.
DROP_BOARD_MM = (np.asarray(base.MARKER_BOARD_MM[2], dtype=float)
                 + np.asarray(base.MARKER_BOARD_MM[3], dtype=float)) / 2.0
STABLE_FRAMES = 7
MAX_CUBE_JITTER_MM = 3.0
# Start automatically only after the same candidate remains stable for 1 second.
DETECT_TIMEOUT_S = 25.0  # legacy one-shot routine only; continuous loop never times out
# Live markers are only a safety reference: mapping uses saved board_calibration.npz.
# Any TWO recognized ArUco markers suffice; IDs 2 and 3 are NOT required in view.
MIN_VISIBLE_ARUCO = 2
MAX_MARKER_DRIFT_MM = 10.0
DETECTION_CONFIRM_S = 1.0
DROP_GUARD_RADIUS_MM = 39.0
SOURCE_GUARD_RADIUS_MM = 22.0
RESET_CLEAR_FRAMES = 12
# Stop before pickup if a cube is already next to the drop target.
MIN_PICK_TO_DROP_MM = 30.0
# State polling cannot prove physical arrival. Keep real wait time after each go.
# FAST is a shorter, UNVERIFIED dwell time for a supervised first trial.
# ORIGINAL restores the timings of the previous working program.
@dataclass(frozen=True)
class Timing:
    claw_s: float
    xy_base_s: float
    xy_per_mm_s: float
    vertical_base_s: float
    vertical_per_mm_s: float
    preview_s: float


PACE_PROFILES = {
    'fast': Timing(claw_s=1.4, xy_base_s=1.0, xy_per_mm_s=0.032,
                   vertical_base_s=1.0, vertical_per_mm_s=0.032,
                   preview_s=1.5),
    'original': Timing(claw_s=2.5, xy_base_s=2.0, xy_per_mm_s=0.055,
                       vertical_base_s=2.0, vertical_per_mm_s=0.045,
                       preview_s=3.0),
}

# Camera HSV calibration (OpenCV: H = 0..179, S/V = 0..255).
# Gold is a YELLOW/ORANGE class. WHITE is deliberately excluded.
# These thresholds are initial approximations: verify each real cube in dry run.
HSV_RANGES = {
    'red': [((0, 90, 60), (12, 255, 255)),
            ((170, 90, 60), (179, 255, 255))],
    'green': [((35, 65, 55), (90, 255, 255))],
    'blue': [((92, 65, 55), (135, 255, 255))],
    'gold': [((12, 65, 75), (37, 255, 255))],
}
COLOR_BGR = {
    'red': (0, 0, 255),
    'green': (0, 230, 0),
    'blue': (255, 100, 0),
    'gold': (0, 190, 255),
}
# Reject small reflections and large non-cube highlights using size/shape checks.
# Object size on BOARD is only an estimate because we see the cube TOP at Z~=20mm.
CUBE_WIDTH_MM_MIN = 11.0
CUBE_WIDTH_MM_MAX = 34.0
CUBE_ASPECT_MAX = 1.60


def find_four_color_cubes(frame, search_mask, min_area, H, detector=None,
                          marker_corners=None):
    """Detect cubes only in red, green, blue and gold.

    White is excluded to avoid false positives from the RoboArm gripper.
    Gold may require HSV calibration with physical samples.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    valid_mask = search_mask.copy()
    # Exclude complete ArUco marker squares, including white borders.
    # Small white squares within the markers must never be mistaken for cubes.
    if detector is not None or marker_corners is not None:
        if marker_corners is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners, _, _ = detector.detectMarkers(gray)
        for corner in marker_corners:
            pts = corner.reshape(4, 2).astype(np.float32)
            center = pts.mean(axis=0)
            pts = np.rint(center + 1.32 * (pts - center)).astype(np.int32)
            cv2.fillConvexPoly(valid_mask, pts, 0)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    items = []
    color_masks = {}
    for color, ranges in HSV_RANGES.items():
        binary = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            binary |= cv2.inRange(hsv, np.asarray(low, np.uint8),
                                  np.asarray(high, np.uint8))
        binary = cv2.bitwise_and(binary, valid_mask)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        # Mask AFTER closing so it cannot grow into ArUco exclusions / outside ROI.
        binary = cv2.bitwise_and(binary, valid_mask)
        color_masks[color] = binary
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w/h, h/w)
            if aspect > CUBE_ASPECT_MAX:
                continue
            fill_ratio = area / (w * h)
            if fill_ratio < 0.50:
                continue
            moments = cv2.moments(cnt)
            if moments['m00'] <= 0:
                continue
            cx = float(moments['m10'] / moments['m00'])
            cy = float(moments['m01'] / moments['m00'])
            # Convert bounding-box dimensions to approximate board millimetres.
            # The projected shape is approximate for a raised cube top.
            sample_px = [(cx - w/2, cy), (cx + w/2, cy),
                         (cx, cy - h/2), (cx, cy + h/2)]
            try:
                sample_board = base.project(H, sample_px)
            except cv2.error:
                continue
            width_mm = float(np.linalg.norm(sample_board[1] - sample_board[0]))
            height_mm = float(np.linalg.norm(sample_board[3] - sample_board[2]))
            if not (CUBE_WIDTH_MM_MIN <= width_mm <= CUBE_WIDTH_MM_MAX and
                    CUBE_WIDTH_MM_MIN <= height_mm <= CUBE_WIDTH_MM_MAX):
                continue
            items.append({
                'color': color, 'centre_px': [cx, cy],
                'bbox': (int(x), int(y), int(w), int(h)),
                'area': float(area), 'size_mm_approx': [width_mm, height_mm],
            })
    return items, color_masks


def target_from_cube(cube, H, R, t):
    """Color-independent equivalent to base.select_target; no motion enabled here."""
    board = base.project(H, [cube['centre_px']])[0]
    bx, by = map(float, board)
    if not (np.isfinite(board).all() and
            0 <= bx <= base.BOARD_SIZE_MM[0] and
            0 <= by <= base.BOARD_SIZE_MM[1]):
        raise ValueError('Cube location outside the calibrated plywood board')
    robot = R @ board + t
    if not np.isfinite(robot).all():
        raise ValueError('Non-finite XY for RoboArm')
    return {
        'color': cube['color'],
        'pixel': [round(float(v), 1) for v in cube['centre_px']],
        'board_mm_top_estimate': [round(bx, 2), round(by, 2)],
        'robot_xy_top_estimate': [round(float(robot[0]), 2), round(float(robot[1]), 2)],
        'bbox': cube['bbox'],
        'size_mm_approx': cube['size_mm_approx'],
        'warning': 'Top face of 20-mm cube, not calibrated base center'
    }


def make_work_mask(H, shape):
    outline = zone.workspace_outline_px(H)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [outline], 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask, outline


def check_reference_markers(H, markers):
    """Detect camera/board drift using ANY 2+ visible markers.

    Saved board_calibration.npz, not live marker pairs, defines the fixed
    plywood coordinates and the midpoint between markers 2 and 3.
    Do not bypass the drift check if fewer than two are detected.
    """
    return base.marker_check(H, markers, min_markers=MIN_VISIBLE_ARUCO,
                             max_mm_error=MAX_MARKER_DRIFT_MM)


def examine_frame(frame, detector, H, R, t, work_mask, min_area):
    markers = base.marker_centres(frame, detector)
    refs_ok, msg, _ = check_reference_markers(H, markers)
    cubes, _ = find_four_color_cubes(frame, work_mask, min_area, H, detector)
    if not refs_ok or len(cubes) != 1:
        return None, f'{msg}; detected cubes={len(cubes)}'
    # Reject ambiguous detections (including two different colors).
    cube = cubes[0]
    src = target_from_cube(cube, H, R, t)
    board = np.asarray(src['board_mm_top_estimate'], float)
    rob = np.asarray(src['robot_xy_top_estimate'], float)
    zone.check_workspace(board, 'Pickup', zone.PICK_MARGIN)
    zone.check_board_edge(board, 'Pickup')
    zone.check_robot_xy_envelope(rob, 'Pickup')
    if np.linalg.norm(board - DROP_BOARD_MM) < MIN_PICK_TO_DROP_MM:
        raise RuntimeError('Cube already too close to the destination midpoint: do not move.')
    return src, msg


def choose_stable_target(cap, detector, H, R, t, work_mask, outline,
                         min_area, timeout_s=DETECT_TIMEOUT_S, expected_shape=None):
    recent = deque(maxlen=STABLE_FRAMES)
    started = time.monotonic()
    last_message = 'Searching for ONE cube: red/green/blue/gold'
    last_frame = None
    drop_uv = base.project(np.linalg.inv(H), [DROP_BOARD_MM])[0]
    dx, dy = np.rint(drop_uv).astype(int)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    while time.monotonic() - started < timeout_s:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Camera failed while finding cube')
        if ((last_frame is not None and frame.shape != last_frame.shape)
                or (expected_shape is not None and frame.shape != expected_shape)):
            raise RuntimeError('Camera resolution changed')
        last_frame = frame
        canvas = frame.copy()
        cv2.polylines(canvas, [outline], True, (0, 0, 255), 2)
        cv2.drawMarker(canvas, (dx,dy), (0,255,255), cv2.MARKER_CROSS, 22, 2)
        base.draw_text(canvas, 'DROP CENTER (ArUco 2 + 3)', dx-110, max(18,dy-18), (0,255,255), .44)
        target = None
        try:
            target, last_message = examine_frame(frame, detector, H, R, t,
                                                   work_mask, min_area)
        except (RuntimeError, ValueError) as exc:
            last_message = str(exc)
        if target is None:
            recent.clear()
        else:
            board = np.asarray(target['board_mm_top_estimate'], float)
            if recent and (recent[-1]['color'] != target['color'] or
                           np.linalg.norm(board - np.asarray(recent[-1]['board_mm_top_estimate'])) >
                           MAX_CUBE_JITTER_MM):
                recent.clear()
            recent.append(target)
            px = tuple(np.rint(target['pixel']).astype(int))
            cv2.circle(canvas, px, 8, COLOR_BGR[target['color']], 2)
            x, y, w, h = target['bbox']
            cv2.rectangle(canvas, (x, y), (x+w, y+h), COLOR_BGR[target['color']], 2)
            base.draw_text(canvas, target['color'].upper(), x, max(67,y-8),
                           COLOR_BGR[target['color']], .52)
        base.draw_text(canvas, f'AUTO DETECT: {len(recent)}/{STABLE_FRAMES} stable',
                       7, 25, (0,255,255), .5)
        base.draw_text(canvas, last_message[:90], 7, 50, (70,235,70), .38)
        base.draw_text(canvas, 'Q/ESC = cancel before motion', 7, canvas.shape[0]-15, (255,255,255), .43)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(65) & 0xFF
        if key in (ord('q'), 27):
            raise RuntimeError('Cancelled before motion')
        if len(recent) == STABLE_FRAMES:
            positions = np.asarray([v['board_mm_top_estimate'] for v in recent], float)
            center = np.median(positions, axis=0)
            if np.max(np.linalg.norm(positions - center, axis=1)) <= MAX_CUBE_JITTER_MM:
                selected = recent[-1].copy()
                selected['board_mm_top_estimate'] = [float(x) for x in center]
                robot = R @ center + t
                selected['robot_xy_top_estimate'] = [float(x) for x in robot]
                return selected, frame.copy()
            recent.clear()
    raise RuntimeError('No single stable cube detected within timeout. NO MOVEMENT.')


def validate_static_targets(src, R, t):
    board = np.asarray(src['board_mm_top_estimate'], float)
    target = np.asarray(src['robot_xy_top_estimate'], float)
    dst = DROP_BOARD_MM.copy()
    dst_robot = R @ dst + t
    zone.check_workspace(board, 'Pickup', zone.PICK_MARGIN)
    zone.check_board_edge(board, 'Pickup')
    zone.check_robot_xy_envelope(target, 'Pickup')
    zone.check_workspace(dst, 'Drop midpoint', zone.DROP_MARGIN)
    zone.check_board_edge(dst, 'Drop midpoint')
    zone.check_robot_xy_envelope(dst_robot, 'Drop midpoint')
    if float(np.linalg.norm(board-dst)) < MIN_PICK_TO_DROP_MM:
        raise RuntimeError('Cube already at destination; placement would overlap pickup')
    # The entire route of the held cube must remain inside the user-defined red area.
    zone.safe_xy_path(target, dst_robot, R, t)
    return target, dst, dst_robot


def stage_send(ser, cmd, expected_xyz, expected_claw, label, settle_s,
               progress=None, cancel_event=None):
    # Q requests a halt BEFORE the next command. It does not stop an already
    # commanded servo movement. Physical emergency stop = cut motor power.
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError('Stop requested in camera window: no further go commands sent.')
    if progress is not None:
        progress(label, cmd)
    # Never automatically retry a go command. On any mismatch, fail closed.
    print(f'  >>> {label}: {cmd}', flush=True)
    ser.write((cmd + '\n').encode('ascii'))
    ser.flush()
    # Do not rely on firmware state for physical servo completion: wait first.
    print(f'      waiting {settle_s:.2f}s for physical movement...', flush=True)
    time.sleep(max(0.0, settle_s))
    zone.wait_pose(ser, expected_xyz, expected_claw, label, deadline_s=9.0)


def xy_route(ser, from_xy, to_xy, R, t, claw, label, timing,
             progress=None, cancel_event=None):
    """Limit each go x,y segment to 60 mm; precheck entire sampled path."""
    route = zone.safe_xy_path(from_xy, to_xy, R, t)
    for i, pt in enumerate(route, 1):
        now = grip.checked_state(ser)
        grip.near(now[2], zone.TRAVEL_Z, 3.0, label + ' Z')
        grip.near(now[3], claw, 3.0, label + ' claw')
        if np.linalg.norm(now[:2] - np.asarray(pt)) < 0.2:
            continue
        zone.safe_xy_path(now[:2], pt, R, t)
        dist = float(np.linalg.norm(now[:2] - pt))
        stage_send(ser, f'go x={pt[0]:.1f} y={pt[1]:.1f}',
                   np.r_[pt, zone.TRAVEL_Z], claw, f'{label} {i}/{len(route)}',
                   timing.xy_base_s + dist * timing.xy_per_mm_s,
                   progress=progress, cancel_event=cancel_event)


def do_cycle(port, src, dst_robot, R, t, return_up=True, pace='fast',
             progress=None, cancel_event=None):
    """One unattended *software sequence*, NOT unattended physical operation."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError('Install pyserial: pip install pyserial') from exc

    timing = PACE_PROFILES[pace]
    if progress is not None:
        progress('CONNECT', 'Opening Serial / checking position')
    pick_xy = np.asarray(src['robot_xy_top_estimate'], float)
    print(f'Starting AUTO cycle: pace={pace}. Stay beside robot; power cutoff must be accessible.', flush=True)
    print('Opening Serial may reset ESP32. Robot must already be at Z=85.', flush=True)
    with serial.Serial(port, baudrate=115200, timeout=0.2, write_timeout=1.5) as ser:
        time.sleep(2.2)
        state = grip.checked_state(ser)
        grip.near(state[2], zone.TRAVEL_Z, 3.0, 'Initial Z')
        # This only checks programmed coordinates, NOT actual servo angles.
        zone.safe_xy_path(state[:2], pick_xy, R, t)
        zone.safe_xy_path(pick_xy, dst_robot, R, t)
        print(f'Initial state={state.round(1).tolist()}')
        print('Pickup XY:', np.round(pick_xy,1).tolist(), 'Drop XY:', np.round(dst_robot,1).tolist())

        # Stage 1: OPEN
        stage_send(ser, 'go claw=50', state[:3], zone.OPEN_CLAW,
                   'OPEN', timing.claw_s, progress, cancel_event)
        # Stage 2: XY (possibly several small substeps)
        xy_route(ser, state[:2], pick_xy, R, t, zone.OPEN_CLAW, 'XY', timing,
                 progress, cancel_event)
        # Stage 3: DOWN to previously established pickup height
        now = grip.checked_state(ser)
        if np.linalg.norm(now[:2] - pick_xy) > 3.0:
            raise RuntimeError('XY mismatch before pickup DOWN')
        grip.near(now[3], zone.OPEN_CLAW, 3.0, 'Claw before pickup')
        stage_send(ser, 'go z=15', np.r_[pick_xy, zone.PICK_Z], zone.OPEN_CLAW,
                   'DOWN pickup', timing.vertical_base_s +
                   abs(zone.TRAVEL_Z-zone.PICK_Z)*timing.vertical_per_mm_s,
                   progress, cancel_event)
        # Stage 4: GRIP
        stage_send(ser, 'go claw=0', np.r_[pick_xy, zone.PICK_Z], zone.CLOSED_CLAW,
                   'GRIP', timing.claw_s, progress, cancel_event)
        # Stage 5: UP
        stage_send(ser, 'go z=85', np.r_[pick_xy, zone.TRAVEL_Z], zone.CLOSED_CLAW,
                   'UP pickup', timing.vertical_base_s +
                   abs(zone.TRAVEL_Z-zone.PICK_Z)*timing.vertical_per_mm_s,
                   progress, cancel_event)
        # Stage 6: MOVE to physical midpoint between ArUco 2 and 3
        xy_route(ser, pick_xy, dst_robot, R, t, zone.CLOSED_CLAW, 'MOVE', timing,
                 progress, cancel_event)
        # Stage 7: DOWN at placement height Z=25, NOT the pickup height Z=15.
        now = grip.checked_state(ser)
        if np.linalg.norm(now[:2] - dst_robot) > 3.0:
            raise RuntimeError('XY mismatch before place DOWN')
        grip.near(now[3], zone.CLOSED_CLAW, 3.0, 'Claw before release')
        stage_send(ser, 'go z=25', np.r_[dst_robot, zone.PLACE_Z], zone.CLOSED_CLAW,
                   'DOWN place', timing.vertical_base_s +
                   abs(zone.TRAVEL_Z-zone.PLACE_Z)*timing.vertical_per_mm_s,
                   progress, cancel_event)
        # Stage 8: OPEN -> optional upward retract follows as before.
        stage_send(ser, 'go claw=50', np.r_[dst_robot, zone.PLACE_Z], zone.OPEN_CLAW,
                   'OPEN release', timing.claw_s, progress, cancel_event)
        print('\nAUTO CYCLE COMPLETE (firmware states matched).')
        print('Physically verify that cube was released at the midpoint.')
        if return_up:
            # Return above the drop point to enable the next run from Z=85.
            # This is a straight upward retract, NOT a new horizontal path.
            stage_send(ser, 'go z=85', np.r_[dst_robot, zone.TRAVEL_Z],
                       zone.OPEN_CLAW, 'UP empty (ready for next run)',
                       timing.vertical_base_s +
                       abs(zone.TRAVEL_Z-zone.PLACE_Z)*timing.vertical_per_mm_s,
                   progress, cancel_event)
            print('READY: empty gripper opened at Z=85 for next run.')
        else:
            print('Robot ends at Z=25. Move to a verified safe Z=85 before next run.')



class StageMonitor:
    """Thread-safe status only. All OpenCV UI functions run on the main thread."""
    def __init__(self):
        self.lock = threading.Lock()
        self.label = 'PREPARING'
        self.command = ''
        self.started = time.monotonic()

    def update(self, label, command):
        with self.lock:
            self.label = label
            self.command = command
            self.started = time.monotonic()

    def snapshot(self):
        with self.lock:
            return self.label, self.command, time.monotonic() - self.started


def draw_live_frame(frame, outline, H, source, drop_board, stage, command='',
                    stage_elapsed=0, elapsed_total=0, cancelled=False):
    """Overlay the fixed pickup/drop markers on a FRESH camera frame.

    No re-detection or ArUco checks during motion: gripper may occlude both.
    """
    canvas = frame.copy()
    cv2.polylines(canvas, [outline], True, (0, 0, 255), 2)
    uv = base.project(np.linalg.inv(H), [drop_board])[0]
    dpx = tuple(np.rint(uv).astype(int))
    cv2.drawMarker(canvas, dpx, (0, 255, 255), cv2.MARKER_CROSS, 25, 2)
    # Mark where the pickup was found initially; camera image itself is live.
    origin = tuple(np.rint(source['pixel']).astype(int))
    cv2.circle(canvas, origin, 9, COLOR_BGR[source['color']], 2)
    cv2.arrowedLine(canvas, origin, dpx, (80, 220, 240), 2, tipLength=.05)
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w-1, 88), (10, 10, 10), -1)
    if cancelled:
        banner = 'STOP REQUESTED: no more go commands'
        banner_color = (0, 100, 255)
    elif stage == 'DONE':
        banner = 'DONE - verify cube placement'
        banner_color = (0, 220, 0)
    elif stage == 'ERROR':
        banner = 'STOPPED - inspect physical robot'
        banner_color = (0, 80, 255)
    else:
        banner = 'LIVE CAMERA  |  ' + stage
        banner_color = (0, 255, 255)
    base.draw_text(canvas, banner, 8, 24, banner_color, .57)
    base.draw_text(canvas, f'{command}'[:76], 8, 48, (255,255,255), .47)
    base.draw_text(canvas, f'Cycle: {elapsed_total:0.1f}s   Stage: {stage_elapsed:0.1f}s',
                   8, 72, (220,220,220), .44)
    cv2.rectangle(canvas, (0, h-30), (w-1, h-1), (10, 10, 10), -1)
    base.draw_text(canvas, 'Q / ESC: stop after current command, NOT emergency stop',
                   5, h-10, (255,255,255), .36)
    return canvas


def run_with_live_camera(cap, frame_shape, outline, H, source, drop_board,
                         port, dst_robot, R, t, return_up=True, pace='fast',
                         postview_s=1.5):
    """Robot Serial runs in a worker; camera read/imshow/waitKey on main thread.

    The camera is NOT released before movement. It is always closed by main().
    Camera loss requests that no further commands be sent. Q is NOT a
    servo emergency stop; use the physical motor power cutoff when needed.
    """
    monitor = StageMonitor()
    stop_requested = threading.Event()
    finished = threading.Event()
    errors = []
    def robot_worker():
        try:
            do_cycle(port, source, dst_robot, R, t,
                     return_up=return_up, pace=pace,
                     progress=monitor.update, cancel_event=stop_requested)
            monitor.update('DONE', 'Cycle completed (firmware state checks only)')
        except Exception as exc:
            errors.append(exc)
            monitor.update('ERROR', str(exc))
        finally:
            finished.set()

    worker = threading.Thread(target=robot_worker, name='roboarm-serial', daemon=False)
    worker.start()
    start = time.monotonic()
    last_frame = None
    try:
        while not finished.is_set():
            ok, current_frame = cap.read()
            if not ok or current_frame.shape != frame_shape:
                stop_requested.set()
                # Keep current command un-interrupted but stop further commands.
                raise RuntimeError('Camera failed/resolution changed DURING motion; stop requested.')
            last_frame = current_frame
            label, command, stage_elapsed = monitor.snapshot()
            canvas = draw_live_frame(current_frame, outline, H, source, drop_board,
                                     label, command, stage_elapsed,
                                     time.monotonic()-start, stop_requested.is_set())
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                stop_requested.set()
            # Do NOT stop UI while active Serial commands are settling.
        worker.join()
        # Keep a short visible confirmation on screen after the last movement.
        end_at = time.monotonic() + postview_s
        while time.monotonic() < end_at and cap.isOpened():
            ok, current_frame = cap.read()
            if not ok or current_frame.shape != frame_shape:
                break
            last_frame = current_frame
            label, command, stage_elapsed = monitor.snapshot()
            canvas = draw_live_frame(current_frame, outline, H, source, drop_board,
                                     label, command, stage_elapsed,
                                     time.monotonic()-start, stop_requested.is_set())
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                break
        if errors and not stop_requested.is_set():
            raise errors[0]
        return stop_requested.is_set()
    finally:
        if worker.is_alive():
            stop_requested.set()
            worker.join()  # wait for current command to settle; never leave worker running


def inspect_scene(frame, detector, H, R, t, work_mask, min_area, diagnostics=None):
    """Inspect ALL detected cubes even when next motion is not permitted.

    The return format is kept backwards compatible: candidate (eligible for
    immediate pickup), drop_occupied, observed_positions, status, references_ok.
    When provided, `diagnostics` receives *raw* detections and blockers for UI.
    This is important: `target=None` means 'cannot MOVE', not 'no cube seen'.
    """
    marker_centers = base.marker_centres(frame, detector)
    refs_ok, refs_message, errors = check_reference_markers(H, marker_centers)
    if not refs_ok and len(marker_centers) < MIN_VISIBLE_ARUCO:
        refs_message = (f'MARKERS: need any {MIN_VISIBLE_ARUCO} of 4; visible IDs ' +
                        str(sorted(marker_centers)))
    cubes, color_masks = find_four_color_cubes(frame, work_mask, min_area, H, detector)
    sources = []
    drop_occupied = False
    detected_board_positions = []
    drawn = []
    for cube in cubes:
        # Keep a visible box even for a candidate that fails geometric validation.
        entry = dict(cube)
        entry['role'] = 'REJECT'
        entry['reason'] = ''
        try:
            obj = target_from_cube(cube, H, R, t)
            board = np.asarray(obj['board_mm_top_estimate'], float)
            detected_board_positions.append(board)
            if np.linalg.norm(board - DROP_BOARD_MM) <= DROP_GUARD_RADIUS_MM:
                drop_occupied = True
                entry['role'] = 'DROP'
            else:
                sources.append(obj)
                entry['role'] = 'SOURCE'
        except (ValueError, RuntimeError, cv2.error) as exc:
            entry['reason'] = str(exc)
        drawn.append(entry)

    # Check the geometry of the source independently of marker visibility
    # and the status of the drop point, so it can still be highlighted.
    candidate = None
    unsafe_reason = ''
    if len(sources) == 1:
        candidate = sources[0]
        try:
            board = np.asarray(candidate['board_mm_top_estimate'], float)
            robot = np.asarray(candidate['robot_xy_top_estimate'], float)
            zone.check_workspace(board, 'Pickup', zone.PICK_MARGIN)
            zone.check_board_edge(board, 'Pickup')
            zone.check_robot_xy_envelope(robot, 'Pickup')
            if np.linalg.norm(board - DROP_BOARD_MM) < MIN_PICK_TO_DROP_MM:
                raise RuntimeError('Cube too close to DROP center')
        except (RuntimeError, ValueError) as exc:
            unsafe_reason = str(exc)
            candidate = None

    if not refs_ok:
        status = 'PAUSED: ' + refs_message
    elif drop_occupied:
        status = 'WAIT DROP: remove cube/object at destination'
    elif len(sources) != 1:
        status = f'SEARCHING: {len(sources)} pickup candidates (need 1)'
    elif unsafe_reason:
        status = 'BLOCKED: ' + unsafe_reason
    else:
        status = 'FOUND: ' + candidate['color'].upper()

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            'markers': sorted(marker_centers),
            'marker_error_mm': max(errors) if errors else None,
            'marker_message': refs_message,
            'cubes': drawn,
            'sources': len(sources),
            'drop_occupied': drop_occupied,
            'color_pixels': {color: int(cv2.countNonZero(m))
                             for color, m in color_masks.items()},
            'blocked_reason': status,
            'eligible': candidate is not None and refs_ok and not drop_occupied,
        })
    # Do not arm the robot when a blocker is present, even though the camera
    # can still draw the candidate(s) found above.
    if not refs_ok or drop_occupied:
        candidate = None
    return candidate, drop_occupied, detected_board_positions, status, refs_ok


def draw_search_frame(frame, outline, H, picked, status, progress,
                      hold_remaining, ncycles, diagnostics=None):
    canvas = frame.copy()
    cv2.polylines(canvas, [outline], True, (0, 0, 255), 2)
    invH = np.linalg.inv(H)
    dp = base.project(invH, [DROP_BOARD_MM])[0]
    dx, dy = tuple(np.rint(dp).astype(int))
    # Yellow ring visualizes the area that must stay empty for safe placement.
    theta = np.linspace(0, 2*np.pi, 41)
    ring_board = [DROP_BOARD_MM + DROP_GUARD_RADIUS_MM *
                  np.array([np.cos(v), np.sin(v)]) for v in theta]
    ring_px = np.rint(base.project(invH, ring_board)).astype(np.int32)
    cv2.polylines(canvas, [ring_px], True, (0, 220, 255), 1)
    cv2.drawMarker(canvas, (dx, dy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
    base.draw_text(canvas, 'DROP (2+3)', dx-54, max(100,dy-22), (0,255,255), .42)

    if diagnostics is not None:
        for obj in diagnostics.get('cubes', []):
            x, y, w, h = obj['bbox']
            role = obj['role']
            colour = (0, 255, 255) if role == 'DROP' else (
                (0, 0, 255) if role == 'REJECT' else COLOR_BGR[obj['color']])
            cv2.rectangle(canvas, (x,y), (x+w,y+h), colour, 2)
            base.draw_text(canvas, obj['color'].upper() + ' ' + role,
                           x, min(canvas.shape[0]-35,max(100,y-7)), colour, .42)
    elif picked is not None:
        x,y,w,h = picked['bbox']
        colour = COLOR_BGR[picked['color']]
        cv2.rectangle(canvas,(x,y),(x+w,y+h),colour,2)

    # Compact opaque panel; lines are separated so crucial reasons aren't
    # clipped, as they were in the previous version of the continuous loop.
    h,w = canvas.shape[:2]
    banner_height = 112
    cv2.rectangle(canvas, (0,0), (w-1,banner_height), (15,15,15),-1)
    # Break status into separate display rows with a small font.
    if status.startswith('PAUSED:'):
        first, second = 'PAUSED: marker check failed', status[len('PAUSED: '):]
    elif status.startswith('WAIT DROP'):
        first, second = 'WAIT DROP: destination occupied', 'Remove deposited cube / false detection'
    elif status.startswith('BLOCKED:'):
        first, second = 'BLOCKED: pickup point', status[len('BLOCKED: '):]
    elif status.startswith('WAIT CLEAR'):
        first, second = 'WAIT CLEAR: last cycle not reset', status[len('WAIT CLEAR:'):].strip()
    else:
        first, second = status, ''
    base.draw_text(canvas, first[:70], 6, 22, (0,220,255), .46)
    if second:
        base.draw_text(canvas, second[:100], 6, 42, (0,220,255), .34)
    if diagnostics:
        d=diagnostics
        marker_line = ('ARUCO ' + str(d.get('markers',[])) +
                       ' | raw cubes ' + str(len(d.get('cubes',[]))) +
                       ' | source ' + str(d.get('sources',0)) +
                       ' | drop ' + ('BUSY' if d.get('drop_occupied') else 'FREE'))
        base.draw_text(canvas, marker_line[:100], 6, 62, (230,230,230), .36)
    base.draw_text(canvas, f'{progress} | completed {ncycles}',
                   6, 83, (230,230,230), .38)
    if hold_remaining is not None:
        base.draw_text(canvas, f'Auto start in {hold_remaining:.1f}s',
                       6, 103, (100,255,255), .40)
    else:
        base.draw_text(canvas, 'Place ONE cube. Keep yellow DROP ring empty.',
                       6, 103, (220,220,220), .37)
    cv2.rectangle(canvas, (0,h-27), (w-1,h-1),(10,10,10),-1)
    base.draw_text(canvas,'ESC = exit (not emergency stop)', 5,h-9,
                   (255,255,255), .42)
    return canvas


def run_continuous(cap, frame_shape, outline, H, R, t, detector, mask,
                   min_area, execute, port, pace):
    """One process, uninterrupted live window, many cycles until ESC.

    No auto-retry on robot errors. After each cycle the workspace must be
    observed clear before rearming. Never trust firmware 'state' as an obstacle
    or gripper force sensor.
    """
    stable = deque(maxlen=STABLE_FRAMES)
    selected = None
    selected_since = None
    cycles = 0
    # After cycle, do not rearm until the previous source and drop are empty.
    awaiting_reset = False
    last_source = None
    reset_clear_frames = 0
    dry_reported = None
    last_status = ''
    last_report = 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print('CONTINUOUS mode: place one cube, await 1 second after stable detection.')
    print('After placement remove the cube from DROP before the next cycle.')
    print('Press Esc to stop; Esc is NOT an immediate motor emergency stop.', flush=True)
    while True:
        ok, frame = cap.read()
        if not ok or frame.shape != frame_shape:
            raise RuntimeError('Camera lost / resolution changed; no new movement')
        diagnostics = {}
        target, drop_occupied, observed_positions, status, refs_ok = inspect_scene(
            frame, detector, H, R, t, mask, min_area, diagnostics=diagnostics)
        # Post-cycle interlock: no new pickup until previous source has disappeared
        # and the last deposited cube is removed from the DROP zone.
        if awaiting_reset:
            previous_still_at_source = any(
                float(np.linalg.norm(v - last_source)) <= SOURCE_GUARD_RADIUS_MM
                for v in observed_positions)
            if not refs_ok or drop_occupied or previous_still_at_source:
                reset_clear_frames = 0
            else:
                reset_clear_frames += 1
            if reset_clear_frames < RESET_CLEAR_FRAMES:
                if not refs_ok:
                    status = 'WAIT CLEAR: marker check failed'
                elif drop_occupied:
                    status = 'WAIT CLEAR: drop occupied'
                elif previous_still_at_source:
                    status = 'WAIT CLEAR: previous pickup spot still occupied'
                else:
                    status = (f'RESETTING: verifying clear '
                              f'{reset_clear_frames}/{RESET_CLEAR_FRAMES}')
                target = None
            else:
                awaiting_reset = False
                reset_clear_frames = 0
                stable.clear()
                selected = None
                selected_since = None
                # A new target must pass stable detection FROM THIS point onward.
                target = None
                status = 'READY: waiting for next new cube'
        if target is None:
            stable.clear()
            selected = None
            selected_since = None
            dry_reported = None
        else:
            board = np.asarray(target['board_mm_top_estimate'], float)
            if stable and (stable[-1]['color'] != target['color'] or
                           np.linalg.norm(board - np.asarray(
                               stable[-1]['board_mm_top_estimate'], float)) > MAX_CUBE_JITTER_MM):
                stable.clear()
                selected = None
                selected_since = None
            stable.append(target)
            if len(stable) == STABLE_FRAMES:
                positions = np.asarray([v['board_mm_top_estimate'] for v in stable], float)
                median = np.median(positions, axis=0)
                if np.max(np.linalg.norm(positions - median, axis=1)) <= MAX_CUBE_JITTER_MM:
                    if selected_since is None:
                        selected_since = time.monotonic()
                    selected = stable[-1].copy()
                    selected['board_mm_top_estimate'] = list(map(float, median))
                    pxy = R @ median + t
                    selected['robot_xy_top_estimate'] = list(map(float, pxy))
                    status = 'FOUND: ' + selected['color'].upper()
                else:
                    stable.clear()
                    selected = None
                    selected_since = None
        remaining = (max(0.0, DETECTION_CONFIRM_S - (time.monotonic() - selected_since))
                     if selected_since is not None else None)
        progress = f'Stable: {len(stable)}/{STABLE_FRAMES}'
        if not execute and selected is not None and remaining == 0.0:
            progress = 'DRY RUN READY (no motor commands)'
        # Print blockers and observed cube count periodically, to distinguish
        # actual color detection problems from safety interlocks.
        now = time.monotonic()
        report = (status, len(diagnostics['cubes']), diagnostics['sources'],
                  diagnostics['drop_occupied'], tuple(diagnostics['markers']))
        if report != last_status and now - last_report > 0.8:
            print('VISION:', status, '| markers:', diagnostics['markers'],
                  '| raw color candidates:', len(diagnostics['cubes']),
                  '| pickup candidates:', diagnostics['sources'],
                  '| drop occupied:', diagnostics['drop_occupied'], flush=True)
            last_status, last_report = report, now
        canvas = draw_search_frame(frame, outline, H, target, status,
                                   progress, remaining, cycles, diagnostics=diagnostics)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            print('ESC: loop ended; no new motion.', flush=True)
            return cycles
        if selected is None or remaining > 0:
            continue
        try:
            pick_xy, drop_board, dst_robot = validate_static_targets(selected, R, t)
        except (RuntimeError, ValueError) as exc:
            # Do not move on an unsafe candidate; keep scanning.
            print('Candidate rejected:', exc, flush=True)
            stable.clear(); selected = None; selected_since = None
            continue
        if not execute:
            if dry_reported != (selected['color'], tuple(round(x) for x in
                              selected['board_mm_top_estimate'])):
                print('DRY RUN: target', selected['color'],
                      'board=', np.round(selected['board_mm_top_estimate'], 1).tolist(),
                      'robot XY=', np.round(pick_xy, 1).tolist(), flush=True)
                dry_reported = (selected['color'], tuple(round(x) for x in
                                selected['board_mm_top_estimate']))
            continue
        # Last camera frame before any serial movement is a CURRENT, valid
        # frame containing exactly one non-drop source and no detected cube on DROP.
        if target is None or target['color'] != selected['color'] or (
            np.linalg.norm(np.asarray(target['board_mm_top_estimate'],float) -
                           np.asarray(selected['board_mm_top_estimate'],float)) > 5.0):
            stable.clear(); selected = None; selected_since = None
            continue
        print('\n'+'='*62)
        print('AUTO CYCLE', cycles + 1, '|', selected['color'], '| pickup board',
              np.round(selected['board_mm_top_estimate'],1).tolist(),
              '| place board', np.round(DROP_BOARD_MM,1).tolist(), flush=True)
        source_xy = np.asarray(selected['board_mm_top_estimate'], float)
        # The worker executes one checked physical sequence; live video stays on.
        stopped = run_with_live_camera(cap, frame_shape, outline, H, selected,
                                       drop_board, port, dst_robot, R, t,
                                       return_up=True, pace=pace, postview_s=0)
        if stopped:
            print('ESC: stop requested during the robot cycle.', flush=True)
            return cycles
        cycles += 1
        print(f'CYCLE {cycles} complete; waiting for user to remove placed cube.', flush=True)
        awaiting_reset = True
        last_source = source_xy
        reset_clear_frames = 0
        stable.clear(); selected = None; selected_since = None


def main(argv=None):
    ap = argparse.ArgumentParser(description='CONTINUOUS LIVE 4-color RoboArm R1 pick-and-place until ESC')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--port', default=None, help='e.g. /dev/ttyUSB0')
    ap.add_argument('--execute', action='store_true', help='REALLY move motors on each detected cube')
    ap.add_argument('--pace', choices=sorted(PACE_PROFILES), default='fast')
    ap.add_argument('--board', type=Path, default=Path('board_calibration.npz'))
    ap.add_argument('--fit', type=Path, default=Path('robot_alignment_far_DIAGNOSTIC.npz'))
    ap.add_argument('--min-area', type=float, default=150.0)
    args = ap.parse_args(argv)
    if args.execute and not args.port:
        ap.error('For automatic motor motion specify --port /dev/ttyUSB0')
    if args.min_area < 50:
        ap.error('Invalid min-area setting')
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open camera {args.camera}')
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Camera did not return a frame')
        H, _ = base.load_board(args.board, frame.shape)
        R, t, source = base.load_alignment(args.fit, args.execute)
        mask, outline = make_work_mask(H, frame.shape)
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())
        if args.execute:
            print('ARMED: --execute enabled. Keep physical servo power-off accessible.')
        print('Board calibration:', args.board, '| Robot alignment:', source)
        print('Drop fixed at board XY', DROP_BOARD_MM.tolist())
        print('ArUco safety check: ANY', MIN_VISIBLE_ARUCO, 'markers, max drift', MAX_MARKER_DRIFT_MM, 'mm')
        print('Pickup Z=15, place Z=25, carry Z=85, pace=',args.pace)
        n = run_continuous(cap, frame.shape, outline, H, R, t, detector,
                           mask, args.min_area, args.execute, args.port, args.pace)
        print('Stopped normally; completed cycles:', n, flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print('STOPPED:', exc, file=sys.stderr)
        print('IMPORTANT: partial movement MAY have occurred. Inspect physical robot; '
              'never automatically retry after an error.', file=sys.stderr)
        raise SystemExit(1)
