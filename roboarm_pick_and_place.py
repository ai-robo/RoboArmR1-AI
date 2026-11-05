#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboArm R1: pick one cube, place near ArUco 2 or 3.

Stages OPEN, XY, DOWN (pickup Z=15), GRIP, UP, MOVE,
DOWN (placement Z=25), OPEN. Each requires terminal confirmation.

Pick, place and travel are constrained to the NEW user-marked rounded
working polygon. Coordinates are in board millimetres: defined using
four measured ArUco markers and the attached annotated camera screenshot.
This area is a REQUESTED test area, not a physically verified workspace.
Calibration was fitted only near original FAR references (1..4).

Requires existing roboarm_move_to_cube.py, roboarm_pickup_cube.py,
and saved .npz calibrations. Default is DRY RUN.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import roboarm_move_to_cube as base
import roboarm_pickup_cube as grip

# Measured ArUco centres: ID2=(105,340), ID3=(305,345) mm (BOARD coordinates).
# Keep placement spots next to tags to preserve ArUco visibility.
# Do not place directly ON printed marker. Both points are inside RED zone.
DROP_BOARD = {
    2: np.array([170.0, 335.0]),   # beside bottom-left ArUco ID2
    3: np.array([248.0, 329.0]),   # beside bottom-right ArUco ID3
}
# Workspace approximates the RED rounded contour drawn by the user on
# Screenshot from 2026-10-31 20-53-12.png. NOT screenshot pixels: physical
# BOARD millimetres, derived using known ArUco centres on that screenshot.
# Polygon makes it independent of resizing the app window. Edit only after
# new physical verification. Clockwise from top-left, approx.
WORKSPACE_BOARD_MM = np.array([
    [115.0, 215.0],
    [278.0, 218.0],
    [293.0, 224.0],
    [303.0, 234.0],
    [309.0, 250.0],
    [309.0, 336.0],
    [304.0, 351.0],
    [295.0, 365.0],
    [115.0, 363.0],
    [102.0, 356.0],
    [94.0, 342.0],
    [89.0, 250.0],
    [94.0, 233.0],
    [103.0, 220.0],
], dtype=np.float32).reshape(-1, 1, 2)
# Small tolerance accounts for marker and cube center jitter only.
PICK_MARGIN = 0.0
DROP_MARGIN = 5.0
PATH_MARGIN = 0.0
BOARD_EDGE_MARGIN = 15.0
MAX_ROBOT_RADIUS_MM = 190.0  # approximate workspace sanity check, NOT collision proof
MIN_ROBOT_X_MM = 50.0       # avoid previously unreliable near-base zone
MAX_ABS_ROBOT_Y_MM = 110.0
MIN_OTHER_CUBE_MM = 30.0
MIN_PICK_TO_DROP_MM = 30.0
MAX_XY_COMMAND_MM = 60.0
TRAVEL_Z = 85.0
PICK_Z = 15.0
PLACE_Z = 25.0   # user-verified/set placement height, NOT the gripping height
OPEN_CLAW = 50.0
CLOSED_CLAW = 0.0
WINDOW = 'RoboArm - PICK then PLACE near ArUco 2 / 3'


def wait_pose(ser, expected_xyz, expected_claw, label, deadline_s=8.0):
    """Check firmware commanded-state; NEVER resend a go command."""
    goal = np.r_[np.asarray(expected_xyz, dtype=float), float(expected_claw)]
    limit = time.monotonic() + deadline_s
    while True:
        state = grip.checked_state(ser)
        if (np.max(np.abs(state[:3] - goal[:3])) <= 4.0 and
                abs(state[3] - goal[3]) <= 3.0):
            print(f'  STATE {label}: {state.round(1).tolist()} (not physical feedback)')
            return state
        if time.monotonic() >= limit:
            raise RuntimeError(
                f'{label}: reported {state.round(1).tolist()} instead of '
                f'{goal.round(1).tolist()}. STOP. Do NOT resend automatically.'
            )
        time.sleep(0.5)


def approve(token, text):
    print(f'\n{token}: {text}')
    print(f'If physical movement is safe, type exactly {token}; Enter cancels.')
    if input('> ').strip() != token:
        raise RuntimeError(f'Cancelled at {token}. No further commands.')


def send(ser, command, goal_xyz, goal_claw, label):
    print('  SENDING:', command)
    ser.write((command + '\n').encode('ascii'))
    ser.flush()
    time.sleep(1.6)
    return wait_pose(ser, goal_xyz, goal_claw, label)


def check_destination(marker_id, H, R, t, objs, src_board):
    if marker_id not in DROP_BOARD:
        raise ValueError('Choose ArUco 2 or 3')
    dst_board = DROP_BOARD[marker_id].copy()
    check_workspace(dst_board, f'DROP {marker_id}', DROP_MARGIN)
    if np.linalg.norm(dst_board - src_board) < MIN_PICK_TO_DROP_MM:
        raise RuntimeError('Pickup cube is too close to the proposed drop position.')
    for other in objs:
        other_board = base.project(H, [other['centre_px']])[0]
        if np.linalg.norm(other_board - src_board) < 15.0:
            continue  # likely the cube to pick
        if np.linalg.norm(other_board - dst_board) < MIN_OTHER_CUBE_MM:
            raise RuntimeError('A coloured object is too close to the drop position.')
    target_xy = R @ dst_board + t
    if not np.isfinite(target_xy).all():
        raise RuntimeError('Invalid drop robot coordinates.')
    check_robot_xy_envelope(target_xy, 'Drop')
    return dst_board, target_xy


def check_robot_xy_envelope(xy, label):
    """Rough coordinate gate: NOT a collision checker or IK guarantee."""
    x, y = map(float, np.asarray(xy))
    if not (np.isfinite([x, y]).all()
            and MIN_ROBOT_X_MM <= x <= MAX_ROBOT_RADIUS_MM
            and abs(y) <= MAX_ABS_ROBOT_Y_MM
            and math.hypot(x, y) <= MAX_ROBOT_RADIUS_MM):
        raise RuntimeError(f'{label}: RoboArm XY ({x:.1f}, {y:.1f}) outside '
                           'preliminary conservative coordinate envelope. '
                           'Do not move automatically.')


def check_board_edge(xy, label):
    """Exclude positions near physical plywood edges."""
    x, y = map(float, np.asarray(xy))
    w, h = base.BOARD_SIZE_MM
    if not (BOARD_EDGE_MARGIN <= x <= w - BOARD_EDGE_MARGIN
            and BOARD_EDGE_MARGIN <= y <= h - BOARD_EDGE_MARGIN):
        raise RuntimeError(f'{label}: too close to plywood edge.')


def workspace_margin_mm(board_xy):
    """Signed physical distance to the RED requested working polygon, in mm."""
    return float(cv2.pointPolygonTest(
        WORKSPACE_BOARD_MM, tuple(map(float, np.asarray(board_xy))), True
    ))


def workspace_outline_px(H):
    """Project the red workspace polygon into live camera pixels."""
    coords = base.project(np.linalg.inv(H), WORKSPACE_BOARD_MM.reshape(-1, 2))
    return np.rint(coords).astype(np.int32)


def check_workspace(board_xy, label, minimum_margin=PATH_MARGIN):
    dist = workspace_margin_mm(board_xy)
    if dist < minimum_margin - 1e-4:
        raise RuntimeError(f'{label} outside RED work area: signed distance '
                           f'{dist:+.1f} mm (required >= {minimum_margin:.1f} mm).')
    return dist


def verify_path_in_workspace(start_robot, end_robot, R, t):
    """Check entire commanded XY line using 5-mm sampled robot positions."""
    start = np.asarray(start_robot, dtype=float)
    end = np.asarray(end_robot, dtype=float)
    distance = float(np.linalg.norm(end - start))
    invR = np.linalg.inv(R)
    for alpha in np.linspace(0.0, 1.0, max(2, math.ceil(distance / 5.0) + 1)):
        robot_pt = start + alpha * (end - start)
        check_robot_xy_envelope(robot_pt, 'Travel')
        board_pt = invR @ (robot_pt - t)
        check_board_edge(board_pt, 'Travel')
        check_workspace(board_pt, 'Travel', PATH_MARGIN)



def xy_parts(start, end, max_length=MAX_XY_COMMAND_MM):
    """Break longer moves into manageable parts; work-area checks follow."""
    distance = float(np.linalg.norm(end - start))
    n = max(1, math.ceil(distance / max_length))
    return [(start + (end - start) * (i / n)) for i in range(1, n + 1)]


def safe_xy_path(start, end, R, t):
    current = np.array(start, dtype=float)
    stages = xy_parts(current, np.asarray(end, dtype=float))
    check_robot_xy_envelope(current, 'Path start')
    for subgoal in stages:
        check_robot_xy_envelope(subgoal, 'Path waypoint')
        if np.linalg.norm(subgoal - current) >= 0.2:
            verify_path_in_workspace(current, subgoal, R, t)
        current = subgoal
    return stages


def run_motion(args, source, marker_id, dst_board, dst_xy, R, t):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError('Install pyserial: pip install pyserial') from exc

    src_board = np.asarray(source['board_mm_top_estimate'], dtype=float)
    src_xy = np.asarray(source['robot_xy_top_estimate'], dtype=float)
    pick_clearance = check_workspace(src_board, 'Pickup', PICK_MARGIN)
    check_workspace(dst_board, 'Placement', DROP_MARGIN)
    check_board_edge(src_board, 'Pick')
    check_board_edge(dst_board, 'Drop')
    check_robot_xy_envelope(src_xy, 'Pick')
    check_robot_xy_envelope(dst_xy, 'Drop')

    print('\nOpening USB Serial can reset ESP32. Observe actual robot position.')
    with serial.Serial(args.port, baudrate=115200, timeout=0.2,
                       write_timeout=1.5) as ser:
        time.sleep(2.2)
        state = grip.checked_state(ser)
        grip.near(state[2], TRAVEL_Z, 3.0, 'Initial Z')
        approach_parts = safe_xy_path(state[:2], src_xy, R, t)
        move_parts = safe_xy_path(src_xy, dst_xy, R, t)

        print('\n--- PREVIEW / NO COMMANDS YET ---')
        print('Pick cube:', source['color'], 'board', src_board.round(1).tolist(),
              'robot', src_xy.round(1).tolist())
        print(f'Place near ArUco ID {marker_id}: board', dst_board.tolist(),
              'robot', dst_xy.round(1).tolist())
        print('Existing robot state:', state.round(1).tolist())
        print('XY route approach segments:', len(approach_parts),
              '| MOVE segments:', len(move_parts))
        print('Pickup: Z=15; placement: Z=25 (new). Verify placement clearance physically.')
        print('CUBE MUST BE CLEARLY held at travel Z, without hitting the board on lowering.')
        print('The whole articulated arm must clear obstacles, not only the gripper tip.')
        print(f'Pickup distance to RED region boundary: {pick_clearance:+.1f} mm.')
        if base.far_margin_mm(src_board) < 0:
            print('WARNING: pickup outside old 1..4 calibration hull. '
                  'XY/Z accuracy unverified: check physical position before DOWN.')
        print('User-marked RED area is NOT a safety-certified robot workspace.')
        print('Stages: OPEN - XY - DOWN(Z15) - GRIP - UP - MOVE - DOWN(Z25) - OPEN.')
        print('No automatic recovery/retry; no retract after last OPEN.')
        approve('OPEN', '1/8 Open fingers at Z=85; check hands clear.')
        send(ser, 'go claw=50', state[:3], OPEN_CLAW, 'OPEN')

        approve('XY', f'2/8 Travel above cube to robot XY={src_xy.round(1)} at Z=85. '
                'Verify entire arm sweep. Multiple short go commands possible.')
        now = grip.checked_state(ser)
        grip.near(now[2], TRAVEL_Z, 3.0, 'Z before approach')
        # State might change while user approves; verify again.
        approach_parts = safe_xy_path(now[:2], src_xy, R, t)
        for j, subgoal in enumerate(approach_parts, 1):
            now = grip.checked_state(ser)
            if np.linalg.norm(now[:2] - subgoal) < 0.2:
                continue
            safe_xy_path(now[:2], subgoal, R, t)
            print(f'  Approach segment {j}/{len(approach_parts)}')
            send(ser, f'go x={subgoal[0]:.1f} y={subgoal[1]:.1f}',
                 np.r_[subgoal, TRAVEL_Z], OPEN_CLAW, 'XY')

        approve('DOWN', '3/8 Check jaws are directly over cube and empty space below. '
                'Lower from Z=85 to Z=15 (validated at pick locations used so far).')
        now = grip.checked_state(ser)
        if np.linalg.norm(now[:2] - src_xy) > 3:
            raise RuntimeError('XY mismatch before first DOWN')
        grip.near(now[3], OPEN_CLAW, 3.0, 'Claw before first DOWN')
        send(ser, 'go z=15', np.r_[src_xy, PICK_Z], OPEN_CLAW, 'DOWN pick')

        approve('GRIP', '4/8 Visually check jaws surround cube; close gently (claw=0).')
        now = grip.checked_state(ser)
        grip.near(now[2], PICK_Z, 3.0, 'Z before GRIP')
        send(ser, 'go claw=0', np.r_[src_xy, PICK_Z], CLOSED_CLAW, 'GRIP')

        approve('UP', '5/8 Confirm cube is physically held and lift route clear; raise to Z=85.')
        now = grip.checked_state(ser)
        grip.near(now[3], CLOSED_CLAW, 3.0, 'Claw before UP')
        send(ser, 'go z=85', np.r_[src_xy, TRAVEL_Z], CLOSED_CLAW, 'UP')

        approve('MOVE', f'6/8 Place NEAR marker {marker_id} at BOARD {dst_board.tolist()} '
                f'(not on the ArUco paper), ROBOT {dst_xy.round(1)}. '
                'Check cube held and full sweep clear.')
        now = grip.checked_state(ser)
        grip.near(now[2], TRAVEL_Z, 3.0, 'Z before MOVE')
        grip.near(now[3], CLOSED_CLAW, 3.0, 'Claw before MOVE')
        move_parts = safe_xy_path(now[:2], dst_xy, R, t)
        for j, subgoal in enumerate(move_parts, 1):
            now = grip.checked_state(ser)
            grip.near(now[2], TRAVEL_Z, 3.0, 'Z during MOVE')
            grip.near(now[3], CLOSED_CLAW, 3.0, 'Claw during MOVE')
            safe_xy_path(now[:2], subgoal, R, t)
            if np.linalg.norm(now[:2] - subgoal) < 0.2:
                continue
            print(f'  MOVE segment {j}/{len(move_parts)}')
            send(ser, f'go x={subgoal[0]:.1f} y={subgoal[1]:.1f}',
                 np.r_[subgoal, TRAVEL_Z], CLOSED_CLAW, 'MOVE')

        approve('DOWN', f'7/8 CHECK EMPTY AREA near marker {marker_id}. '
                'Place Z=25 mm (NOT pickup Z=15). Lower only if physically safe; '
                'cube must not press into board or ArUco paper.')
        now = grip.checked_state(ser)
        if np.linalg.norm(now[:2] - dst_xy) > 3:
            raise RuntimeError('XY mismatch before drop DOWN')
        grip.near(now[2], TRAVEL_Z, 3, 'Z before drop DOWN')
        send(ser, 'go z=25', np.r_[dst_xy, PLACE_Z], CLOSED_CLAW, 'DOWN place')

        approve('OPEN', f'8/8 Confirm cube is resting on board and fingers free to open '
                f'near marker {marker_id}; OPEN claw=50.')
        now = grip.checked_state(ser)
        grip.near(now[2], PLACE_Z, 3.0, 'Z before release')
        send(ser, 'go claw=50', np.r_[dst_xy, PLACE_Z], OPEN_CLAW, 'OPEN release')

        print('\nFINISHED. Check cube was actually released next to ID', marker_id)
        print('Robot remains at Z=25, claw=50 by design (no final UP requested).')
        print('Do not manually force powered servos; use tested commands for retraction.')


def main():
    ap = argparse.ArgumentParser(description='One RoboArm pick-and-place within RED region near ArUco 2 or 3.')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--port', default=None)
    ap.add_argument('--enable-motion', action='store_true')
    ap.add_argument('--board', type=Path, default=Path('board_calibration.npz'))
    ap.add_argument('--fit', type=Path, default=Path('robot_alignment_far_DIAGNOSTIC.npz'))
    ap.add_argument('--min-area', type=float, default=150.0)
    args = ap.parse_args()
    if args.enable_motion and not args.port:
        ap.error('Motion requires --port /dev/ttyUSB0')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Camera {args.camera} not available')
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError('Camera returned no image')

    try:
        H, _centres = base.load_board(args.board, frame.shape)
        R, t, source = base.load_alignment(args.fit, args.enable_motion)
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())
        work_px = workspace_outline_px(H)
        # Detect cubes only in red workspace rather than the old yellow ArUco ROI.
        work_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(work_mask, [work_px], 255)
        work_mask = cv2.erode(work_mask, np.ones((5, 5), np.uint8), iterations=1)

        chosen_color = 'green'
        chosen_marker = 2
        frozen = False
        freeze_frame = None
        candidate = None
        already_tried = False
        message = 'G green / R red; 2 or 3 destination; SPACE freeze; P preview; M execute'
        print('Calibration alignment:', source)
        print(f'RED working polygon: pickup >= {PICK_MARGIN} mm inside; '
              f'drop >= {DROP_MARGIN} mm inside. Not verified for all robot poses.')
        print('Pickup Z=15; placement Z=25; travel Z=85; each step requires approval.')
        print('NO MOVEMENT unless --enable-motion; every step needs terminal approval.')
        print('Keys: G/R select cube, 2/3 choose drop marker, SPACE freeze, P preview, M run, Q quit')
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

        while True:
            if frozen:
                shown = freeze_frame.copy()
            else:
                ok, shown = cap.read()
                if not ok:
                    raise RuntimeError('Camera disconnected')
                if shown.shape != frame.shape:
                    raise RuntimeError('Camera resolution changed; recalibrate')
            found = base.marker_centres(shown, detector)
            ref_ok, ref_text, _ = base.marker_check(H, found)
            cubes = base.find_cubes(shown, work_mask, args.min_area)
            canvas = shown.copy()
            cv2.polylines(canvas, [work_px], True, (0, 0, 255), 3)

            for mid, point in DROP_BOARD.items():
                uv = base.project(np.linalg.inv(H), [point])[0]
                pos = tuple(np.rint(uv).astype(int))
                cv2.circle(canvas, pos, 12, (0, 255, 255) if chosen_marker == mid else (255, 230, 170), 2)
                base.draw_text(canvas, f'DROP {mid}', pos[0]+12, max(18,pos[1]-15),
                               (0, 255, 255), .5)
            for cube in cubes:
                x,y,w,h = cube['bbox']
                cv2.rectangle(canvas, (x,y),(x+w,y+h),
                              (0,220,0) if cube['color']=='green' else (0,0,240), 2)
                base.draw_text(canvas, cube['color'], x, max(25,y-7), scale=.45)
            base.draw_text(canvas, f'cube:{chosen_color} -> marker:{chosen_marker} '
                           f'| {"FROZEN" if frozen else "LIVE"} '
                           f'| {"MOTION" if args.enable_motion else "DRY RUN"}',
                           8, 22, (0, 255, 255), .45)
            base.draw_text(canvas, ref_text, 8, 44,
                           (65, 230, 65) if ref_ok else (0, 0, 255), .43)
            base.draw_text(canvas, message[:100], 8, canvas.shape[0]-14, scale=.4)
            cv2.imshow(WINDOW, canvas)
            k = cv2.waitKey(25) & 0xFF
            if k in (27, ord('q')):
                break
            elif k in (ord('g'), ord('r')):
                chosen_color = 'green' if k == ord('g') else 'red'
                candidate = None
                message = f'Chosen cube {chosen_color}'
            elif k in (ord('2'), ord('3')):
                chosen_marker = int(chr(k))
                candidate = None
                message = f'Place NEAR marker ID {chosen_marker}'
            elif k == ord(' '):
                frozen = not frozen
                freeze_frame = shown.copy() if frozen else None
                candidate = None
                message = 'Frozen: press P' if frozen else 'Live image'
            elif k == ord('p'):
                if not frozen or not ref_ok:
                    message = 'Freeze and ensure stable ArUco reference (>=3)' 
                else:
                    try:
                        # base.select_target calculates board and robot coordinates;
                        # its OLD FAR boundary is no longer a blocking condition.
                        target = base.select_target(cubes, chosen_color, H, R, t, -1000.0)
                        check_workspace(target['board_mm_top_estimate'], 'Pick', PICK_MARGIN)
                        check_board_edge(target['board_mm_top_estimate'], 'Pick')
                        check_robot_xy_envelope(target['robot_xy_top_estimate'], 'Pick')
                        dst_board, dst_xy = check_destination(
                            chosen_marker, H, R, t, cubes,
                            np.asarray(target['board_mm_top_estimate']))
                        candidate = (target, chosen_marker, dst_board, dst_xy)
                        print('\nPREVIEW ONLY:')
                        print('Pick', chosen_color, 'robot XY', target['robot_xy_top_estimate'])
                        clearance = workspace_margin_mm(target['board_mm_top_estimate'])
                        print(f'RED work margin: {clearance:+.1f} mm.')
                        if base.far_margin_mm(target['board_mm_top_estimate']) < 0:
                            print('WARNING: outside old 4-point calibration hull. '
                                  'Check physical accuracy and descend only after inspection.')
                        print('Place near ArUco', chosen_marker,
                              '| board', dst_board.tolist(),
                              '| robot XY', dst_xy.round(1).tolist())
                        print('Stages: OPEN -> XY -> DOWN -> GRIP -> UP -> MOVE -> DOWN -> OPEN')
                        message = 'PREVIEW OK in RED work area. Press M when physically safe.' 
                    except (RuntimeError, ValueError) as e:
                        candidate = None
                        message = f'PREVIEW BLOCKED: {e}'
                        print(message)
            elif k == ord('m'):
                if already_tried:
                    message = 'One physical attempt per launch. Restart app.'
                elif not args.enable_motion:
                    message = 'DRY RUN. Nothing sent to robot.'
                    print(message)
                elif not frozen or candidate is None or not ref_ok:
                    message = 'First freeze, preview (P), and check ArUco.'
                else:
                    already_tried = True
                    try:
                        run_motion(args, *candidate, R, t)
                        message = 'Sequence finished. Inspect robot and cube.'
                    except (Exception, KeyboardInterrupt) as e:
                        message = f'STOPPED: {e}'
                        print(message)
                        print('Partial movement may have occurred. Check physically; no retry.')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        raise SystemExit(1)
