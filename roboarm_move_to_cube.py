#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboArm R1: first camera-guided *approach* to a red or green cube.

Runs DRY-RUN by default. In movement mode, sends at most ONE single
`go x=... y=...` command per process (Z and claw are NOT commanded).

Movement requires:
  --enable-motion --port PORT --travel-z VERIFIED_Z --confirm-xy-checks
  frozen camera snapshot, selected unique cube, enough reference markers,
  far-zone checks, live `state`, small step, terminal confirmation MOVE.

It deliberately does not close the claw, descend, or plan complex paths.
"""

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Physical centres of ArUco markers (mm) on 410 x 480 mm board.
MARKER_BOARD_MM = {
    0: (100.0, 145.0), 1: (310.0, 145.0),
    2: (105.0, 340.0), 3: (305.0, 345.0),
}
BOARD_SIZE_MM = (410.0, 480.0)

# Only a conservative portion of the 1..4 reference-point hull is allowed.
FAR_REF = np.array([
    [199.8, 255.4], [198.2, 354.2],
    [148.0, 348.0], [276.1, 338.6],
], dtype=np.float32)
FAR_HULL = cv2.convexHull(FAR_REF).reshape(-1, 1, 2)

# These numbers are for diagnostic preview ONLY when no saved V3 file exists.
R_PREVIEW = np.array([[-0.055738, 0.998445],
                      [0.998445, 0.055738]], dtype=np.float64)
T_PREVIEW = np.array([-157.34, -213.00], dtype=np.float64)

WINDOW = 'RoboArm R1 - camera-guided approach (one move max)'


def project(H, xy):
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(coords, H).reshape(-1, 2)


def marker_centres(frame, detector):
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    found = {}
    if ids is not None:
        for number, arr in zip(ids.flatten(), corners):
            if int(number) in MARKER_BOARD_MM:
                found[int(number)] = arr[0].mean(axis=0).astype(float)
    return found


def load_board(path, frame_shape):
    if not path.is_file():
        raise RuntimeError(f'Board calibration file not found: {path}')
    with np.load(path, allow_pickle=False) as f:
        H = np.asarray(f['H_pixel_to_board'], dtype=float)
        expected = (int(f['frame_height_px']), int(f['frame_width_px']))
        saved_centres = np.asarray(f['marker_centres_px'], dtype=float)
        board_w = float(f['board_width_mm'])
        board_h = float(f['board_height_mm'])
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise RuntimeError('Invalid board homography.')
    if expected != tuple(frame_shape[:2]):
        raise RuntimeError(f'Camera resolution differs from saved calibration: '
                           f'{frame_shape[:2]} vs {expected}')
    if saved_centres.shape != (4, 2) or not np.isfinite(saved_centres).all():
        raise RuntimeError('Saved ArUco centres are missing/invalid.')
    if abs(board_w - 410.0) > 0.01 or abs(board_h - 480.0) > 0.01:
        raise RuntimeError('Unexpected board dimensions in calibration file.')
    return H, saved_centres


def load_alignment(path, for_motion):
    if not path.is_file():
        if for_motion:
            raise RuntimeError(f'For motion the V3 alignment file is REQUIRED: {path}')
        print('WARNING: V3 alignment file missing; PREVIEW-only fallback will be used.')
        return R_PREVIEW.copy(), T_PREVIEW.copy(), 'FALLBACK / PREVIEW'
    with np.load(path, allow_pickle=False) as f:
        R = np.asarray(f['R_board_to_robot'], dtype=float)
        t = np.asarray(f['t_board_to_robot_mm'], dtype=float)
        if for_motion:
            required = ('preliminary_pass', 'rms_error_mm', 'max_error_mm', 'fit_ids')
            if any(key not in f for key in required):
                raise RuntimeError('Incomplete V3 diagnostic file.')
            if not bool(f['preliminary_pass']):
                raise RuntimeError('V3 calibration does not have PRELIMINARY PASS.')
            if float(f['rms_error_mm']) > 8.0 or float(f['max_error_mm']) > 12.0:
                raise RuntimeError('V3 calibration errors exceed limits.')
            if set(map(int, f['fit_ids'].tolist())) != {1, 2, 3, 4}:
                raise RuntimeError('V3 fit points must be exactly 1,2,3,4.')
    if (R.shape != (2, 2) or t.shape != (2,) or
            not np.isfinite(R).all() or not np.isfinite(t).all()):
        raise RuntimeError('Invalid V3 alignment coefficients.')
    if np.max(np.abs(R.T @ R - np.eye(2))) > 0.02 or abs(abs(np.linalg.det(R)) - 1) > 0.02:
        raise RuntimeError('Alignment R is not approximately isometric.')
    return R, t, str(path)


def find_cubes(frame, mask, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_ranges = {
        'green': [((35, 65, 55), (90, 255, 255))],
        'red': [((0, 90, 60), (12, 255, 255)),
                ((170, 90, 60), (179, 255, 255))],
    }
    objs = []
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for color, ranges in hsv_ranges.items():
        binary = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            binary |= cv2.inRange(hsv, np.array(low, dtype=np.uint8),
                                  np.array(high, dtype=np.uint8))
        binary = cv2.bitwise_and(binary, mask)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue
            moments = cv2.moments(cnt)
            if moments['m00'] <= 1e-8:
                continue
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            x, y, w, h = cv2.boundingRect(cnt)
            objs.append({'color': color, 'centre_px': [float(cx), float(cy)],
                         'bbox': (x, y, w, h), 'area': area})
    return objs


def draw_text(frame, msg, x, y, colour=(255, 255, 255), scale=0.5):
    cv2.putText(frame, msg, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, 1, cv2.LINE_AA)


def far_margin_mm(board_xy):
    return cv2.pointPolygonTest(FAR_HULL, tuple(map(float, board_xy)), True)


def marker_check(H, found, min_markers=3, max_mm_error=10.0):
    if len(found) < min_markers:
        return False, f'Only {len(found)} ArUco (need >= {min_markers})', None
    err = []
    for marker_id, uv in found.items():
        calc = project(H, [uv])[0]
        expected = np.asarray(MARKER_BOARD_MM[marker_id], dtype=float)
        err.append(float(np.linalg.norm(calc - expected)))
    if not np.isfinite(err).all() or max(err) > max_mm_error:
        return False, f'ArUco moved / camera changed: max ref error={max(err):.1f} mm', err
    return True, f'ArUco reference OK (max {max(err):.1f} mm)', err


def select_target(objs, chosen_color, H, R, t, safety_margin):
    if chosen_color not in ('red', 'green'):
        raise ValueError('Select a color with G or R first.')
    options = [item for item in objs if item['color'] == chosen_color]
    if len(options) != 1:
        raise ValueError(f'Expected exactly ONE {chosen_color} cube; found {len(options)}. '
                         'Temporarily remove other same-color objects.')
    target = options[0]
    bx, by = project(H, [target['centre_px']])[0]
    if not (0 <= bx <= BOARD_SIZE_MM[0] and 0 <= by <= BOARD_SIZE_MM[1]):
        raise ValueError('Detected point lies beyond the plywood board.')
    margin = far_margin_mm((bx, by))
    if margin < safety_margin:
        raise ValueError(f'Cube is too close to far-zone boundary: {margin:.1f} mm '
                         f'(require >= {safety_margin:.1f} mm).')
    robot = R @ np.array([bx, by], dtype=float) + t
    if not np.isfinite(robot).all():
        raise ValueError('Non-finite robot coordinates.')
    return {'color': target['color'], 'pixel': [round(v, 1) for v in target['centre_px']],
            'board_mm_top_estimate': [round(float(bx), 2), round(float(by), 2)],
            'robot_xy_top_estimate': [round(float(robot[0]), 2), round(float(robot[1]), 2)],
            'far_boundary_margin_mm': round(float(margin), 2),
            'warning': '20mm cube TOP viewed by overhead camera; NOT corrected to base centre'}


# Match firmware's `state` reply, documented as x=100 y=50 z=40 claw=20.
FLOAT = r'[-+]?\d+(?:\.\d+)?'
STATE_RE = re.compile(rf'\bx\s*=\s*({FLOAT})\s+y\s*=\s*({FLOAT})'
                      rf'\s+z\s*=\s*({FLOAT})\b', re.IGNORECASE)


def read_robot_state(ser, timeout_s=4.0):
    ser.reset_input_buffer()
    ser.write(b'state\n')
    ser.flush()
    started = time.monotonic()
    lines = []
    while time.monotonic() - started < timeout_s:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode('utf-8', errors='replace').strip()
        if line:
            lines.append(line)
            match = STATE_RE.search(line)
            if match:
                vals = np.asarray([float(v) for v in match.groups()])
                if np.isfinite(vals).all():
                    return vals, lines
    raise RuntimeError('No valid state XYZ from RoboArm. Serial log: ' + repr(lines[-12:]))


def verify_small_far_motion(state, target_robot, R, t, max_step, margin, z_expected):
    if abs(float(state[2]) - z_expected) > 3.0:
        raise RuntimeError(f'Robot state Z={state[2]:.1f} mm but verified travel Z '
                           f'is {z_expected:.1f} mm. Move manually to a SAFE Z first.')
    distance = float(np.linalg.norm(target_robot - state[:2]))
    if distance < 0.2:
        raise RuntimeError('Already at target (within 0.2 mm); movement unnecessary.')
    if distance > max_step:
        raise RuntimeError(f'Proposed XY movement {distance:.1f} mm > {max_step:.1f} mm '
                           'limit. Stage robot manually closer at verified travel Z.')
    # Approximate XY segment only: does NOT guarantee collision-free robot arm sweep.
    from_robot = np.linalg.inv(R)
    for percent in np.linspace(0.0, 1.0, 16):
        pos_robot = state[:2] * (1-percent) + target_robot * percent
        pos_board = from_robot @ (pos_robot - t)
        clearance = far_margin_mm(pos_board)
        if clearance < margin:
            raise RuntimeError(f'Planned XY segment exits approved far-zone interior '
                               f'(clearance {clearance:.1f} mm < {margin:.1f} mm). '
                               'Reposition robot manually to a staging point inside the zone.')
    return distance


def do_one_move(port, baud, target, R, t, z_ok, max_step, margin):
    try:
        import serial
    except ImportError:
        raise RuntimeError('pyserial is required: python3 -m pip install pyserial')
    print('\nOpening serial port. WARNING: opening ESP32 USB Serial may RESET the controller.')
    with serial.Serial(port=port, baudrate=baud, timeout=0.2, write_timeout=1.0) as ser:
        time.sleep(2.2)
        state, lines = read_robot_state(ser)
        x, y, z = map(float, state)
        xy = np.asarray(target['robot_xy_top_estimate'], dtype=float)
        dist = verify_small_far_motion(state, xy, R, t, max_step, margin, z_ok)
        cmd = f'go x={xy[0]:.1f} y={xy[1]:.1f}'
        print('Robot reports: x=%.1f y=%.1f z=%.1f mm' % (x, y, z))
        print('One proposed command:', cmd)
        print('XY travel distance: %.1f mm. Z/claw NOT changed.' % dist)
        print('YOU must check the entire path is clear, including the robot links.')
        print('The model does not know actual gripper position; state may be commanded pose.')
        response = input('If physically safe, type EXACTLY MOVE in terminal (otherwise Enter): ').strip()
        if response != 'MOVE':
            print('CANCELLED: nothing sent to robot.')
            return False
        # Recheck state immediately before sending, in case a different controller moved it.
        state_now, _ = read_robot_state(ser)
        if np.linalg.norm(state_now - state) > 2.5:
            raise RuntimeError('Robot state changed during confirmation; move cancelled.')
        ser.write((cmd + '\n').encode('ascii'))
        ser.flush()
        print('COMMAND SENT ONCE:', cmd)
        print('Arrival is NOT confirmed by Serial send. Visually verify tool position.')
        return True


def save_snapshot(folder, frame, view, target, found, motion_sent=False):
    folder.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime('approach_%Y%m%d_%H%M%S_%f')
    base = folder / stem
    cv2.imwrite(str(base) + '_raw.png', frame)
    cv2.imwrite(str(base) + '_annotated.png', view)
    with open(str(base) + '.json', 'w', encoding='utf-8') as stream:
        json.dump({'target': target, 'visible_markers': sorted(found),
                   'command_sent_this_session': bool(motion_sent),
                   'motion_status': 'sent once; physical arrival unverified' if motion_sent else 'not sent'},
                  stream, indent=2, ensure_ascii=False)
    print('Saved images and JSON:', base)


def main():
    ap = argparse.ArgumentParser(description='Camera-guided RoboArm R1: ONE confirmed approach, no grasp.')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--board', type=Path, default=Path('board_calibration.npz'))
    ap.add_argument('--fit', type=Path, default=Path('robot_alignment_far_DIAGNOSTIC.npz'))
    ap.add_argument('--min-area', type=float, default=150.0)
    ap.add_argument('--far-margin', type=float, default=10.0,
                    help='minimum distance inside far reference hull, mm (min allowed 10)')
    ap.add_argument('--max-step-mm', type=float, default=65.0,
                    help='maximum XY displacement for one command (max allowed 65)')
    ap.add_argument('--enable-motion', action='store_true')
    ap.add_argument('--serial-check', action='store_true',
                    help='read RoboArm state via Serial only; no camera or motion')
    ap.add_argument('--confirm-xy-checks', action='store_true',
                    help='confirm independent physical checks at A/C were completed')
    ap.add_argument('--port', help='serial port e.g. /dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--travel-z', type=float,
                    help='independently verified safe Z position for XY travel, in robot mm')
    ap.add_argument('--save-dir', type=Path, default=Path('cube_approach_runs'))
    args = ap.parse_args()
    if not (10 <= args.far_margin <= 25):
        ap.error('--far-margin must be 10..25 mm')
    if not (1 <= args.max_step_mm <= 65):
        ap.error('--max-step-mm must be 1..65 mm')
    if args.serial_check and args.enable_motion:
        ap.error('--serial-check cannot be combined with --enable-motion')
    if args.serial_check:
        if not args.port:
            ap.error('--serial-check requires --port')
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError('Install pyserial: python3 -m pip install pyserial') from exc
        print('Serial check only. Opening USB may reset ESP32; no go commands will be sent.')
        with serial.Serial(args.port, 115200, timeout=0.2, write_timeout=1.0) as ser:
            time.sleep(2.2)
            state, _ = read_robot_state(ser)
            print('RoboArm state: X=%.1f Y=%.1f Z=%.1f mm' % tuple(state))
        return
    if args.enable_motion:
        if not args.confirm_xy_checks:
            ap.error('--confirm-xy-checks is required for motion')
        if not args.port or args.travel_z is None or not math.isfinite(args.travel_z):
            ap.error('Motion requires --port and a manually VERIFIED --travel-z')
        if not (0 < args.travel_z < 250):
            ap.error('--travel-z outside conservative 0..250 mm range')
        if args.baud != 115200:
            ap.error('RoboArm R1 firmware expects baud 115200')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError('Camera opened but did not return any frame')
    H, centres = load_board(args.board, frame.shape)
    R, t, alignment_source = load_alignment(args.fit, args.enable_motion)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    roi_poly = np.rint(centres[[0, 1, 3, 2]]).astype(np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, roi_poly, 255)
    mask = cv2.erode(mask, np.ones((13, 13), dtype=np.uint8), iterations=1)

    print('\nCAMERA-GUIDED APPROACH - no grasp, no descent, no continuous control.')
    print('Alignment:', alignment_source)
    print('Default is DRY-RUN. Motion requested:', args.enable_motion)
    print('G/R select GREEN/RED, SPACE freeze/unfreeze, P preview, S save, M propose one move, Q exit.')
    selected_color = None
    frozen = False
    frozen_frame = None
    candidate = None
    message = 'G/R select color, SPACE freeze, P preview; M only with explicit motion flags'
    sent_once = False
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            if not frozen:
                ok, frame = cap.read()
                if not ok:
                    print('Camera disconnected.'); break
                if frame.shape[:2] != mask.shape:
                    print('Camera resolution changed. Stopping.'); break
            else:
                frame = frozen_frame.copy()
            found = marker_centres(frame, detector)
            ref_ok, ref_msg, _ = marker_check(H, found)
            objects = find_cubes(frame, mask, args.min_area)
            canvas = frame.copy()
            cv2.polylines(canvas, [roi_poly], True, (0, 255, 255), 2)
            far_img = project(np.linalg.inv(H), FAR_HULL.reshape(-1, 2))
            cv2.polylines(canvas, [np.rint(far_img).astype(np.int32)], True, (255, 140, 0), 2)
            for obj in objects:
                x, y, w, h = obj['bbox']
                px, py = obj['centre_px']
                colour = (20, 220, 20) if obj['color'] == 'green' else (20, 20, 230)
                if obj['color'] == selected_color:
                    colour = (255, 220, 0)
                cv2.rectangle(canvas, (x, y), (x+w, y+h), colour, 2)
                cv2.drawMarker(canvas, (round(px), round(py)), (0, 255, 255),
                               cv2.MARKER_CROSS, 14, 2)
                board = project(H, [(px, py)])[0]
                robot = R @ board + t
                draw_text(canvas, f'{obj["color"]}: board({board[0]:.0f},{board[1]:.0f}) '
                          f'robot?({robot[0]:.0f},{robot[1]:.0f})', max(2, x-9), max(85, y-10), scale=.42)
            mode = 'DRY' if not args.enable_motion else 'MOTION ARMED (gated)'
            draw_text(canvas, f'{mode} | {"FROZEN" if frozen else "LIVE"} | '
                      f'color:{selected_color or "NONE"} | markers:{sorted(found)}', 7, 21,
                      (0, 255, 255), .48)
            draw_text(canvas, ref_msg, 7, 43,
                      (60, 220, 60) if ref_ok else (0, 0, 255), .42)
            draw_text(canvas, message[:90], 7, canvas.shape[0]-18, (255, 255, 255), .43)
            if candidate is not None and frozen:
                draw_text(canvas,
                          f'TARGET {candidate["color"]} robot XY {candidate["robot_xy_top_estimate"]}',
                          7, 65, (60, 230, 255), .44)
            if sent_once:
                draw_text(canvas, 'ONE MOVE SENT - RESTART APP FOR ANOTHER', 7, 90, (0, 0, 255), .5)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(30) & 0xff
            if key in (ord('q'), 27):
                break
            if key in (ord('g'), ord('r')):
                selected_color = 'green' if key == ord('g') else 'red'
                candidate = None
                message = f'Selected {selected_color}; freeze with SPACE'
            elif key == ord(' '):
                frozen = not frozen
                frozen_frame = frame.copy() if frozen else None
                candidate = None
                message = 'FROZEN: P preview, M move proposal' if frozen else 'LIVE: change objects, then SPACE'
            elif key == ord('p'):
                if not frozen:
                    message = 'Freeze using SPACE before preview'
                elif not ref_ok:
                    message = 'BLOCKED: ' + ref_msg
                else:
                    try:
                        candidate = select_target(objects, selected_color, H, R, t, args.far_margin)
                        print('\nPROPOSED TARGET (NO MOVEMENT):', json.dumps(candidate, indent=2))
                        message = 'PREVIEW READY. Check frame and clearance. M requests move.'
                    except ValueError as exc:
                        message = str(exc)
                        print('Target rejected:', exc)
                        candidate = None
            elif key == ord('s'):
                save_snapshot(args.save_dir, frame, canvas, candidate, found, sent_once)
                message = 'Snapshot saved to ' + str(args.save_dir)
            elif key == ord('m'):
                if sent_once:
                    message = 'Move already sent. Restart app for another move.'
                elif not args.enable_motion:
                    message = 'DRY RUN: movement disabled. No command sent.'
                    print(message)
                elif not frozen or candidate is None:
                    message = 'Freeze image and press P to preview before M.'
                else:
                    # Do not trust old preview if live reference validation failed.
                    if not ref_ok:
                        message = 'BLOCKED: ' + ref_msg
                    else:
                        print('\nFinal preview:', json.dumps(candidate, ensure_ascii=False, indent=2))
                        try:
                            sent = do_one_move(args.port, args.baud, candidate, R, t,
                                               args.travel_z, args.max_step_mm, args.far_margin)
                            if sent:
                                sent_once = True
                                message = 'ONE GO COMMAND SENT. Visually check actual pose!'
                            else:
                                message = 'Cancelled. No movement.'
                        except (Exception, KeyboardInterrupt) as exc:
                            message = 'MOVE BLOCKED/FAILED: ' + str(exc)
                            print(message)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        raise SystemExit(1)
