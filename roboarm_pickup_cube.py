#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboArm R1 / OpenCV: guarded grip-and-lift trial within calibrated FAR zone.

Requires roboarm_move_to_cube.py (existing, tested approach program) alongside
this file, plus its two .npz calibration files. No automatic releases/drops.

DEFAULT: dry-run / camera preview only.
MOTION: explicit --enable-motion, --confirm-tested-grip and human confirmation
for EACH of OPEN, XY, DOWN, GRIP, UP. No repeated moves in one process.

The cube centre must lie within the calibrated FAR region (at least 10 mm
inside its border). Grip height Z=15 was tested at A, not throughout the
FAR region: visually check the entire descent before confirming DOWN.
Do NOT treat firmware 'state' as independent physical servo feedback.
"""
from __future__ import annotations

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

import roboarm_move_to_cube as base

WINDOW = 'RoboArm R1 - guarded FAR-zone pickup (one trial per launch)'
MAX_XY_STEP_MM = 65.0
BOARD_MARGIN_MM = 10.0
STATE_CLAW = re.compile(r'\bclaw\s*=\s*([-+]?\d+(?:\.\d+)?)\b', re.I)


def checked_state(ser):
    """Request current commanded pose; firmware is NOT metrology feedback."""
    xyz, raw = base.read_robot_state(ser, timeout_s=5.0)
    claw_val = None
    for line in raw:
        match = STATE_CLAW.search(line)
        if match:
            claw_val = float(match.group(1))
    if claw_val is None or not math.isfinite(claw_val):
        raise RuntimeError('RoboArm state did not include a valid claw=... value. No movement.')
    return np.r_[xyz, claw_val]


def wait_after_grip(ser, expected_xyz, expected_claw, max_wait_s=7.0):
    """Poll read-only state after GRIP; NEVER resend a motion command.

    Firmware state is commanded pose, not physical joint/force feedback.
    """
    deadline = time.monotonic() + max_wait_s
    last_pose = None
    attempts = 0
    while True:
        attempts += 1
        last_pose = checked_state(ser)
        xyz_err = np.max(np.abs(last_pose[:3] - expected_xyz))
        claw_err = abs(last_pose[3] - expected_claw)
        if xyz_err <= 4.0 and claw_err <= 3.0:
            print(f'Firmware state after GRIP (poll {attempts}): '
                  f'x={last_pose[0]:.1f}, y={last_pose[1]:.1f}, '
                  f'z={last_pose[2]:.1f}, claw={last_pose[3]:.1f}')
            print('Visually confirm the cube is held before approving UP. '
                  'Firmware state is not force/physical position feedback.')
            return last_pose
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f'After GRIP, state still {last_pose.round(1)} vs expected '
                f'{np.r_[expected_xyz, expected_claw].round(1)} '
                f'after {attempts} checks. Command was sent ONCE, never retried. '
                'Do not lift until physical gripper and firmware response are investigated.'
            )
        print(f'GRIP state not updated yet (poll {attempts}): '
              f'claw={last_pose[3]:.1f}, expected={expected_claw:.1f}; '
              'waiting without resending...')
        time.sleep(0.7)


def near(value, expected, tol, label):
    if not math.isfinite(float(value)) or abs(float(value) - expected) > tol:
        raise RuntimeError(f'{label}={value:.1f} does not match expected {expected:.1f} +/-{tol:.1f}. STOP.')


def user_approval(token, explanation):
    print('\n' + '=' * 68)
    print(explanation)
    print(f'If this is physically safe, type EXACTLY {token}. Enter alone cancels.')
    answer = input('> ').strip()
    if answer != token:
        raise RuntimeError(f'Cancelled at stage {token}. No further commands sent.')


def send_single(ser, command, wait_s=1.2):
    """Send only one command per confirmed step; no scenario queued."""
    print('SENDING:', command)
    ser.write((command + '\n').encode('ascii'))
    ser.flush()
    time.sleep(wait_s)
    print('Sent. The command may still be executing: inspect the physical robot.')


def after_step(ser, expected_xyz, expected_claw, label):
    """Fail closed on state mismatch; additional visual approval is required."""
    pose = checked_state(ser)
    xyz_tol = 4.0
    claw_tol = 3.0
    if np.max(np.abs(pose[:3] - expected_xyz)) > xyz_tol or abs(pose[3] - expected_claw) > claw_tol:
        raise RuntimeError(f'After {label}, state {pose.round(1)} does not match expected '
                           f'{np.r_[expected_xyz, expected_claw].round(1)}. '
                           'Do not continue until investigated.')
    print(f'Firmware state after {label}: x={pose[0]:.1f}, y={pose[1]:.1f}, '
          f'z={pose[2]:.1f}, claw={pose[3]:.1f}')
    print('NOTE: this does not prove actual physical arrival or object grip.')
    return pose


def perform_pickup(args, target, R, t):
    """Exactly one guarded open/approach/down/grip/up attempt."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError('Install pyserial: python3 -m pip install pyserial') from exc

    dest_board = np.asarray(target['board_mm_top_estimate'], float)
    target_xy = np.asarray(target['robot_xy_top_estimate'], float)
    if base.far_margin_mm(dest_board) < BOARD_MARGIN_MM:
        raise RuntimeError('Cube too close to far-zone boundary.')
    if not np.isfinite(target_xy).all():
        raise RuntimeError('Invalid target coordinates.')

    print('\nOpening ESP32 USB Serial may restart controller. Ensure safe pose.')
    with serial.Serial(args.port, args.baud, timeout=0.2, write_timeout=1.5) as ser:
        time.sleep(2.2)
        pose = checked_state(ser)
        near(pose[2], args.travel_z, 3, 'Initial Z')
        # Only allow a moderate XY displacement entirely inside calibrated far zone.
        move_dist = float(np.linalg.norm(pose[:2] - target_xy))
        if move_dist > MAX_XY_STEP_MM:
            raise RuntimeError(f'XY distance {move_dist:.1f} mm > {MAX_XY_STEP_MM:.0f} mm; '
                               'first manually move robot to a safe staging point.')
        if move_dist >= 0.2:
            base.verify_small_far_motion(pose[:3], target_xy, R, t,
                                         MAX_XY_STEP_MM, BOARD_MARGIN_MM, args.travel_z)
        print('\n--- PRE-FLIGHT ---')
        print(f'Cube: {target["color"]}, BOARD={dest_board.round(1)}, '
              f'ROBOARM XY={target_xy.round(1)}')
        print(f'Current robot state (commanded): {pose.round(1)}')
        print(f'XY approach distance: {move_dist:.1f} mm')
        print(f'Tested travel Z={args.travel_z}, pickup Z={args.pick_z}, '
              f'open claw={args.open_claw}, holding claw={args.closed_claw}')
        print('Pickup permitted only inside FAR region with >=10 mm boundary margin.')
        print('Z=15 was validated at A only; verify clearance at THIS target before DOWN.')
        print('Cube-top centre is only an estimate of its base centre.')
        print('Keep hand clear, power cut-off accessible. Check entire arm sweep, not just XY.')

        x0, y0, z0 = map(float, pose[:3])
        # OPEN while raised, even when reported already open: step is explicit.
        user_approval('OPEN', f'1/5: On safe Z={args.travel_z}, open gripper to claw={args.open_claw}. '
                       'No fingers or objects in jaws.')
        send_single(ser, f'go claw={int(args.open_claw)}')
        pose = after_step(ser, np.array([x0,y0,args.travel_z]), args.open_claw, 'OPEN')

        if move_dist >= 0.2:
            user_approval('XY', f'2/5: Move horizontally to X={target_xy[0]:.1f}, '
                          f'Y={target_xy[1]:.1f} while keeping Z={args.travel_z:.1f}. '
                          'Verify trajectory clear of cube, stand, wires and entire robot arm.')
            # Latest state must match the known staging pose before XY.
            now = checked_state(ser)
            if np.linalg.norm(now[:3] - pose[:3]) > 2.5:
                raise RuntimeError('Pose changed before XY. Stop.')
            base.verify_small_far_motion(now[:3], target_xy, R, t,
                                         MAX_XY_STEP_MM, BOARD_MARGIN_MM, args.travel_z)
            send_single(ser, f'go x={target_xy[0]:.1f} y={target_xy[1]:.1f}', wait_s=1.5)
            pose = after_step(ser, np.r_[target_xy, args.travel_z], args.open_claw, 'XY')
        else:
            print('2/5 XY: already at target by reported state (within 0.2 mm); skipped command.')

        user_approval('DOWN', f'3/5: Z=15 was tested at A, not at this XY. '
                       'BEFORE descent: visually verify gripper directly above '
                       f'cube and open. Descend vertically from Z={args.travel_z:.1f} '
                       f'to your tested Z={args.pick_z:.1f}. '
                       'DO NOT confirm if at an offset or if links/jaws can strike the board.')
        now = checked_state(ser)
        if np.linalg.norm(now[:2] - target_xy) > 3.0:
            raise RuntimeError('XY changed; descent blocked.')
        near(now[2], args.travel_z, 3.0, 'Z before descent')
        near(now[3], args.open_claw, 3.0, 'Claw before descent')
        send_single(ser, f'go z={args.pick_z:.1f}', wait_s=1.5)
        pose = after_step(ser, np.r_[target_xy, args.pick_z], args.open_claw, 'DOWN')

        user_approval('GRIP', f'4/5: Visually verify fingers surround cube, no board contact. '
                       f'Close to experimentally verified claw={args.closed_claw:.1f}. '
                       'Stop if servo stalls, buzzes or strains.')
        now = checked_state(ser)
        near(now[2], args.pick_z, 3.0, 'Z before grip')
        # Use the integer syntax shown in the RoboArm R1 documentation.
        # This still sends a SINGLE command; do not repeat on a slow response.
        send_single(ser, f'go claw={int(args.closed_claw)}')
        pose = wait_after_grip(ser, np.r_[target_xy, args.pick_z], args.closed_claw)

        user_approval('UP', f'5/5: Check cube is held and vertical lift path is clear. '
                       f'Raise Z={args.pick_z:.1f} -> {args.travel_z:.1f} (XY unchanged).')
        now = checked_state(ser)
        near(now[2], args.pick_z, 3.0, 'Z before lift')
        near(now[3], args.closed_claw, 3.0, 'Claw before lift')
        send_single(ser, f'go z={args.travel_z:.1f}', wait_s=1.5)
        pose = after_step(ser, np.r_[target_xy, args.travel_z], args.closed_claw, 'UP')
        print('\nTRIAL FINISHED. Gripper commanded closed at travel Z.')
        print('Physically confirm cube was lifted. No automatic release or reset.')
        print('To put cube down, use manual verified commands; do not force servos.')
        return pose


def main():
    ap = argparse.ArgumentParser(description='Guarded RoboArm pick-up within calibrated FAR zone.')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--board', type=Path, default=Path('board_calibration.npz'))
    ap.add_argument('--fit', type=Path, default=Path('robot_alignment_far_DIAGNOSTIC.npz'))
    ap.add_argument('--port', help='RoboArm Serial port e.g. /dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--enable-motion', action='store_true')
    ap.add_argument('--confirm-tested-grip', action='store_true',
                    help='acknowledge tested claw positions; Z=15 was validated at A only, so re-check target before DOWN')
    ap.add_argument('--travel-z', type=float, default=85)
    ap.add_argument('--pick-z', type=float, default=15)
    ap.add_argument('--open-claw', type=float, default=50)
    ap.add_argument('--closed-claw', type=float, default=0)
    ap.add_argument('--min-area', type=float, default=150)
    ap.add_argument('--save-dir', type=Path, default=Path('cube_pickup_trials'))
    args = ap.parse_args()
    if args.baud != 115200:
        ap.error('RoboArm R1 expected Serial baud 115200')
    if (args.travel_z, args.pick_z, args.open_claw, args.closed_claw) != (85, 15, 50, 0):
        ap.error('First experiment locks tested values: travel-z=85 pick-z=15 open=50 closed=0')
    if args.enable_motion and (not args.port or not args.confirm_tested_grip):
        ap.error('For motion supply --port ... and --confirm-tested-grip')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open camera {args.camera}')
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('Camera did not return frame')
    H, centres = base.load_board(args.board, frame.shape)
    R, t, fit_source = base.load_alignment(args.fit, args.enable_motion)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    roi_poly = np.rint(centres[[0,1,3,2]]).astype(np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, roi_poly, 255)
    mask = cv2.erode(mask, np.ones((13,13),np.uint8), iterations=1)

    selected = None
    frozen = False
    frozen_frame = None
    candidate = None
    tried = False
    msg = 'G/R select color; SPACE freeze; P preview; M gated pickup; Q quit'
    print('\nFAR-ZONE PICKUP: dry-run by default; one trial per launch.')
    print('Fit:', fit_source)
    print('G/R choose, SPACE freeze, P preview, M pickup, S save, Q exit.')
    if args.enable_motion:
        print('MOTION ENABLED but 5 stages each require independent confirmation in TERMINAL.')
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            if not frozen:
                ok, frame = cap.read()
                if not ok:
                    print('Camera disconnected.'); break
                if frame.shape[:2] != mask.shape:
                    raise RuntimeError('Camera frame size changed.')
            else:
                frame = frozen_frame.copy()
            found = base.marker_centres(frame, detector)
            ref_ok, ref_text, _ = base.marker_check(H, found)
            objects = base.find_cubes(frame, mask, args.min_area)
            canvas = frame.copy()
            cv2.polylines(canvas, [roi_poly], True, (0,255,255), 2)
            far_px = base.project(np.linalg.inv(H), base.FAR_HULL.reshape(-1,2))
            cv2.polylines(canvas, [np.rint(far_px).astype(np.int32)], True, (255,140,0), 2)
            for obj in objects:
                cx,cy = obj['centre_px']; x,y,w,h = obj['bbox']
                color = (0,255,0) if obj['color'] == 'green' else (0,0,255)
                cv2.rectangle(canvas, (x,y),(x+w,y+h),color,2)
                cv2.drawMarker(canvas,(round(cx),round(cy)),(255,255,0),cv2.MARKER_CROSS,12,2)
            mode = 'MOTION ARMED' if args.enable_motion else 'DRY RUN'
            base.draw_text(canvas, f'{mode} | {"FROZEN" if frozen else "LIVE"} | '
                           f'{selected or "NO COLOR"} | markers {sorted(found)}', 7, 22,
                           (0, 255, 255), .45)
            base.draw_text(canvas, ref_text, 7, 45,
                           (70, 225, 70) if ref_ok else (0,0,255), .42)
            base.draw_text(canvas, msg[:100], 7, canvas.shape[0]-16, scale=.42)
            if candidate is not None and frozen:
                base.draw_text(canvas, 'Target R XY: '+str(candidate['robot_xy_top_estimate']),
                               7, 68, (0,255,255), .44)
            if tried:
                base.draw_text(canvas, 'PICKUP TRIAL ALREADY USED - RESTART TO TRY AGAIN',
                               7, 90, (0,0,255), .44)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(30) & 0xff
            if key in (ord('q'),27):
                break
            elif key in (ord('g'), ord('r')):
                selected = 'green' if key==ord('g') else 'red'
                candidate = None
                msg = f'Selected {selected}; SPACE to freeze'
            elif key == ord(' '):
                frozen = not frozen
                frozen_frame = frame.copy() if frozen else None
                candidate = None
                msg = 'FROZEN: press P for preview' if frozen else 'LIVE'
            elif key == ord('p'):
                if not frozen or not ref_ok:
                    msg = 'Freeze image and have >=3 stable ArUco before preview'
                else:
                    try:
                        candidate = base.select_target(objects, selected, H, R, t, BOARD_MARGIN_MM)
                        print('\nTARGET PREVIEW (NO MOVEMENT):\n' +
                              json.dumps(candidate, ensure_ascii=False, indent=2))
                        msg = 'Preview OK: within FAR zone; press M when ready'
                    except (ValueError,RuntimeError) as exc:
                        msg = str(exc)
                        candidate = None
            elif key == ord('s'):
                base.save_snapshot(args.save_dir, frame, canvas, candidate, found, tried)
                msg = 'Snapshot saved'
            elif key == ord('m'):
                if tried:
                    msg = 'Only one trial per launch. Restart program.'
                elif not args.enable_motion:
                    msg = 'DRY RUN: no robot commands sent.'
                    print(msg)
                elif not frozen or candidate is None or not ref_ok:
                    msg = 'Freeze camera and press P with valid markers before M.'
                else:
                    tried = True  # prevents re-run even if cancelled or half-complete
                    try:
                        perform_pickup(args, candidate, R, t)
                        msg = 'Commands issued; verify actual grip and lift visually.'
                    except (Exception,KeyboardInterrupt) as exc:
                        msg = 'PICKUP STOPPED: '+str(exc)
                        print(msg)
                        print('NOTE: partial actions MAY already have happened. '
                              'Never retry without checking live pose and scene.')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        raise SystemExit(1)
