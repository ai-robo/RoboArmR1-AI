#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboArm R1 + DeepSeek + Vosk: voice -> text -> safe JSON -> OpenCV -> serial.

This is the voice-enabled copy of the latest text controller. Text mode is
still available with --input text. In voice mode an utterance is ignored until
the literal wake word "Робот" is recognized by Vosk; commands are never sent
directly to motors without DeepSeek JSON validation and geometric checks.

Based on the user's tested four-colour RoboArm code. White cubes are excluded.
The camera UI stays live during API calls AND servo motion.

IMPORTANT: `state` is the controller's requested pose, not servo position feedback.
No collision sensor / force sensor / actual successful-grasp feedback is available.
Stay beside the robot with a physical motor-power cut-off. ESC stops sending
NEXT commands; it cannot interrupt a movement already issued to firmware.

The first executed task captures HOME XY at Z=85. After each successful pickup
and placement, RoboArm is COMMANDED to go to saved HOME XY and close its claw.
No firmware XY-position check is used at HOME or when starting the NEXT task.
The return command may fail physically without detection; supervise the robot.
Other motion/geometry guards remain active and lock motion on detected faults.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

import roboarm_move_to_cube as base
import roboarm_pickup_cube as grip
import roboarm_pick_and_place as zone
import roboarm_auto_pick_and_place_4colors_LOOP as vision

COLORS = frozenset(('red', 'green', 'blue', 'gold'))
MODEL = 'deepseek-flash'
# Stacking release height (user-selected; not yet physically tested on this robot).
STACK_PLACE_Z_DEFAULT = 40.0
BASE_URL = 'https://api.deepseek.com'
WINDOW = 'RoboArm + DeepSeek | live camera'
# One authoritative threshold for THIS DeepSeek controller.
# Do not inherit an old 6 mm value from imported vision modules.
MAX_MARKER_DRIFT_MM = 10.0
# Maximum time to find and validate the requested cubes after the DeepSeek plan.
# Expiry drops the task WITHOUT opening the serial port and returns to wake-word listening.
VISION_TIMEOUT_S = 8.0

# Right side of the robot (from its HOME viewpoint) is negative RoboArm Y.
# Field-observed pickup error: on the RIGHT side the gripper travels ~10 mm
# too far right. At Z=85, make a FINAL +10 mm Y correction before LOWER_TO_PICK.
# Apply only when the target is at least 25 mm right of the saved HOME Y.
RIGHT_PICK_SIDE_FROM_HOME_MM = 25.0
RIGHT_PICK_FINAL_Y_MM = 10.0

# The AI outputs symbolic steps: Python resolves the real coordinates and commands.
POINT_STEPS = ['FIND_SOURCE', 'OPEN', 'MOVE_TO_SOURCE', 'LOWER_TO_PICK', 'GRIP',
               'LIFT', 'MOVE_TO_TARGET', 'LOWER_TO_PLACE', 'RELEASE', 'RETRACT']
STACK_STEPS = ['FIND_SOURCE', 'FIND_TARGET', 'OPEN', 'MOVE_TO_SOURCE',
               'LOWER_TO_PICK', 'GRIP', 'LIFT', 'MOVE_TO_TARGET',
               'LOWER_TO_PLACE', 'RELEASE', 'RETRACT']

PLANNER_PROMPT = '''You plan ONE pick-and-place action for a RoboArm R1 robot. The user writes Russian text.
Return ONLY a valid JSON object, no markdown, no comments. Use exactly the documented keys.

Allowed cube colors: red, green, blue, gold. White is NOT supported.
Russian-to-JSON color mapping is MANDATORY:
  красный/красного/красным = red
  зелёный/зеленый/зелёного/зеленого = green
  синий/синего/синим/синюю = blue (SUPPORTED, not an error)
  золотой/золотого/золотым = gold
NEVER reject because BLUE, RED, GREEN or GOLD is an unsupported color. ALL FOUR ARE VALID.
Reject only when the actual requested operation or destination is unsupported/ambiguous.
Allowed destinations:
  - a fixed position called point 1 / точка 1 / центр между ArUco 2 и 3
  - ON TOP OF a cube of a different color.
Supported operations are only pick ONE cube and place it at ONE target.
No plural commands, no travel distances, no servo coordinates, no other tools.
Do not guess the object's coordinates: OpenCV measures them later.

JSON for an accepted command:
{"mode":"sequence","pick_color":"green","target":{"type":"point","id":1},"steps":["FIND_SOURCE","OPEN","MOVE_TO_SOURCE","LOWER_TO_PICK","GRIP","LIFT","MOVE_TO_TARGET","LOWER_TO_PLACE","RELEASE","RETRACT"],"reason":""}
For stacking onto another colored cube, target is {"type":"on_cube","color":"red"}, and
steps are ["FIND_SOURCE","FIND_TARGET","OPEN","MOVE_TO_SOURCE","LOWER_TO_PICK","GRIP","LIFT","MOVE_TO_TARGET","LOWER_TO_PLACE","RELEASE","RETRACT"].
For an unsupported/ambiguous request, return:
{"mode":"reject","pick_color":null,"target":null,"steps":[],"reason":"Brief explanation in Russian"}

Examples:
"возьми зеленый кубик и перенеси его в точку 1" -> green, point 1.
"возьми красный кубик и перенеси его в точку 1" -> red, point 1.
"возьми синий кубик и положи его в точку один" -> blue, point 1 (sequence).
"найди синий кубик и перенеси его в точку 1" -> blue, point 1 (sequence).
"возьми синий кубик и поставь на красный" -> blue, on_cube red (sequence).
"возьми зеленый кубик и поставь его на красный кубик" -> green, on_cube red.
"перенеси белый кубик" -> reject.
Make no extra steps. JSON only.
'''


def validate_plan(raw: Any) -> dict:
    """Fail closed. Never execute arbitrary model-produced motor instructions."""
    if not isinstance(raw, dict):
        raise ValueError('DeepSeek must return JSON object')
    allowed = {'mode', 'pick_color', 'target', 'steps', 'reason'}
    if set(raw) != allowed:
        raise ValueError(f'Plan keys mismatch: {sorted(set(raw) ^ allowed)}')
    mode = raw['mode']
    if mode == 'reject':
        if raw['pick_color'] is not None or raw['target'] is not None or raw['steps'] != []:
            raise ValueError('Invalid reject plan')
        if not isinstance(raw['reason'], str):
            raise ValueError('Invalid reason')
        return raw
    if mode != 'sequence':
        raise ValueError('Unsupported mode: ' + str(mode))
    color = raw['pick_color']
    if color not in COLORS or not isinstance(color, str):
        raise ValueError('Unsupported source cube color')
    target = raw['target']
    if not isinstance(target, dict):
        raise ValueError('Missing target specification')
    if target == {'type': 'point', 'id': 1}:
        expected = POINT_STEPS
    elif (set(target) == {'type', 'color'} and target['type'] == 'on_cube'
          and type(target['color']) is str and target['color'] in COLORS
          and target['color'] != color):
        expected = STACK_STEPS
    else:
        raise ValueError('Unsupported target; only point 1 or other colored cube')
    if raw['steps'] != expected:
        raise ValueError('Invalid/changed sequence; no motion authorized')
    if not isinstance(raw['reason'], str):
        raise ValueError('Invalid reason')
    return raw



# A second, independent guard against the model accidentally swapping colors.
# This is NOT the primary language parser; it only checks explicit color words
# already present in a command after DeepSeek has interpreted its meaning.
COLOR_WORD_RE = re.compile(r'\b(красн\w*|зел[её]н\w*|син(?:ий|его|ему|им|яя|юю|ей|ими|ие|их)|золот\w*|бел\w*)\b', re.I)


def validate_text_consistency(text: str, plan: dict) -> None:
    if plan['mode'] != 'sequence':
        return
    colors = []
    for hit in COLOR_WORD_RE.finditer(text.lower()):
        token = hit.group(1)
        if token.startswith('красн'):
            colors.append('red')
        elif token.startswith('зел'):
            colors.append('green')
        elif token.startswith('син'):
            colors.append('blue')
        elif token.startswith('золот'):
            colors.append('gold')
        else:
            colors.append('white')
    if colors and plan['pick_color'] != colors[0]:
        raise ValueError('DeepSeek source color contradicts explicit user text')
    if 'white' in colors:
        raise ValueError('White is unsupported; refusing to move')
    target = plan['target']
    if target['type'] == 'on_cube' and len(colors) >= 2 and target['color'] != colors[-1]:
        raise ValueError('DeepSeek destination color contradicts explicit user text')
    if target['type'] == 'point' and re.search(r'точк\w*\s*[2-9]\b', text.lower()):
        raise ValueError('Only point 1 exists')


# A model may occasionally hallucinate "blue is unsupported" while listing
# blue as allowed. This is a MODEL error, not a reason to block valid colors.
# Detect such contradictions, request ONE fresh plan, and fail closed if the
# follow-up is still contradictory. Never silently turn reject into motion.
_SUPPORTED_COLOR_TERMS = re.compile(
    r'\b(?:красн\w*|зел[её]н\w*|син\w*|золот\w*)\b', re.I)
_UNSUPPORTED_CLAIM = re.compile(
    r'(?:не\s+поддерж\w*|не\s+допустим\w*|не\s+разреш\w*|'
    r'не\s+доступ\w*|недоступ\w*|не\s+предусмотр\w*|запрещ\w*)', re.I)


def inconsistent_color_reject(user_text: str, plan: dict) -> bool:
    """Check for a reject claiming one of our requested supported colors is invalid.

    Work on each clause separately to avoid false positives from a later
    explanatory list of allowed colors. Only detect a claim about a color
    explicitly mentioned in the user's command.
    """
    if plan['mode'] != 'reject':
        return False
    user_colors = {m.group(0)[:3].replace('ё', 'е')
                   for m in _SUPPORTED_COLOR_TERMS.finditer(user_text.lower())}
    if not user_colors:
        return False
    for clause in re.split(r'[.;!?\n]', plan['reason'].lower()):
        # Ignore long enumerations ("допустимые цвета: красный, ...")
        clause = clause.split('допустимые цвета:')[0].split('разрешённые цвета:')[0]
        if not _UNSUPPORTED_CLAIM.search(clause):
            continue
        if any(m.group(0)[:3].replace('ё', 'е') in user_colors
               for m in _SUPPORTED_COLOR_TERMS.finditer(clause)):
            return True
    return False


_RETRY_PROMPT = ("IMPORTANT CORRECTION: ALL these source/target colors are supported: "
                 "red=красный, green=зелёный, blue=синий, gold=золотой. "
                 "Your previous rejection claimed that a supported color was invalid. "
                 "Reconsider the SAME original Russian instruction; return a fresh JSON. "
                 "If the instruction is unsupported for ANOTHER reason, reject with "
                 "the actual reason. Do not invent motor coordinates or steps.")


def deepseek_plan(text: str, model: str = MODEL) -> dict:
    # API calls live in a worker so the OpenCV window remains responsive.
    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise RuntimeError('Set DEEPSEEK_API_KEY in the environment')
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url=BASE_URL,
                    timeout=30.0, max_retries=0)
    for attempt in range(2):
        prompt = PLANNER_PROMPT if attempt == 0 else PLANNER_PROMPT + '\n' + _RETRY_PROMPT
        result = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': prompt},
                      {'role': 'user', 'content': text}],
            response_format={'type': 'json_object'},
            max_tokens=380,
            extra_body={'thinking': {'type': 'disabled'}},
        )
        if not result.choices or result.choices[0].finish_reason != 'stop':
            raise RuntimeError('DeepSeek response unfinished; refusing to move')
        content = result.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError('DeepSeek returned an empty JSON response')
        plan = validate_plan(json.loads(content))
        if inconsistent_color_reject(text, plan):
            if attempt == 0:
                print('DEEPSEEK WARNING: contradictory color rejection; retrying plan once',
                      flush=True)
                continue
            raise RuntimeError('DeepSeek twice claimed a supported color is unsupported; '
                               'no motion authorized. Please repeat the command.')
        validate_text_consistency(text, plan)
        return plan
    raise AssertionError('Unreachable: both planning attempts consumed')


def board_xy(obj: dict) -> np.ndarray:
    return np.asarray(obj['board_mm_top_estimate'], dtype=float)


def robot_xy(obj: dict) -> np.ndarray:
    return np.asarray(obj['robot_xy_top_estimate'], dtype=float)


def corrected_right_pick_xy(nominal_xy, home_xy):
    """Correct right-side pickup ONLY; leave all placement/HOME XYZ unchanged.

    +Y is robot-left, -Y is robot-right (not the camera image's X axis).
    Moving +10 mm Y pulls the gripper BACK toward HOME from robot-right.
    Return (corrected XY, whether correction was applied).
    """
    nominal = np.asarray(nominal_xy, dtype=float)
    home = np.asarray(home_xy, dtype=float)
    if nominal.shape != (2,) or home.shape != (2,) or not np.isfinite(nominal).all() or not np.isfinite(home).all():
        raise ValueError('Invalid XY for right-side pickup correction')
    corrected = nominal.copy()
    active = bool(nominal[1] <= home[1] - RIGHT_PICK_SIDE_FROM_HOME_MM)
    if active:
        corrected[1] += RIGHT_PICK_FINAL_Y_MM
    return corrected, active


def same_scene(a: dict, b: dict, mm: float = 4.0) -> bool:
    """Check that target colors/positions are stable across live camera frames."""
    if a['color'] != b['color']:
        return False
    return float(np.linalg.norm(board_xy(a) - board_xy(b))) <= mm


class MarkerStabilityMonitor:
    """Temporal ArUco guard: require two consistently calibrated markers.

    For each marker ID separately, at least 4 of the last 5 frames must
    contain an in-tolerance observation (< 10 mm). A single spurious
    reading or a third unreliable marker does not block a reliable pair.
    If the camera really moves, persistent errors drop the quorum.
    """

    def __init__(self, threshold_mm=MAX_MARKER_DRIFT_MM, window=5, good_frames=4, min_markers=2):
        if not (0 < good_frames <= window and min_markers >= 2):
            raise ValueError('Invalid ArUco stability window')
        self.threshold_mm = float(threshold_mm)
        self.window = int(window)
        self.good_frames = int(good_frames)
        self.min_markers = int(min_markers)
        self.history = deque(maxlen=self.window)

    def reset(self):
        self.history.clear()

    def observe(self, H, markers):
        """Accept board-to-marker residuals; never change saved calibration."""
        errors = {}
        for marker_id, centre_px in markers.items():
            if marker_id not in base.MARKER_BOARD_MM:
                continue
            try:
                expected = np.asarray(base.MARKER_BOARD_MM[marker_id], dtype=float)
                measured = base.project(H, [centre_px])[0]
                error = float(np.linalg.norm(measured - expected))
                if np.isfinite(error):
                    errors[int(marker_id)] = error
            except (cv2.error, ValueError, TypeError):
                continue
        self.history.append(errors)
        n = len(self.history)
        if n < self.window:
            return False, f'Collecting ArUco frames: {n}/{self.window}'

        # Quorum is based on SAME marker IDs, not two unrelated markers
        # that happened to look accurate in different frames.
        good_ids = []
        for marker_id in base.MARKER_BOARD_MM:
            good_readings = [sample[marker_id] for sample in self.history
                             if marker_id in sample
                             and sample[marker_id] < self.threshold_mm]
            if len(good_readings) >= self.good_frames:
                good_ids.append(marker_id)
        ok = len(good_ids) >= self.min_markers
        if ok:
            return True, ('ArUco OK: stable IDs ' + ','.join(map(str, good_ids))
                          + f' (>= {self.good_frames}/{self.window}, '
                          + f'< {self.threshold_mm:g} mm)')
        current = ', '.join(f'{k}:{v:.1f}' for k,v in sorted(errors.items())) or 'none'
        return False, (f'ArUco stability {len(good_ids)}/{self.min_markers}: '
                       f'need >= {self.good_frames}/{self.window} frames '
                       f'within < {self.threshold_mm:g} mm; current {current}')



def perception_due(now, previous, hz):
    """Rate limit costly CV perception independent of the live camera FPS."""
    return float(hz) > 0 and now - previous >= 1.0 / float(hz) - 1e-7


def detect_marker_data(frame, detector):
    """Detect ArUco ONCE; reuse both centers and polygon corners."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    markers = {}
    if ids is not None:
        for marker_id, corner in zip(ids.flatten(), corners):
            if int(marker_id) in base.MARKER_BOARD_MM:
                markers[int(marker_id)] = corner[0].mean(axis=0).astype(float)
    return markers, corners


class CpuDiagnostics:
    """Optional CPU/vision timing measurements without external dependencies."""
    def __init__(self, enabled=False, interval=10.0):
        self.enabled = bool(enabled)
        self.interval = float(interval)
        self.last_wall = time.monotonic()
        self.last_cpu = time.process_time()
        self.vision_calls = 0
        self.vision_sec = 0.0

    def add_vision(self, elapsed):
        if self.enabled:
            self.vision_calls += 1
            self.vision_sec += max(0.0, float(elapsed))

    def report(self):
        if not self.enabled:
            return
        now = time.monotonic()
        dt = now - self.last_wall
        if dt < self.interval:
            return
        cpu = time.process_time() - self.last_cpu
        avg = 1000.0 * self.vision_sec / self.vision_calls if self.vision_calls else 0.0
        print(f'CPU STATS: process CPU~{100*cpu/max(dt,1e-6):.0f}% ' 
              f'| vision {self.vision_calls/dt:.1f} scans/s '
              f'| average {avg:.1f} ms/scan '
              f'| interval {dt:.1f}s', flush=True)
        self.last_wall = now
        self.last_cpu = time.process_time()
        self.vision_calls = 0
        self.vision_sec = 0.0


def make_board_search_mask(H, shape):
    """Full *plywood* mask, NOT the smaller red work polygon.

    This lets us tell "requested cube absent" from "requested cube visible
    but OUTSIDE the red workspace" without treating colored room objects
    beyond the plywood as candidate cubes.
    """
    width, height = base.BOARD_SIZE_MM
    polygon = np.asarray([(0., 0.), (width, 0.), (width, height),
                          (0., height)], dtype=np.float32)
    pixels = np.rint(base.project(np.linalg.inv(H), polygon)).astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pixels, 255)
    return cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)


def detect_board_objects(frame, H, R, t, detector, min_area, board_mask=None,
                         marker_corners=None):
    """Find colored cubes on the plywood, including *outside* the red ROI.

    This is diagnostic perception, NOT permission to move: permission is
    checked separately by select_scene() for BOTH required objects.
    """
    if board_mask is None:
        board_mask = make_board_search_mask(H, frame.shape)
    cubes, _ = vision.find_four_color_cubes(
        frame, board_mask, min_area, H, detector, marker_corners=marker_corners)
    objects = []
    for cube in cubes:
        try:
            item = vision.target_from_cube(cube, H, R, t)
            margin = zone.workspace_margin_mm(board_xy(item))
            item['workspace_distance_mm'] = round(margin, 2)
            item['inside_workspace'] = bool(margin >= zone.PICK_MARGIN - 1e-4)
            objects.append(item)
        except (RuntimeError, ValueError, cv2.error, np.linalg.LinAlgError):
            continue
    return objects


def _object_status(objects):
    if not objects:
        return 'на фанере не обнаружено ни одного цветного кубика'
    names = ', '.join(o['color'] + (' (вне зоны)' if not o['inside_workspace'] else '')
                      for o in objects)
    return 'видны: ' + names


def select_scene(frame, plan, detector, H, R, t, mask, min_area,
                 marker_monitor=None, all_objects=None, observed_markers=None):
    """Check requested source+destination, independent of unrelated cubes.

    Return a scene ONLY if source and destination exist uniquely, are allowed
    and all previous geometry / ArUco checks pass. Unrelated cubes outside the
    red zone are displayed but do not prevent an otherwise safe operation.
    """
    markers = (base.marker_centres(frame, detector)
               if observed_markers is None else observed_markers)
    if marker_monitor is None:
        marker_ok, marker_message, _ = vision.check_reference_markers(H, markers)
    else:
        marker_ok, marker_message = marker_monitor.observe(H, markers)
    objects = (detect_board_objects(frame, H, R, t, detector, min_area)
               if all_objects is None else all_objects)
    if not objects:
        return None, objects, marker_ok, '[NO_CUBES] ' + _object_status(objects)

    pick_matches = [o for o in objects if o['color'] == plan['pick_color']]
    if not pick_matches:
        return None, objects, marker_ok, (f'[SOURCE_NOT_FOUND] нет кубика '
            f'{plan["pick_color"]}; ' + _object_status(objects))
    if len(pick_matches) != 1:
        return None, objects, marker_ok, (f'[AMBIGUOUS_SOURCE] '
            f'found {len(pick_matches)} {plan["pick_color"]} cubes; '
            'нужно оставить один кубик указанного цвета')
    pick = pick_matches[0]
    if not pick['inside_workspace']:
        return None, objects, marker_ok, (
            f'[SOURCE_OUTSIDE] кубик {plan["pick_color"]} вне красной зоны '
            f'({pick["workspace_distance_mm"]:+.1f} мм); переместите его внутрь')

    target_spec = plan['target']
    destination = None
    if target_spec['type'] == 'point':
        dest_board = vision.DROP_BOARD_MM.copy()
        # Point 1 must be clear even when some of the *other* cubes are outside.
        if any(np.linalg.norm(board_xy(o) - dest_board) < vision.DROP_GUARD_RADIUS_MM
               for o in objects):
            return None, objects, marker_ok, '[POINT_OCCUPIED] point 1 is occupied'
    else:
        wanted = target_spec['color']
        target_matches = [o for o in objects if o['color'] == wanted]
        if not target_matches:
            return None, objects, marker_ok, (f'[TARGET_NOT_FOUND] '
                f'нет кубика-цели {wanted}; ' + _object_status(objects))
        if len(target_matches) != 1:
            return None, objects, marker_ok, (f'[AMBIGUOUS_TARGET] '
                f'found {len(target_matches)} {wanted} target cubes; нужен один')
        destination = target_matches[0]
        if not destination['inside_workspace']:
            return None, objects, marker_ok, (
                f'[TARGET_OUTSIDE] кубик-цель {wanted} вне красной зоны '
                f'({destination["workspace_distance_mm"]:+.1f} мм); '
                'установку не выполняем')
        dest_board = board_xy(destination)

    if not marker_ok:
        return None, objects, False, '[ARUCO_WAIT] ' + marker_message

    src_board = board_xy(pick)
    src_robot = robot_xy(pick)
    dest_robot = R @ dest_board + t
    # Repeat geometrical checks in the motion worker before opening Serial.
    try:
        zone.check_workspace(src_board, 'Pick', zone.PICK_MARGIN)
        zone.check_board_edge(src_board, 'Pick')
        zone.check_robot_xy_envelope(src_robot, 'Pick')
        zone.check_workspace(dest_board, 'Place', zone.DROP_MARGIN)
        zone.check_board_edge(dest_board, 'Place')
        zone.check_robot_xy_envelope(dest_robot, 'Place')
        if np.linalg.norm(src_board - dest_board) < vision.MIN_PICK_TO_DROP_MM:
            raise RuntimeError('Pickup and target positions too close')
        if destination is not None:
            for o in objects:
                if o is pick or o is destination:
                    continue
                if float(np.linalg.norm(board_xy(o) - dest_board)) < 32.0:
                    raise RuntimeError('Another cube is too close to stack target')
        zone.safe_xy_path(src_robot, dest_robot, R, t)
    except (RuntimeError, ValueError) as exc:
        return None, objects, marker_ok, '[GEOMETRY_BLOCKED] ' + str(exc)
    return {'source': pick, 'target': destination,
            'dest_board': dest_board, 'dest_robot': dest_robot}, objects, marker_ok, 'READY'


def vision_wait_expired(started_at, timeout_s, now=None):
    """Pure guard, convenient for offline testing; no hardware side effects."""
    now = time.monotonic() if now is None else float(now)
    return started_at is not None and now - float(started_at) >= float(timeout_s)


class HomeAnchor:
    """Preserve INITIAL travel-height XY across multiple text commands.

    First hardware task captures XY as the fixed return target. Subsequent
    tasks do NOT compare current XY with HOME: small positioning offsets are fine.
    The actual next-task start is still checked for a safe outbound route.
    """

    def __init__(self):
        self.xy = None

    def validate_and_get(self, current_xyz, pick_xy, dest_xy, R, t):
        current = np.asarray(current_xyz, dtype=float)
        if current.shape != (4,) or not np.isfinite(current).all():
            raise RuntimeError('Cannot establish HOME: invalid RoboArm state')
        grip.near(current[2], zone.TRAVEL_Z, 3.0, 'HOME start Z')
        candidate = current[:2].copy() if self.xy is None else self.xy.copy()
        # Validate the actual CURRENT start -> pickup route, even when its XY
        # differs from the saved HOME target. Do not enforce a HOME XY tolerance.
        zone.safe_xy_path(current[:2], pick_xy, R, t)
        zone.safe_xy_path(pick_xy, dest_xy, R, t)
        zone.safe_xy_path(dest_xy, candidate, R, t)
        if self.xy is None:
            self.xy = candidate.copy()
            print('HOME CAPTURED: XY=', np.round(candidate, 1).tolist(),
                  'Z=', zone.TRAVEL_Z, 'initial claw=', float(current[3]), flush=True)
        return candidate.copy()


def stage_move(ser, cmd, xyz, claw, stage, seconds, status, stop):
    vision.stage_send(ser, cmd, xyz, claw, stage, seconds,
                      progress=status.update, cancel_event=stop)

def return_home_without_xy_check(ser, from_xy, home_xy, R, t, pace, status, stop):
    """Send HOME XY movement(s) without checking reported HOME coordinates.

    Keep the tested workspace/path limits and <=60-mm horizontal waypoints.
    A timed pause is NOT proof that the actual servos reached HOME.
    """
    waypoints = zone.safe_xy_path(from_xy, home_xy, R, t)
    previous = np.asarray(from_xy, dtype=float)
    for index, point in enumerate(waypoints, 1):
        target = np.asarray(point, dtype=float)
        distance = float(np.linalg.norm(target - previous))
        if distance < 0.2:
            previous = target
            continue
        if stop.is_set():
            raise RuntimeError('Stop requested: no further HOME commands sent')
        command = f'go x={target[0]:.1f} y={target[1]:.1f}'
        label = f'RETURN_HOME {index}/{len(waypoints)}'
        status.update(label, command + ' (no HOME XY verification)')
        print(f'  >>> {label}: {command} [no HOME XY check]', flush=True)
        ser.write((command + '\n').encode('ascii'))
        ser.flush()
        delay = pace.xy_base_s + distance * pace.xy_per_mm_s
        print(f'      waiting {delay:.2f}s for movement (not pose feedback)...', flush=True)
        time.sleep(max(0.0, delay))
        previous = target


def close_home_without_xy_check(ser, pace, status, stop):
    """Close at HOME without requiring exact HOME XY in `state`."""
    if stop.is_set():
        raise RuntimeError('Stop requested: HOME close was not sent')
    command = 'go claw=0'
    status.update('CLOSE_HOME', command + ' (no HOME XY verification)')
    print('  >>> CLOSE_HOME:', command, '[no HOME XY check]', flush=True)
    ser.write((command + '\n').encode('ascii'))
    ser.flush()
    time.sleep(max(0.0, pace.claw_s))


def verified_motion(plan, scene, args, R, t, status, stop, home_anchor=None):
    """Resolve validated symbolic stages to actual serial commands.

    The DeepSeek response NEVER supplies raw serial commands or numeric motor coordinates.
    """
    import serial
    if home_anchor is None:
        home_anchor = HomeAnchor()
    # Nominal XY comes directly from calibrated OpenCV. The right-only offset
    # is resolved AFTER HOME is known, and only for the final pickup approach.
    nominal_pick_xy = robot_xy(scene['source'])
    dest_xy = np.asarray(scene['dest_robot'], float)
    src_board = board_xy(scene['source'])
    dest_board = np.asarray(scene['dest_board'], float)
    zone.check_workspace(src_board, 'Pick', zone.PICK_MARGIN)
    zone.check_board_edge(src_board, 'Pick')
    zone.check_robot_xy_envelope(nominal_pick_xy, 'Pick')
    zone.check_workspace(dest_board, 'Place', zone.DROP_MARGIN)
    zone.check_board_edge(dest_board, 'Place')
    zone.check_robot_xy_envelope(dest_xy, 'Place')
    if np.linalg.norm(src_board - dest_board) < vision.MIN_PICK_TO_DROP_MM:
        raise RuntimeError('Pickup/placement distance too short')
    zone.safe_xy_path(nominal_pick_xy, dest_xy, R, t)

    stack = plan['target']['type'] == 'on_cube'
    # Stacking is enabled when --execute is supplied; scene/geometry checks still apply.
    place_z = float(args.stack_place_z if stack else zone.PLACE_Z)
    if not (zone.PICK_Z + 4 <= place_z <= zone.TRAVEL_Z - 15):
        raise RuntimeError('Unsafe/invalid place Z')
    if stack and not 35.0 <= place_z <= 55.0:
        raise RuntimeError('Stack placement height outside expected 20-mm-cube range')
    pace = vision.PACE_PROFILES[args.pace]
    status.update('CONNECT', 'opening RoboArm serial port')
    with serial.Serial(args.port, 115200, timeout=0.2, write_timeout=1.5) as ser:
        # Current RoboArm code uses this pause since opening USB may reset ESP32.
        for _ in range(22):
            if stop.is_set():
                raise RuntimeError('ESC before serial start')
            time.sleep(.1)
        current = grip.checked_state(ser)
        home_xy = home_anchor.validate_and_get(current, nominal_pick_xy, dest_xy, R, t)
        pick_xy, right_pick_active = corrected_right_pick_xy(nominal_pick_xy, home_xy)
        if right_pick_active:
            # Preflight the EXTRA final motion and the corrected pick-to-place
            # route BEFORE sending even the first motor command.
            zone.check_robot_xy_envelope(pick_xy, 'Right pickup correction')
            corrected_board = np.linalg.solve(R, pick_xy - t)
            zone.check_workspace(corrected_board, 'Corrected pickup', zone.PICK_MARGIN)
            zone.check_board_edge(corrected_board, 'Corrected pickup')
            zone.safe_xy_path(nominal_pick_xy, pick_xy, R, t)
            zone.safe_xy_path(pick_xy, dest_xy, R, t)
        status.update('HOME', f'Home XY saved: {home_xy[0]:.1f}, {home_xy[1]:.1f} at Z=85')
        print('VERIFIED PLAN:', json.dumps(plan, ensure_ascii=False), flush=True)
        if right_pick_active:
            print('RIGHT-SIDE PICKUP: nominal XY', np.round(nominal_pick_xy, 1).tolist(),
                  '-> corrected XY', np.round(pick_xy, 1).tolist(),
                  '(final +10 mm RoboArm Y at Z=85)', flush=True)
        print('Pick board XY:', np.round(src_board, 1).tolist(),
              '| place board XY:', np.round(dest_board, 1).tolist(),
              '| release Z:', place_z, flush=True)
        # Matches the only supported, fully validated symbolic action list.
        stage_move(ser, 'go claw=50', current[:3], zone.OPEN_CLAW,
                   'OPEN', pace.claw_s, status, stop)
        vision.xy_route(ser, current[:2], nominal_pick_xy, R, t, zone.OPEN_CLAW,
                        'MOVE_TO_SOURCE', pace, progress=status.update,
                        cancel_event=stop)
        if right_pick_active:
            correction_cmd = f'go x={pick_xy[0]:.1f} y={pick_xy[1]:.1f}'
            stage_move(ser, correction_cmd, np.r_[pick_xy, zone.TRAVEL_Z], zone.OPEN_CLAW,
                       'RIGHT_PICK_CORRECTION',
                       pace.xy_base_s + RIGHT_PICK_FINAL_Y_MM * pace.xy_per_mm_s,
                       status, stop)
        current = grip.checked_state(ser)
        if np.linalg.norm(current[:2] - pick_xy) > 3.0:
            raise RuntimeError('XY mismatch before pickup DOWN')
        stage_move(ser, 'go z=15', np.r_[pick_xy, zone.PICK_Z], zone.OPEN_CLAW,
                   'LOWER_TO_PICK', pace.vertical_base_s +
                   abs(zone.TRAVEL_Z - zone.PICK_Z)*pace.vertical_per_mm_s,
                   status, stop)
        stage_move(ser, 'go claw=0', np.r_[pick_xy, zone.PICK_Z], zone.CLOSED_CLAW,
                   'GRIP', pace.claw_s, status, stop)
        stage_move(ser, 'go z=85', np.r_[pick_xy, zone.TRAVEL_Z], zone.CLOSED_CLAW,
                   'LIFT', pace.vertical_base_s +
                   abs(zone.TRAVEL_Z - zone.PICK_Z)*pace.vertical_per_mm_s,
                   status, stop)
        vision.xy_route(ser, pick_xy, dest_xy, R, t, zone.CLOSED_CLAW,
                        'MOVE_TO_TARGET', pace, progress=status.update,
                        cancel_event=stop)
        current = grip.checked_state(ser)
        if np.linalg.norm(current[:2] - dest_xy) > 3.0:
            raise RuntimeError('XY mismatch before placement DOWN')
        grip.near(current[3], zone.CLOSED_CLAW, 3.0, 'Claw before release')
        stage_move(ser, f'go z={place_z:g}', np.r_[dest_xy, place_z], zone.CLOSED_CLAW,
                   'LOWER_TO_PLACE', pace.vertical_base_s +
                   abs(zone.TRAVEL_Z - place_z)*pace.vertical_per_mm_s,
                   status, stop)
        stage_move(ser, 'go claw=50', np.r_[dest_xy, place_z], zone.OPEN_CLAW,
                   'RELEASE', pace.claw_s, status, stop)
        stage_move(ser, 'go z=85', np.r_[dest_xy, zone.TRAVEL_Z], zone.OPEN_CLAW,
                   'RETRACT', pace.vertical_base_s +
                   abs(zone.TRAVEL_Z - place_z)*pace.vertical_per_mm_s,
                   status, stop)
        # The user does not want strict/any XY comparison for HOME.
        # RETRACT already checked Z=85; keep safe route validation, but just
        # send timed HOME command(s) and then close the claw without readback.
        status.update('RETURN_HOME', 'Commanding saved HOME XY at Z=85 (no XY check)')
        return_home_without_xy_check(ser, dest_xy, home_xy, R, t, pace, status, stop)
        close_home_without_xy_check(ser, pace, status, stop)
        status.update('READY', 'HOME and CLOSE commands sent; awaiting next command')


def dry_preview(plan, scene, args):
    pick_xy = robot_xy(scene['source'])
    dst_xy = np.asarray(scene['dest_robot'])
    stack = plan['target']['type'] == 'on_cube'
    place_z = args.stack_place_z if stack else zone.PLACE_Z
    print('\nDRY RUN: DeepSeek JSON checked; NO SERIAL COMMANDS SENT.', flush=True)
    print('  Pick', scene['source']['color'], 'at', np.round(pick_xy, 1).tolist())
    print('  Right-of-HOME pickup correction: if Y <= HOME_Y - 25 mm, '
          'final +10 mm RoboArm Y at Z=85 before DOWN.')
    print('  Place at board', np.round(scene['dest_board'], 1).tolist(),
          'robot', np.round(dst_xy, 1).tolist(), 'Z', place_z)
    print('  PLAN:', ' -> '.join(plan['steps']))
    print('  Then RETURN_HOME: startup XY at Z=85, then CLOSE_HOME: go claw=0')
    for cmd in ('go claw=50', 'go x=<pickup X> y=<pickup Y>',
                '[right side only: go y=<pickup Y + 10 mm> at Z=85]', 'go z=15',
                'go claw=0', 'go z=85', 'go x=<target X> y=<target Y>',
                f'go z={float(place_z):g}', 'go claw=50', 'go z=85',
                'go x=<HOME X> y=<HOME Y>', 'go claw=0'):
        print('   ', cmd)
    if stack:
        print(f'STACK: release Z={float(place_z):g} is user-selected and NOT physically verified.')


class LiveStatus:
    def __init__(self):
        self._lock = threading.Lock()
        self._stage = 'WAIT COMMAND'
        self._message = ''
        self._started = time.monotonic()

    def update(self, stage: str, message: str):
        with self._lock:
            self._stage, self._message = stage, message
            self._started = time.monotonic()
        print('STAGE:', stage, '|', message, flush=True)

    def read(self):
        with self._lock:
            return self._stage, self._message, time.monotonic()-self._started


def keyboard_input(commands: queue.Queue, shutdown: threading.Event):
    print('\nEnter a Russian command in this terminal. ESC in OpenCV window = stop.')
    print('Example: возьми зеленый кубик и перенеси его в точку 1\n', flush=True)
    while not shutdown.is_set():
        try:
            phrase = input('Команда > ').strip()
        except (EOFError, KeyboardInterrupt):
            return
        if phrase:
            commands.put(phrase)


def draw_frame(frame, outline, H, objects, scene, status, busy, active_plan,
               drop_marker_px=None):
    canvas = frame.copy()
    cv2.polylines(canvas, [outline], True, (0, 0, 255), 2)
    if drop_marker_px is None:
        invH = np.linalg.inv(H)
        point = base.project(invH, [vision.DROP_BOARD_MM])[0]
        point = tuple(np.rint(point).astype(int))
    else:
        point = drop_marker_px
    cv2.drawMarker(canvas, point, (0, 230, 255), cv2.MARKER_CROSS, 26, 2)
    for obj in objects:
        x, y, w, h = obj['bbox']
        inside = obj.get('inside_workspace', True)
        rgb = vision.COLOR_BGR.get(obj['color'], (200, 200, 200)) if inside else (255, 0, 255)
        cv2.rectangle(canvas, (x,y), (x+w,y+h), rgb, 2)
        label = obj['color'].upper() + ('' if inside else ' OUTSIDE RED')
        base.draw_text(canvas, label, x, max(90,y-6), rgb, .44)
    if scene:
        p = tuple(np.rint(scene['source']['pixel']).astype(int))
        cv2.drawMarker(canvas, p, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        if scene['target'] is not None:
            p2 = tuple(np.rint(scene['target']['pixel']).astype(int))
            cv2.drawMarker(canvas, p2, (0, 255, 255), cv2.MARKER_CROSS, 21, 2)
    stage, message, elapsed = status.read()
    w, h = canvas.shape[1], canvas.shape[0]
    cv2.rectangle(canvas, (0,0), (w-1,84), (12,12,12), -1)
    base.draw_text(canvas, 'DeepSeek | ' + stage, 7, 20, (0,220,255), .53)
    base.draw_text(canvas, str(message)[:90], 7, 41, (235,235,235), .35)
    base.draw_text(canvas, f'{"BUSY" if busy else "READY"} | stage {elapsed:.1f}s | point 1 = (205,342.5)',
                   7, 64, (150,220,255), .36)
    if active_plan:
        base.draw_text(canvas, (f'{active_plan["pick_color"]} -> '
                   f'{active_plan["target"]["type"]}')[:65], 7, 81, (180,255,180), .31)
    cv2.rectangle(canvas, (0,h-26), (w-1,h-1), (10,10,10), -1)
    base.draw_text(canvas,'ESC requests stop AFTER current go; power cutoff = emergency stop',
                   5, h-8, (255,255,255), .30)
    return canvas


def main(argv=None):
    ap = argparse.ArgumentParser(description='RoboArm R1 voice/text + Vosk + DeepSeek')
    ap.add_argument('--input', choices=('voice', 'text'), default='voice',
                    help='voice (default: requires Robot wake word) or legacy keyboard text')
    ap.add_argument('--list-mics', action='store_true',
                    help='list PortAudio INPUT indices; no DeepSeek/camera/robot required')
    ap.add_argument('--check-mic', action='store_true',
                    help='Vosk + PCM2902 standalone test; no DeepSeek/camera/robot required')
    ap.add_argument('--silence-seconds', type=float, default=3.0,
                    help='execute voice command after N seconds without new Vosk text (default 3)')
    ap.add_argument('--silence-rms', type=float, default=240.0,
                    help='legacy option retained for compatibility; no longer blocks Vosk command completion')
    ap.add_argument('--mic-device', default=None,
                    help='PortAudio input INDEX (not lsusb Bus/Device); auto-detect PCM2902 if omitted')
    ap.add_argument('--vosk-model', default='vosk-model-small-ru-0.22',
                    help='path to unpacked Russian Vosk model folder')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--vision-fps', type=float, default=8.0,
                    help='max costly OpenCV object+ArUco scans per second during task (default 8)')
    ap.add_argument('--preview-fps', type=float, default=2.0,
                    help='OpenCV preview scans/sec when waiting for voice (default 2)')
    ap.add_argument('--display-fps', type=float, default=20.0,
                    help='live camera window refresh rate (default 20); separate from vision-fps')
    ap.add_argument('--opencv-threads', type=int, default=1,
                    help='limit OpenCV internal worker threads (default 1)')
    ap.add_argument('--cpu-stats', action='store_true',
                    help='print measured process CPU and CV scan cost every ~10s')
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--execute', action='store_true', help='enable real servo movement (default DRY RUN)')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--pace', choices=sorted(vision.PACE_PROFILES), default='fast')
    ap.add_argument('--board', type=Path, default=Path('board_calibration.npz'))
    ap.add_argument('--fit', type=Path, default=Path('robot_alignment_far_DIAGNOSTIC.npz'))
    ap.add_argument('--min-area', type=float, default=150.0)
    ap.add_argument('--vision-timeout', type=float, default=VISION_TIMEOUT_S,
                    help='seconds to find/validate named cube(s) before aborting task and listening again (default 8)')
    ap.add_argument('--stack-place-z', type=float, default=STACK_PLACE_Z_DEFAULT,
                    help='release Z for placing on a cube (default 40 mm; not physically verified)')
    args = ap.parse_args(argv)
    # Audio diagnostics work without credentials, calibration files or motors.
    if args.list_mics:
        from robot_microphone import RobotMicrophone
        RobotMicrophone.list_devices()
        return
    if args.check_mic:
        from roboarm_voice_input import microphone_check
        microphone_check(args.vosk_model, args.mic_device, args.silence_seconds, args.silence_rms)
        return
    if not np.isfinite(args.silence_seconds) or not 1.0 <= args.silence_seconds <= 10.0:
        ap.error('--silence-seconds must be 1..10')
    if not np.isfinite(args.silence_rms) or not 20 <= args.silence_rms <= 5000:
        ap.error('--silence-rms must be 20..5000')
    if args.input == 'voice' and not Path(args.vosk_model).is_dir():
        ap.error(f'Vosk model directory not found: {args.vosk_model}')
    if (not 2 <= args.vision_fps <= 20 or
            not 0 <= args.preview_fps <= args.vision_fps or
            not 5 <= args.display_fps <= 30 or
            not 1 <= args.opencv_threads <= 4):
        ap.error('FPS: vision 2..20, preview 0..vision, display 5..30; OpenCV threads 1..4')
    cv2.setNumThreads(args.opencv_threads)
    if args.min_area < 50:
        ap.error('--min-area must be >= 50')
    if not np.isfinite(args.vision_timeout) or not 2.0 <= args.vision_timeout <= 120.0:
        ap.error('--vision-timeout must be 2..120 seconds')
    if not np.isfinite(args.stack_place_z) or not 35.0 <= args.stack_place_z <= 55.0:
        ap.error('--stack-place-z must be 35..55 mm (still requires physical calibration)')
    if not os.environ.get('DEEPSEEK_API_KEY'):
        ap.error('DEEPSEEK_API_KEY is not configured (API call required even in dry-run)')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('Camera not available')
    # Most V4L2 drivers accept a one-frame buffer; some ignore it. A small
    # buffer reduces risk of delayed video when UI FPS is lower than sensor FPS.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    status = LiveStatus()
    command_queue = queue.Queue(maxsize=1 if args.input == 'voice' else 0)
    shutdown = threading.Event()
    voice_busy = threading.Event()
    worker_stop = threading.Event()
    home_anchor = HomeAnchor()
    motion_fault = False  # On error, block new motion until process restart.
    completed_tasks = 0
    if args.input == 'voice':
        from roboarm_voice_input import run_voice_input
        input_thread = threading.Thread(
            target=run_voice_input,
            args=(command_queue, shutdown, voice_busy, args.vosk_model,
                  args.mic_device, status, None, args.silence_seconds, args.silence_rms), name='roboarm-vosk', daemon=True)
    else:
        input_thread = threading.Thread(target=keyboard_input,
                                        args=(command_queue, shutdown),
                                        name='roboarm-stdin', daemon=True)
    planner_future = None
    motion_future = None
    plan = None
    fixed_scene = None
    start_confirm = None
    vision_started = None
    last_vision_reason = ''
    samples = deque(maxlen=vision.STABLE_FRAMES)
    marker_monitor = MarkerStabilityMonitor(threshold_mm=MAX_MARKER_DRIFT_MM)
    print(f'ACTIVE ARUCO LIMIT: < {marker_monitor.threshold_mm:g} mm | '
          f'{marker_monitor.good_frames}/{marker_monitor.window} frames | '
          f'{marker_monitor.min_markers} stable IDs | '
          f'program: {Path(__file__).resolve()}', flush=True)
    last_scene = None
    last_objects = []
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Cannot read camera')
        shape = frame.shape
        H, _ = base.load_board(args.board, shape)
        R, t, align_source = base.load_alignment(args.fit, args.execute)
        mask, outline = vision.make_work_mask(H, shape)
        board_search_mask = make_board_search_mask(H, shape)
        drop_marker_px = tuple(np.rint(base.project(
            np.linalg.inv(H), [vision.DROP_BOARD_MM])[0]).astype(int))
        last_vision_at = float('-inf')
        last_markers = {}
        cv_stats = CpuDiagnostics(args.cpu_stats)
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())
        print('Calibration:', args.board, align_source)
        print('Mode:', 'EXECUTE' if args.execute else 'DRY RUN',
              '| input:', args.input,
              '| stack enabled with --execute', '| stack release Z:', args.stack_place_z,
              '| vision timeout:', f'{args.vision_timeout:g}s')
        print(f'CPU mode: display <= {args.display_fps:g} fps, '
              f'vision <= {args.vision_fps:g} fps while planning, '
              f'preview <= {args.preview_fps:g} fps while idle, '
              f'vision OFF during motor movement, OpenCV threads={cv2.getNumThreads()}', flush=True)
        input_thread.start()
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        with ThreadPoolExecutor(max_workers=2) as pool:
            while not shutdown.is_set():
                ok, frame = cap.read()
                if not ok or frame.shape != shape:
                    worker_stop.set()
                    raise RuntimeError('Camera lost / resolution changed. Stop next go command.')

                if planner_future is not None and planner_future.done():
                    try:
                        plan = planner_future.result()
                        print('DEEPSEEK JSON:', json.dumps(plan, ensure_ascii=False, indent=2))
                        if plan['mode'] == 'reject':
                            status.update('REJECT', plan['reason'])
                            plan = None
                        else:
                            status.update('VISION', 'Looking for specified cube(s)')
                            marker_monitor.reset()  # fresh 5-frame check for each new task
                            samples.clear()
                            start_confirm = None
                            vision_started = time.monotonic()
                            last_vision_reason = ''
                    except Exception as exc:
                        status.update('API ERROR', str(exc))
                        plan = None
                        vision_started = None
                    planner_future = None

                if motion_future is not None and motion_future.done():
                    try:
                        motion_future.result()
                        completed_tasks += 1
                        status.update('VOICE WAIT' if args.input == 'voice' else 'WAIT COMMAND',
                            f'HOME and claw-close sent; {completed_tasks} task(s); '
                            + ('say ROBOT for next task' if args.input == 'voice'
                               else 'enter next command'))
                    except Exception as exc:
                        motion_fault = True
                        status.update('MOTION FAULT', str(exc)[:95])
                        print('ROBOT ERROR:', repr(exc), file=sys.stderr)
                        print('Motion locked until program restart: robot might hold a cube, '
                              'be lowered, or not be at HOME.', file=sys.stderr)
                    finally:
                        if args.input == 'voice':
                            # Never execute voice commands recorded while the
                            # gripper was moving (stale audio queued just as
                            # motion completed).
                            while True:
                                try:
                                    command_queue.get_nowait()
                                except queue.Empty:
                                    break
                        motion_future = None
                        fixed_scene = None
                        plan = None
                        vision_started = None
                        samples.clear()
                        start_confirm = None
                        last_scene = None

                busy = planner_future is not None or motion_future is not None or plan is not None
                if args.input == 'voice':
                    if busy or motion_fault:
                        voice_busy.set()
                        # Do not defer utterances made during another task.
                        while True:
                            try:
                                command_queue.get_nowait()
                            except queue.Empty:
                                break
                    else:
                        voice_busy.clear()
                if not busy and not motion_fault:
                    try:
                        user_text = command_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        if args.input == 'voice':
                            voice_busy.set()  # prevent second command during API call
                        print('\nUSER:', user_text, flush=True)
                        status.update('DEEPSEEK', 'Converting text into JSON action plan')
                        planner_future = pool.submit(deepseek_plan, user_text, args.model)

                # Live camera runs at display-fps; heavy HSV/ArUco scanning is
                # much slower when idle and completely skipped while motors or
                # DeepSeek are busy. This avoids starving PortAudio/Vosk.
                vision_active = (plan is not None and motion_future is None
                                 and planner_future is None and not motion_fault)
                scan_hz = (args.vision_fps if vision_active else
                           args.preview_fps if (motion_future is None
                                                and planner_future is None
                                                and not motion_fault) else 0.0)
                now_scan = time.monotonic()
                vision_fresh = False
                if perception_due(now_scan, last_vision_at, scan_hz):
                    tick_vision = time.perf_counter()
                    last_markers, marker_corners = detect_marker_data(frame, detector)
                    last_objects = detect_board_objects(
                        frame, H, R, t, detector, args.min_area,
                        board_search_mask, marker_corners=marker_corners)
                    last_vision_at = now_scan
                    vision_fresh = True
                    cv_stats.add_vision(time.perf_counter() - tick_vision)

                if vision_active and vision_fresh:
                    scene, _, refs_ok, message = select_scene(
                        frame, plan, detector, H, R, t, mask, args.min_area,
                        marker_monitor=marker_monitor, all_objects=last_objects,
                        observed_markers=last_markers)
                    last_vision_reason = message
                    last_scene = scene
                    if scene is None:
                        samples.clear()
                        start_confirm = None
                        status.update('VISION WAIT', message) if status.read()[1] != message else None
                    else:
                        sig = (scene['source'], scene['target'])
                        if samples:
                            previous = samples[-1]
                            stable = same_scene(previous[0], sig[0]) and (
                                (sig[1] is None and previous[1] is None) or
                                (sig[1] is not None and previous[1] is not None
                                 and same_scene(previous[1], sig[1])))
                            if not stable:
                                samples.clear()
                                start_confirm = None
                        samples.append(sig)
                        if len(samples) == vision.STABLE_FRAMES:
                            if start_confirm is None:
                                start_confirm = time.monotonic()
                                status.update('TARGET STABLE', 'Waiting 1 sec before motion')
                            if time.monotonic() - start_confirm >= 1.0:
                                # Re-run geometric checks at last observed frame.
                                fixed_scene = scene
                                if not args.execute:
                                    dry_preview(plan, fixed_scene, args)
                                    status.update('DRY DONE', 'No motor commands sent; say ROBOT again'
                                                  if args.input == 'voice'
                                                  else 'No motor commands sent; enter next command')
                                    plan = None
                                else:
                                    status.update('MOTION', 'Starting validated serial sequence')
                                    motion_future = pool.submit(verified_motion, plan,
                                        fixed_scene, args, R, t, status, worker_stop, home_anchor)
                                samples.clear()
                                start_confirm = None
                                vision_started = None

                    # Do not wait indefinitely for a non-existent or out-of-zone cube.
                    # Timeout NEVER opens Serial or sends commands. Voice re-arms on
                    # the next main-loop iteration for another 'Робот, ...' phrase.
                    if plan is not None and vision_wait_expired(
                            vision_started, args.vision_timeout):
                        reason = last_vision_reason or '[VISION_TIMEOUT] target was not validated'
                        status.update('VISION REJECT',
                            f'No movement: {reason}; ready for a NEW command')
                        print(f'VISION REJECT after {args.vision_timeout:g}s: '
                              f'{reason}. NO MOTOR COMMANDS SENT.', flush=True)
                        plan = None
                        fixed_scene = None
                        last_scene = None
                        vision_started = None
                        samples.clear()
                        start_confirm = None

                busy = planner_future is not None or motion_future is not None or plan is not None
                canvas = draw_frame(frame, outline, H, last_objects,
                                    fixed_scene or last_scene, status, busy, plan,
                                    drop_marker_px=drop_marker_px)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xff
                cv_stats.report()
                # Bound the live preview loop even on cameras that return
                # cached frames immediately. ESC is checked every iteration.
                remaining = 1.0 / args.display_fps - (time.monotonic() - now_scan)
                if remaining > 0:
                    time.sleep(remaining)
                if key == 27:
                    worker_stop.set()
                    shutdown.set()
                    status.update('STOP REQUESTED', 'ESC: no subsequent go commands')
                    break
            if shutdown.is_set():
                worker_stop.set()
            if motion_future is not None:
                print('ESC: waiting for current serial command to finish; physical power-off for emergency.', flush=True)
                while not motion_future.done():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    canvas = draw_frame(frame, outline, H, last_objects,
                                        fixed_scene, status, True, plan,
                                        drop_marker_px=drop_marker_px)
                    cv2.imshow(WINDOW, canvas)
                    cv2.waitKey(30)
                # All submitted futures are joined by executor context on exit.
    finally:
        shutdown.set()
        worker_stop.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print('STOPPED:', exc, file=sys.stderr)
        raise SystemExit(1)
