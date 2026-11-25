"""Vosk wake-word command collection for RoboArm R1.

One Vosk FINAL is NOT one user command. Keep accumulating FINAL segments
and the current PARTIAL until either:
  * three seconds WITHOUT NEW RECOGNIZED WORDS (regardless of microphone noise), or
  * the recognized word "выполняй" (FINAL or stable PARTIAL).
Why: constant ambient noise above the old fixed RMS threshold made commands
wait forever, even when Vosk had already produced all FINAL fragments.
Commands lacking the wake word or one of its approved Vosk variants are never submitted.
"""
import queue
import re
import time

import numpy as np

# Vosk occasionally recognizes the wake word with a different ending or
# consonant. Accept only a SMALL explicit list, not arbitrary fuzzy words:
# a false wake-up would otherwise cause an unintended robot movement.
WAKE_RE = re.compile(
    r'(?<!\w)(?:робот(?:а|у|ом|е|ик)?|робо|роб|робат|работа|работ)(?!\w)',
    re.I,
)
EXECUTE_RE = re.compile(r'(?<!\w)выполняй(?!\w)', re.I)

# A distorted wake word is permitted at the beginning of an utterance, or
# immediately after a short address. The ORIGINAL word 'робот' keeps its
# previous behaviour (can appear anywhere in the phrase).
_ADDRESS_RE = re.compile(
    r'^(?:(?:эй|ну|слушай|привет|алло|пожалуйста|слышь)\s*[,!?.;:—-]?\s*)*$',
    re.I,
)
# 'работа' is a common ordinary Russian word. For it to act as a wake word
# WITH OTHER WORDS in the same utterance, there must be a robot action.
# The word alone can arm a wake window, but cannot trigger any movement.
_ROBOT_ACTION_RE = re.compile(
    r'(?<!\w)(?:возьм\w*|взять|бери|найд\w*|ищи|полож\w*|'
    r'перенес\w*|переноси|перемест\w*|постав\w*|сделай|'
    r'отнес\w*|подним\w*|передвин\w*|убер\w*|'
    r'достань|сложи|помести|перелож\w*)(?!\w)',
    re.I,
)

# Normalize only unambiguous, explicitly supported Vosk substitutions,
# and ONLY in a command that has already passed the wake-word check.
# Avoid broad edit-distance correction of arbitrary Russian vocabulary.
_CUBE_ALIASES = {
    'киби': 'кубик', 'кибик': 'кубик', 'кибику': 'кубику',
    'куби': 'кубик', 'куб': 'кубик', 'кут': 'кубик',
    'кибика': 'кубика', 'кибиком': 'кубиком',
}
_CUBE_ALIAS_RE = re.compile(
    r'(?<!\w)(?:' + '|'.join(map(re.escape, sorted(_CUBE_ALIASES, key=len, reverse=True)))
    + r')(?!\w)', re.I,
)


def _wake_matches(text):
    """Return safe wake-word matches. False positives must not start a robot."""
    matches = []
    for match in WAKE_RE.finditer(text):
        word = match.group().casefold()
        if word == 'робот':
            matches.append(match)
            continue
        if not _ADDRESS_RE.fullmatch(text[:match.start()].strip()):
            continue
        remaining = text[match.end():].strip(' ,.!?;:-—\t')
        if word in ('работа', 'работ') and remaining and not _ROBOT_ACTION_RE.search(remaining):
            # 'работа выполнена' is not the wake phrase 'Робот'.
            continue
        matches.append(match)
    return matches


def normalize_command_body(text):
    """Fix known misrecognized cube names; never invent missing colors."""
    text = ' '.join(str(text).split()).strip()
    return _CUBE_ALIAS_RE.sub(lambda m: _CUBE_ALIASES[m.group().casefold()], text)
WAKE_WINDOW_SECONDS = 7.0
SILENCE_SECONDS = 3.0
# Signed PCM16 amplitude. If ambient noise is stronger, change --silence-rms.
DEFAULT_SILENCE_RMS = 240.0


class WakeWordGate:
    """Legacy one-final gate, retained for compatibility with earlier tests/tools."""
    def __init__(self, window_s=WAKE_WINDOW_SECONDS):
        self.window_s = float(window_s)
        self.armed_until = 0.0

    def feed_final(self, text, at=None):
        now = time.monotonic() if at is None else float(at)
        text = ' '.join(str(text).split()).strip()
        if not text:
            return 'ignored', None
        matches = _wake_matches(text)
        if matches:
            tail = text[matches[-1].end():].strip(' ,.!?;:-—\t')
            if tail:
                self.armed_until = 0
                return 'command', normalize_command_body(tail)
            self.armed_until = now + self.window_s
            return 'wake', None
        if now < self.armed_until:
            self.armed_until = 0
            return 'command', normalize_command_body(text)
        self.armed_until = 0
        return 'ignored', None


def _clean(text):
    return ' '.join(str(text).split()).strip()


def pcm16_rms(chunk):
    """Energy only; not voice recognition. Called outside the PortAudio callback."""
    samples = np.frombuffer(chunk, dtype='<i2')
    if not samples.size:
        return 0.0
    a = samples.astype(np.float32)
    return float(np.sqrt(np.mean(a * a)))


class SpokenCommandBuffer:
    """Collect Vosk segments until silence or explicit execution marker.

    `parts` are immutable Vosk FINAL segments; `partial` is the current
    replaceable, tentative Vosk partial (never append repeated partials).
    """
    def __init__(self, silence_s=SILENCE_SECONDS, wake_window_s=WAKE_WINDOW_SECONDS,
                 partial_execute_hold_s=0.3):
        self.silence_s = float(silence_s)
        self.wake_window_s = float(wake_window_s)
        self.partial_execute_hold_s = float(partial_execute_hold_s)
        self.reset()

    def reset(self):
        self.parts = []
        self.partial = ''
        self.last_sound_at = None
        self.last_text_change_at = None
        self.wake_only_at = None
        self.partial_execute_since = None

    def text(self):
        return _clean(' '.join(self.parts + ([self.partial] if self.partial else [])))

    def observe_audio(self, rms, at, threshold=DEFAULT_SILENCE_RMS):
        if float(rms) >= float(threshold):
            self.last_sound_at = float(at)

    def accept(self, kind, text, at):
        """Record a Vosk result; do not emit a movement command here."""
        text = _clean(text)
        now = float(at)
        if kind == 'final':
            if text:
                self.parts.append(text)
                self.last_text_change_at = now
                self.wake_only_at = None
            self.partial = ''
            self.partial_execute_since = None
        elif kind == 'partial':
            if text != self.partial:
                self.partial = text
                if text:
                    self.last_text_change_at = now
                    self.wake_only_at = None
                self.partial_execute_since = None
            # Require a stable provisional word for 0.3s. A transient Vosk
            # guess must not trigger hardware movements.
            if EXECUTE_RE.search(self.text()):
                if self.partial_execute_since is None:
                    self.partial_execute_since = now
            else:
                self.partial_execute_since = None

    def explicit_triggered(self, at):
        all_final = _clean(' '.join(self.parts))
        if EXECUTE_RE.search(all_final):
            return True
        return (EXECUTE_RE.search(self.text()) is not None
                and self.partial_execute_since is not None
                and float(at) - self.partial_execute_since >= self.partial_execute_hold_s)

    def quiet_long_enough(self, at):
        """Finish a phrase after N seconds without NEW Vosk text.

        Do NOT gate this on a fixed microphone RMS amplitude. A noisy USB
        microphone can exceed a constant 240-PCM16 threshold even in silence,
        resetting that timer indefinitely (the reported bug). Every new Vosk
        FINAL or *changed* PARTIAL restarts the clock; repeated partials do not.

        This is a recognizer-inactivity timeout, not an acoustic VAD guarantee.
        The explicit word "выполняй" remains available for precise boundaries.
        """
        last = self.last_text_change_at
        return (bool(self.text()) and last is not None
                and float(at) - last >= self.silence_s)

    def wake_expired(self, at):
        return self.wake_only_at is not None and float(at) - self.wake_only_at >= self.wake_window_s

    def decide(self, at):
        """Flush already-recognized text, return (event, command) on completion.

        `wake` means 'Робот' alone and no command yet; retain that token for
        another 7 seconds. A completed command resets the buffer.
        """
        now = float(at)
        content = self.text()
        if not content:
            return 'ignored', None
        if self.wake_expired(now):
            self.reset()
            return 'expired', None
        matches = _wake_matches(content)
        if not matches:
            self.reset()
            return 'ignored', None
        body = content[matches[-1].end():]
        # Stop at the first explicit execution word, not after it.
        execute = EXECUTE_RE.search(body)
        if execute:
            body = body[:execute.start()]
        body = body.strip(' ,.!?;:-—\t')
        if not body:
            # 'Робот' on its own is a wake request, never motor command.
            self.parts = ['робот']
            self.partial = ''
            self.wake_only_at = now
            self.last_sound_at = None
            self.last_text_change_at = None
            self.partial_execute_since = None
            return 'wake', None
        self.reset()
        return 'command', normalize_command_body(body)


def run_voice_input(commands, shutdown, busy, model_path, mic_device, status=None,
                    on_final=None, silence_s=SILENCE_SECONDS,
                    silence_rms=DEFAULT_SILENCE_RMS, check_only=False):
    """Always drain the mic. Never send speech to DeepSeek before an end marker."""
    from robot_microphone import RobotMicrophone
    from robot_stt import RobotSTT

    try:
        mic = RobotMicrophone(device=mic_device)
        stt = RobotSTT(model_path, mic.sample_rate)
        buffer = SpokenCommandBuffer(silence_s=silence_s)
        print(f'MIC: PortAudio device #{mic.device}, capture={getattr(mic, "capture_rate", mic.sample_rate)}Hz, '
              f'{mic.channels}ch -> Vosk={mic.sample_rate}Hz mono PCM16; '
              f'blocksize={getattr(mic, "blocksize", "unknown")}', flush=True)
        print(f'VOSK: say "Робот, <команда>"; finish by "Выполняй" OR '
              f'{buffer.silence_s:g}s without new recognized words. No wake word = no task.', flush=True)
        if status is not None:
            status.update('VOICE WAIT', 'Say ROBOT + instruction, then silence or EXECUTE')

        # Prevent leftover speech spoken during movement from starting later.
        suspended = busy.is_set()
        queued = False
        with mic:
            last_audio_loss_epoch = getattr(mic, 'loss_epoch', 0)
            for chunk in mic.chunks(stop=shutdown):
                now = time.monotonic()
                # A dropped PCM chunk can turn a valid instruction into a
                # different one. Never continue interpreting a broken phrase.
                if getattr(mic, 'loss_epoch', 0) != last_audio_loss_epoch:
                    last_audio_loss_epoch = getattr(mic, 'loss_epoch', 0)
                    skipped = mic.discard_pending()
                    buffer.reset()
                    stt.reset()
                    if not busy.is_set() and not queued:
                        print('VOICE RESET: input audio was incomplete; '
                              f'please repeat the command (discarded {skipped} queued chunks)',
                              flush=True)
                        if status is not None:
                            status.update('VOICE RESET', 'Audio overflow: repeat ROBOT + command')
                    continue
                # A standalone wake word expires after its own seven-second window.
                if buffer.wake_expired(now):
                    buffer.reset()
                if busy.is_set():
                    if not suspended:
                        print('VOICE IGNORE: task running, discarding speech', flush=True)
                    suspended = True
                    queued = False
                    buffer.reset()
                    # Audio is already drained from the mic queue. Decoding it
                    # with Vosk would waste a CPU core while motors are busy.
                    # Reset recognizer when listening resumes below.
                    continue
                if suspended:
                    suspended = False
                    buffer.reset()
                    stt.reset()
                if queued:
                    # Audio is drained but discarded until main controller
                    # picks the pending command. No ASR CPU work is needed.
                    continue

                # Audio-level RMS is intentionally NOT used to delay sending a
                # recognized command. A constant ambient level above the fixed
                # RMS threshold previously blocked the 3-second timer forever.
                # The last NEW Vosk FINAL/PARTIAL controls the inactivity timer.
                kind, recognized = stt.feed(chunk)
                buffer.accept(kind, recognized, now)
                if kind == 'final' and recognized:
                    print('VOSK FINAL PART:', recognized, flush=True)
                    if on_final is not None:
                        on_final('part', recognized)

                explicit = buffer.explicit_triggered(now)
                quiet = buffer.quiet_long_enough(now)
                if not explicit and not quiet:
                    continue
                # If the recognizer still holds a partial phrase, request its
                # remaining final words before handing off the command.
                # In the explicit-trigger case, retain an already-confirmed
                # partial as a fallback if FinalResult omits the last token.
                candidate_before_flush = buffer.text()
                # A stable PARTIAL containing "выполняй" is our explicit
                # terminator. Prefer its full command over a possibly shorter
                # Vosk FinalResult (which can occasionally omit the final word).
                partial_with_execute = (explicit and not EXECUTE_RE.search(
                    _clean(' '.join(buffer.parts))) and
                    EXECUTE_RE.search(candidate_before_flush))
                tail = '' if partial_with_execute else stt.finish()
                if tail:
                    buffer.accept('final', tail, now)
                    print('VOSK FINAL PART:', tail, flush=True)
                else:
                    # FinalResult sometimes yields an empty string despite a
                    # useful PartialResult. Keep the provisional recognition.
                    # No final segment was added: existing latest PARTIAL is
                    # already present, including a stable execute marker.
                    pass
                event, command = buffer.decide(now)
                if event == 'command':
                    print('VOICE COMMAND:', command,
                          '| trigger:', 'Выполняй' if explicit else f'{buffer.silence_s:g}s Vosk inactivity', flush=True)
                    if on_final is not None:
                        on_final(event, command)
                    if not check_only:
                        try:
                            commands.put_nowait(command)
                            queued = True
                        except queue.Full:
                            print('VOICE IGNORE: previous task still queued', flush=True)
                elif event == 'wake':
                    print('VOICE WAKE: heard "Робот"; waiting for the rest of command', flush=True)
                    if status is not None:
                        status.update('VOICE WAKE', 'Heard ROBOT; waiting for rest of command')
                else:
                    print('VOICE IGNORE: no current wake word "Робот"', flush=True)
                # Reset recognizer after each complete audio command so a
                # later FINAL cannot echo old text.
                stt.reset()
                if event == 'wake':
                    # Retain buffered wake word (and its seven-second TTL).
                    continue
                buffer.reset()
    except Exception as exc:
        print('MIC/VOSK ERROR:', type(exc).__name__, str(exc), flush=True)
        if status is not None:
            status.update('MIC ERROR', str(exc)[:85])
        shutdown.set()  # Fail closed: no future motion.


def microphone_check(model_path, device=None, silence_s=SILENCE_SECONDS,
                     silence_rms=DEFAULT_SILENCE_RMS):
    """Vosk test; nothing is sent to DeepSeek/camera/motors."""
    import threading
    stop = threading.Event()
    busy = threading.Event()
    out = queue.Queue(maxsize=1)
    print('MIC TEST ONLY: no DeepSeek or RoboArm. Ctrl+C to quit.', flush=True)
    try:
        run_voice_input(out, stop, busy, model_path, device,
                        silence_s=silence_s, silence_rms=silence_rms,
                        check_only=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
