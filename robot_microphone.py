"""Reliable USB microphone capture for RoboArm + Vosk.

The PortAudio callback must remain *tiny*.  PCM16 at the device's native rate
(usually 44100 Hz for PCM2902) is captured on PortAudio's thread.  The consumer
thread converts to 16000-Hz mono PCM16 with a **stateful** SoXR resampler.  It
never pretends that samples were captured when PortAudio reports an overflow.

Install: pip install sounddevice numpy soxr
PortAudio device indices are not the USB Bus/Device numbers from lsusb.
"""
from __future__ import annotations

import queue
import time

import numpy as np


def _sd():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError('Install sounddevice: python3 -m pip install sounddevice') from exc
    return sd


class RobotMicrophone:
    """Convert a live USB microphone to mono PCM16 at 16000 Hz for Vosk."""

    def __init__(self, device=None, sample_rate=None, blocksize=0, queue_size=128):
        sd = _sd()
        self.device = self._find_device(device)
        info = sd.query_devices(self.device, 'input')
        channels_available = int(info['max_input_channels'])
        if channels_available < 1:
            raise RuntimeError('The selected device has no input channel')

        native_rate = int(info['default_samplerate'])
        rates = [int(sample_rate)] if sample_rate is not None else [native_rate, 48000, 44100, 16000]
        self.capture_rate = None
        self.channels = None
        errors = []
        for rate in dict.fromkeys(rates):
            for ch in (1, 2):
                if ch > channels_available:
                    continue
                try:
                    sd.check_input_settings(device=self.device, channels=ch,
                                            dtype='int16', samplerate=rate)
                    self.capture_rate, self.channels = rate, ch
                    break
                except Exception as exc:
                    errors.append(f'{rate}Hz/{ch}ch: {exc}')
            if self.capture_rate is not None:
                break
        if self.capture_rate is None:
            raise RuntimeError('Cannot open PCM16 audio input: ' + '; '.join(errors[:4]))

        # Vosk receives a 16k stream; never label 44100Hz samples as 16000Hz.
        self.sample_rate = 16000
        self._soxr_stream = None
        self._soxr_class = None
        if self.capture_rate != self.sample_rate:
            try:
                import soxr
            except ImportError as exc:
                raise RuntimeError(
                    'PCM2902 captures at ' + str(self.capture_rate) +
                    'Hz; stream conversion to 16kHz requires: python3 -m pip install soxr') from exc
            self._soxr_class = soxr.ResampleStream
            self._reset_resampler()

        self.blocksize = int(blocksize)  # 0 = PortAudio/host chooses optimal callback size
        self.q = queue.Queue(maxsize=queue_size)
        self.stream = None
        self.dropped = 0  # dropped software-queue chunks, lifetime total
        self.input_status_count = 0  # statuses, lifetime total
        self.input_overflow_count = 0  # hardware/PortAudio overflow callback events
        self.last_input_status = ''
        self.loss_epoch = 0  # increment every time the stream continuity is compromised
        self._reported_overflows = 0
        self._reported_dropped = 0
        self._status_reported_at = time.monotonic()

    @staticmethod
    def list_devices():
        sd = _sd()
        print('PortAudio recording devices (use INDEX, not lsusb Bus/Device):')
        for i, info in enumerate(sd.query_devices()):
            if int(info['max_input_channels']) > 0:
                print(f'  {i:>2}: {info["name"]}  channels={info["max_input_channels"]} '
                      f'default={info["default_samplerate"]:.0f}Hz')

    @staticmethod
    def _find_device(device):
        sd = _sd()
        devices = sd.query_devices()
        if device is not None:
            requested = str(device).strip()
            if requested.isdecimal():
                index = int(requested)
                if not 0 <= index < len(devices) or int(devices[index]['max_input_channels']) < 1:
                    raise RuntimeError(f'Invalid PortAudio input device: {index}')
                return index
            matches = [(i, d) for i, d in enumerate(devices)
                       if int(d['max_input_channels']) > 0 and requested.casefold() in d['name'].casefold()]
            if len(matches) == 1:
                return matches[0][0]
            raise RuntimeError(f'Microphone name "{requested}" matches {len(matches)} devices; '
                               'use --list-mics and --mic-device INDEX')
        for needle in ('pcm2902', 'usb audio codec', 'usb audio cod', 'texas instruments'):
            matches = [(i, d) for i, d in enumerate(devices)
                       if int(d['max_input_channels']) > 0 and needle in d['name'].casefold()]
            if len(matches) == 1:
                return matches[0][0]
            if len(matches) > 1:
                raise RuntimeError('Ambiguous PCM2902 input. Use --list-mics and --mic-device INDEX')
        raise RuntimeError('PCM2902 not automatically found; use --list-mics and --mic-device INDEX')

    def _reset_resampler(self):
        if self._soxr_class is not None:
            self._soxr_stream = self._soxr_class(
                self.capture_rate, self.sample_rate, 1, dtype='int16', quality='MQ')

    def _callback(self, indata, frames, time_info, status):
        # No printing, resampling, Vosk, ndarray allocation or blocking operations here.
        if status:
            self.input_status_count += 1
            self.last_input_status = str(status)
            if bool(getattr(status, 'input_overflow', False)):
                self.input_overflow_count += 1
                self.loss_epoch += 1
        try:
            self.q.put_nowait(bytes(indata))
        except queue.Full:
            # Never allow unlimited delay. The current utterance is unsafe once
            # any audio chunk has gone missing: the consumer resets its command.
            self.dropped += 1
            self.loss_epoch += 1
            try:
                self.q.get_nowait()  # discard oldest; keep more recent audio
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(bytes(indata))
            except queue.Full:
                self.dropped += 1
                self.loss_epoch += 1

    def __enter__(self):
        sd = _sd()
        self.stream = sd.RawInputStream(
            samplerate=self.capture_rate, blocksize=self.blocksize,
            device=self.device, dtype='int16', channels=self.channels,
            callback=self._callback, latency='high')
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def discard_pending(self):
        """Drop stale audio after a discontinuity; run in consumer thread only."""
        count = 0
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
            count += 1
        self._reset_resampler()  # no samples from a broken utterance leak through
        return count

    def report_losses(self, force=False):
        """Log **new** loss events, not lifetime cumulative counts."""
        now = time.monotonic()
        if not force and now - self._status_reported_at < 5.0:
            return
        n_overflows = self.input_overflow_count - self._reported_overflows
        n_drops = self.dropped - self._reported_dropped
        if n_overflows or n_drops:
            print(f'AUDIO LOSS: +{n_overflows} PortAudio input overflows, '
                  f'+{n_drops} software queue drops in last interval '
                  f'(total {self.input_overflow_count}/{self.dropped}); '
                  f'last PortAudio status: {self.last_input_status}', flush=True)
        self._reported_overflows = self.input_overflow_count
        self._reported_dropped = self.dropped
        self._status_reported_at = now

    def _convert_to_mono(self, raw):
        if self.channels == 1:
            return raw
        samples = np.frombuffer(raw, dtype='<i2').reshape(-1, self.channels)
        # 32-bit accumulator prevents overflow when L+R > int16 max.
        mono = (samples.astype(np.int32).sum(axis=1) // self.channels).astype('<i2')
        return mono.tobytes()

    def _resample(self, raw):
        if self._soxr_stream is None:
            return raw
        samples = np.frombuffer(raw, dtype='<i2')
        converted = self._soxr_stream.resample_chunk(samples, last=False)
        return converted.astype('<i2', copy=False).tobytes()

    def chunks(self, stop=None):
        while stop is None or not stop.is_set():
            try:
                raw = self.q.get(timeout=0.15)
            except queue.Empty:
                self.report_losses()
                continue
            self.report_losses()
            if not raw:
                continue
            out = self._resample(self._convert_to_mono(raw))
            if out:
                yield out
