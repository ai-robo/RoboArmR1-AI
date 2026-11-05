"""USB microphone input (PCM2902 / generic USB Audio CODEC) using sounddevice.

Based on the user's ReSpeaker RobotMicrophone. PortAudio device index is NOT
Linux lsusb's Bus/Device number. Select with --mic-device after --list-mics.
Produces little-endian mono signed 16-bit PCM for Vosk. Stereo-only audio
interfaces are downmixed safely to mono in the callback.
"""
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
    def __init__(self, device=None, sample_rate=None, blocksize=4000, queue_size=160):
        sd = _sd()
        self.device = self._find_device(device)
        info = sd.query_devices(self.device, 'input')
        if int(info['max_input_channels']) < 1:
            raise RuntimeError('Selected audio device has no input channel')
        rates = [int(sample_rate)] if sample_rate else [int(info['default_samplerate']), 48000, 44100, 16000]
        self.sample_rate = None
        self.channels = None
        tried = []
        for rate in dict.fromkeys(rates):
            for channels in (1, 2):
                if int(info['max_input_channels']) < channels:
                    continue
                try:
                    sd.check_input_settings(device=self.device, channels=channels, dtype='int16', samplerate=rate)
                    self.sample_rate = rate
                    self.channels = channels
                    break
                except Exception as exc:
                    tried.append(f'{rate} Hz / {channels} ch: {exc}')
            if self.sample_rate is not None:
                break
        if self.sample_rate is None:
            raise RuntimeError('PCM16 audio input cannot be opened. ' + '; '.join(tried[:4]))
        self.blocksize = blocksize
        self.q = queue.Queue(maxsize=queue_size)
        self.stream = None
        self.dropped = 0
        self.input_status_count = 0
        self.last_input_status = ''
        self._status_reported_at = 0.0
        self._status_reported_count = 0

    @staticmethod
    def list_devices():
        sd = _sd()
        print('PortAudio recording devices (use the INDEX, not USB Bus/Device):')
        for index, info in enumerate(sd.query_devices()):
            if info['max_input_channels'] > 0:
                print(f'  {index:>2}: {info["name"]}  channels={info["max_input_channels"]} '
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
                    raise RuntimeError(f'Invalid PortAudio input device index: {index}')
                return index
            matches = [(i, d) for i, d in enumerate(devices)
                       if int(d['max_input_channels']) > 0 and requested.casefold() in d['name'].casefold()]
            if len(matches) == 1:
                return matches[0][0]
            raise RuntimeError(f'Microphone name "{requested}" matches {len(matches)} devices; '
                               'use --list-mics then --mic-device INDEX')
        # Prefer the exact USB codec identified by the user. Some Linux ALSA/Pulse
        # stacks report the same PCM2902 under multiple names; require explicit index
        # when identification is ambiguous rather than opening an arbitrary mic.
        for needle in ('pcm2902', 'usb audio codec', 'usb audio cod', 'texas instruments'):
            matches = [(i, d) for i, d in enumerate(devices)
                       if int(d['max_input_channels']) > 0 and needle in d['name'].casefold()]
            if len(matches) == 1:
                return matches[0][0]
            if len(matches) > 1:
                options = ', '.join(f'{i} ({d["name"]})' for i, d in matches)
                raise RuntimeError(f'Multiple possible PCM2902 inputs: {options}. '
                                   'Choose explicitly using --mic-device INDEX')
        raise RuntimeError('PCM2902 / USB Audio CODEC not auto-detected. '
                           'Use --list-mics and --mic-device INDEX')

    def _callback(self, indata, frames, time_info, status):
        # Real-time PortAudio callback: do as little as possible.
        # numpy processing/printing here may itself cause INPUT OVERFLOW.
        if status:
            self.input_status_count += 1
            self.last_input_status = str(status)
        try:
            self.q.put_nowait(bytes(indata))
        except queue.Full:
            self.dropped += 1
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(bytes(indata))
            except queue.Full:
                self.dropped += 1

    def __enter__(self):
        sd = _sd()
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate, blocksize=self.blocksize, device=self.device,
            dtype='int16', channels=self.channels, callback=self._callback,
            latency='high')
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def chunks(self, stop=None):
        while stop is None or not stop.is_set():
            try:
                raw = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            # Report hardware/PortAudio losses outside the time-critical callback.
            now = time.monotonic()
            if (self.input_status_count > self._status_reported_count
                    and now - self._status_reported_at >= 5.0):
                print(f'AUDIO WARNING: {self.last_input_status} '
                      f'({self.input_status_count} callbacks affected); '
                      'PortAudio may have lost audio samples.', flush=True)
                self._status_reported_at = now
                self._status_reported_count = self.input_status_count
            if self.channels == 2:
                samples = np.frombuffer(raw, dtype='<i2').reshape(-1, 2)
                raw = ((samples[:, 0].astype(np.int32) +
                        samples[:, 1].astype(np.int32)) // 2).astype('<i2').tobytes()
            yield raw
