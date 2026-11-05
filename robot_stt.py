"""Stream Russian speech using an offline Vosk model.

Vosk FINALs may be only *parts* of a spoken command. The caller accumulates
multiple FINALs and the latest PARTIAL until an explicit end-of-command event.
"""
import json
from pathlib import Path
from vosk import Model, KaldiRecognizer, SetLogLevel


class RobotSTT:
    def __init__(self, model_path, sample_rate):
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f'Папка модели Vosk не найдена: {model_path}')
        SetLogLevel(-1)
        self.model = Model(str(model_path))
        self.sample_rate = int(sample_rate)
        self.reset()

    def reset(self):
        self.rec = KaldiRecognizer(self.model, self.sample_rate)
        self.rec.SetWords(True)

    def feed(self, chunk):
        if self.rec.AcceptWaveform(chunk):
            data = json.loads(self.rec.Result())
            return 'final', data.get('text', '').strip()
        data = json.loads(self.rec.PartialResult())
        return 'partial', data.get('partial', '').strip()

    def finish(self):
        """Flush remaining uncommitted words at an actual command boundary."""
        return json.loads(self.rec.FinalResult()).get('text', '').strip()
