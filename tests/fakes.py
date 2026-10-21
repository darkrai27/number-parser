from __future__ import annotations

from pathlib import Path

import numpy as np

from main.models import Transcript


class FakeTranscriber:
    def __init__(self, *texts: str, alternatives: tuple[str, ...] = ()) -> None:
        self._texts = list(texts) or [""]
        self._alternatives = alternatives
        self.calls = 0

    def _next(self) -> Transcript:
        self.calls += 1
        text = self._texts[0]
        if len(self._texts) > 1:
            self._texts.pop(0)
        return Transcript(text=text, alternatives=self._alternatives)

    def transcribe_file(self, path: Path) -> Transcript:
        return self._next()

    def transcribe_audio(self, audio: np.ndarray) -> Transcript:
        return self._next()


class FailingTranscriber:
    def transcribe_file(self, path: Path) -> Transcript:
        raise RuntimeError("boom")

    def transcribe_audio(self, audio: np.ndarray) -> Transcript:
        raise RuntimeError("boom")
