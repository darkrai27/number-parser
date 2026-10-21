from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

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
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or RuntimeError("boom")

    def _fail(self) -> NoReturn:
        raise self._error

    def transcribe_file(self, path: Path) -> Transcript:
        self._fail()

    def transcribe_audio(self, audio: np.ndarray) -> Transcript:
        self._fail()


class FailingMicrophone:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or RuntimeError("no device")

    def record(self, on_level: Callable[[float], None] | None = None) -> np.ndarray:
        raise self._error

    def cancel(self) -> None:
        pass

    def stop(self) -> None:
        pass
