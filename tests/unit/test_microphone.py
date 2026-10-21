from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from main.audio.microphone import FRAME_SAMPLES, Microphone


class _FakeStream:
    def __init__(self) -> None:
        self._speech_frames = 0

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        return np.zeros((frames, 1), dtype=np.int16), False


class _AlwaysSpeechVad:
    def __init__(self, mode: int) -> None:
        self._calls = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self._calls += 1
        return self._calls <= 3


def test_record_returns_float_audio_and_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(InputStream=lambda **kwargs: _FakeStream())
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=_AlwaysSpeechVad))

    levels: list[float] = []
    audio = Microphone().record(levels.append)

    assert audio.dtype == np.float32
    assert audio.size == (3 + 100) * FRAME_SAMPLES
    assert levels
