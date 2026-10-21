from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from main.audio.microphone import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioReadError,
    CaptureCancelled,
    Microphone,
    NoInputDeviceError,
)


class _FakeStream:
    def __init__(self) -> None:
        self._speech_frames = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

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


class _FailingStream:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> _FailingStream:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        raise RuntimeError("stream died")


class _UnopenableStream(_FailingStream):
    def start(self) -> None:
        raise OSError("permission denied")


class _CancelStream(_FakeStream):
    def __init__(self, microphone: Microphone) -> None:
        super().__init__()
        self._microphone = microphone
        self.closed = False

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        self._microphone.cancel()
        return np.zeros((frames, 1), dtype=np.int16), False

    def close(self) -> None:
        self.closed = True


def test_record_returns_float_audio_and_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: dict[str, object] = {}

    def open_stream(**kwargs: object) -> _FakeStream:
        opened.update(kwargs)
        return _FakeStream()

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(query_devices=lambda **kwargs: {}, InputStream=open_stream),
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=_AlwaysSpeechVad))

    levels: list[float] = []
    audio = Microphone().record(levels.append)

    assert audio.dtype == np.float32
    assert audio.size == (3 + 100) * FRAME_SAMPLES
    assert levels
    assert opened == {
        "samplerate": SAMPLE_RATE,
        "channels": 1,
        "dtype": "int16",
        "blocksize": FRAME_SAMPLES,
    }


def test_missing_input_device_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_device(**kwargs: object) -> None:
        raise OSError("no default input")

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(query_devices=no_device),
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=_AlwaysSpeechVad))

    with pytest.raises(NoInputDeviceError, match="no default input"):
        Microphone().record()


def test_stream_failure_raises_audio_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_devices=lambda **kwargs: {}, InputStream=lambda **kwargs: _FailingStream()
        ),
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=_AlwaysSpeechVad))

    with pytest.raises(AudioReadError, match="stream died"):
        Microphone().record()


def test_unopenable_stream_raises_no_input_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_devices=lambda **kwargs: {}, InputStream=lambda **kwargs: _UnopenableStream()
        ),
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=_AlwaysSpeechVad))

    with pytest.raises(NoInputDeviceError, match="permission denied"):
        Microphone().record()


def test_cancel_during_capture_raises_capture_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    microphone = Microphone()
    stream = _CancelStream(microphone)
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(query_devices=lambda **kwargs: {}, InputStream=lambda **kwargs: stream),
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=_AlwaysSpeechVad))

    with pytest.raises(CaptureCancelled):
        microphone.record()

    assert stream.closed


def test_cancel_before_capture_is_not_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    microphone = Microphone()
    microphone.cancel()
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_devices=lambda **kwargs: {}, InputStream=lambda **kwargs: _FakeStream()
        ),
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=_AlwaysSpeechVad))

    with pytest.raises(CaptureCancelled):
        microphone.record()

    audio = microphone.record()
    assert audio.size == (3 + 100) * FRAME_SAMPLES
