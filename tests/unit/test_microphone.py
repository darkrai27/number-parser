from __future__ import annotations

import sys
import threading
import time
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
    def __init__(self, callback=None) -> None:
        self.callback = callback
        self.closed = False
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stopped.clear()
        if self.callback is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stopped.is_set():
            self.callback(np.zeros((FRAME_SAMPLES, 1), dtype=np.int16), FRAME_SAMPLES, None, None)
            time.sleep(0.001)

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(1.0)

    def close(self) -> None:
        self.closed = True


class _AlwaysSpeechVad:
    def __init__(self, mode: int) -> None:
        self._calls = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self._calls += 1
        return self._calls <= 3


class _UnopenableStream(_FakeStream):
    def start(self) -> None:
        raise OSError("permission denied")


class _CancelStream(_FakeStream):
    def __init__(self, microphone: Microphone) -> None:
        super().__init__()
        self._microphone = microphone

    def start(self) -> None:
        self._microphone.cancel()


class _StalledStream(_FakeStream):
    """Opens but never delivers audio."""

    def start(self) -> None:
        pass


def _install(monkeypatch: pytest.MonkeyPatch, stream: object) -> None:
    def open_stream(**kwargs: object) -> object:
        if hasattr(stream, "callback"):
            stream.callback = kwargs.get("callback")
        return stream

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(query_devices=lambda **kwargs: {}, InputStream=open_stream),
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=_AlwaysSpeechVad))


def test_record_returns_float_audio_and_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: dict[str, object] = {}

    def open_stream(**kwargs: object) -> _FakeStream:
        opened.update(kwargs)
        return _FakeStream(callback=kwargs.get("callback"))

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
    assert opened["samplerate"] == SAMPLE_RATE
    assert opened["channels"] == 1
    assert opened["dtype"] == "int16"
    assert opened["blocksize"] == FRAME_SAMPLES


def test_missing_input_device_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_device(**kwargs: object) -> None:
        raise OSError("no default input")

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(query_devices=no_device),
    )

    with pytest.raises(NoInputDeviceError, match="no default input"):
        Microphone().record()


def test_unopenable_stream_raises_no_input_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _UnopenableStream())

    with pytest.raises(NoInputDeviceError, match="permission denied"):
        Microphone().record()


def test_cancel_during_capture_raises_capture_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    microphone = Microphone()
    stream = _CancelStream(microphone)
    _install(monkeypatch, stream)

    with pytest.raises(CaptureCancelled):
        microphone.record()

    assert stream.closed


def test_cancel_before_capture_is_not_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    microphone = Microphone()
    microphone.cancel()
    _install(monkeypatch, _FakeStream())

    with pytest.raises(CaptureCancelled):
        microphone.record()

    audio = microphone.record()
    assert audio.size == (3 + 100) * FRAME_SAMPLES


def test_no_audio_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("main.audio.microphone.READ_TIMEOUT_S", 0.05)
    _install(monkeypatch, _StalledStream())

    with pytest.raises(AudioReadError, match="microphone read timed out"):
        Microphone().record()


def test_cancel_interrupts_a_stalled_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("main.audio.microphone.READ_TIMEOUT_S", 5.0)
    microphone = Microphone()
    _install(monkeypatch, _StalledStream())
    errors: list[Exception] = []

    def record() -> None:
        try:
            microphone.record()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=record)
    worker.start()
    time.sleep(0.2)
    microphone.cancel()
    worker.join(2.0)

    assert len(errors) == 1
    assert isinstance(errors[0], CaptureCancelled)


def test_failed_open_does_not_poison_the_next_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    microphone = Microphone()
    microphone.stop()
    streams = [_UnopenableStream(), _FakeStream()]

    def open_stream(**kwargs: object) -> object:
        stream = streams.pop(0)
        stream.callback = kwargs.get("callback")
        return stream

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(query_devices=lambda **kwargs: {}, InputStream=open_stream),
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=_AlwaysSpeechVad))

    with pytest.raises(NoInputDeviceError):
        microphone.record()

    audio = microphone.record()
    assert audio.size == (3 + 100) * FRAME_SAMPLES
