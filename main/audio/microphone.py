from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable

import numpy as np

from main.audio.endpoint import Endpointer

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


class NoInputDeviceError(Exception):
    """No usable default input device is available."""


class AudioReadError(Exception):
    """The microphone stream failed after it was opened."""


class CaptureCancelled(Exception):
    """The user cancelled an in-progress capture."""


class Microphone:
    """Records 16 kHz mono speech, stopping on silence."""

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._stop = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def stop(self) -> None:
        self._stop.set()

    def record(self, on_level: Callable[[float], None] | None = None) -> np.ndarray:
        import sounddevice as sd
        import webrtcvad

        try:
            sd.query_devices(kind="input")
        except Exception as exc:
            raise NoInputDeviceError(str(exc)) from exc

        vad = webrtcvad.Vad(3)
        endpointer = Endpointer()

        stream = None
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
            )
            stream.start()
        except Exception as exc:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
            raise NoInputDeviceError(str(exc)) from exc

        cancelled = False
        try:
            while not self._cancel.is_set() and not self._stop.is_set():
                data, _overflowed = stream.read(FRAME_SAMPLES)
                frame = data.tobytes()
                if on_level is not None:
                    level = float(np.sqrt(np.mean(data.astype(np.float32) ** 2))) / 32768.0
                    on_level(min(level, 1.0))
                if endpointer.feed(frame, vad.is_speech(frame, SAMPLE_RATE)) is not None:
                    break
        except Exception as exc:
            raise AudioReadError(str(exc)) from exc
        finally:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()
            cancelled = self._cancel.is_set()
            self._cancel.clear()
            self._stop.clear()

        if cancelled:
            raise CaptureCancelled("capture cancelled")
        raw = endpointer.audio()
        if not raw:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
