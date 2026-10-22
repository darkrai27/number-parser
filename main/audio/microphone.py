from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from main.audio.endpoint import FRAME_MS, SAMPLE_RATE, Endpointer

FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
READ_TIMEOUT_S = 1.0
_POLL_S = 0.1


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
        try:
            return self._capture(on_level)
        finally:
            # Always reset, even when the device never opened: a stale cancel/stop would
            # abort every later capture.
            self._cancel.clear()
            self._stop.clear()

    def _capture(self, on_level: Callable[[float], None] | None) -> np.ndarray:
        import sounddevice as sd
        import webrtcvad

        try:
            sd.query_devices(kind="input")
        except Exception as exc:
            raise NoInputDeviceError(str(exc)) from exc

        vad = webrtcvad.Vad(3)
        endpointer = Endpointer()
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            audio_queue.put(indata.copy())

        stream = None
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=callback,
            )
            stream.start()
        except Exception as exc:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
            raise NoInputDeviceError(str(exc)) from exc

        last_data = time.monotonic()
        try:
            while not self._cancel.is_set() and not self._stop.is_set():
                try:
                    data = audio_queue.get(timeout=_POLL_S)
                except queue.Empty:
                    if self._cancel.is_set() or self._stop.is_set():
                        break
                    if time.monotonic() - last_data >= READ_TIMEOUT_S:
                        raise AudioReadError("microphone read timed out") from None
                    continue
                last_data = time.monotonic()
                frame = data.tobytes()
                if on_level is not None:
                    level = float(np.sqrt(np.mean(data.astype(np.float32) ** 2))) / 32768.0
                    on_level(min(level, 1.0))
                if endpointer.feed(frame, vad.is_speech(frame, SAMPLE_RATE)) is not None:
                    break
        except AudioReadError:
            raise
        except Exception as exc:
            raise AudioReadError(str(exc)) from exc
        finally:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()

        if self._cancel.is_set():
            raise CaptureCancelled("capture cancelled")
        raw = endpointer.audio()
        if not raw:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
