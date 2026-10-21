from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from main.audio.endpoint import Endpointer

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


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

        vad = webrtcvad.Vad(2)
        endpointer = Endpointer()
        self._cancel.clear()
        self._stop.clear()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
        ) as stream:
            while not self._cancel.is_set() and not self._stop.is_set():
                data, _overflowed = stream.read(FRAME_SAMPLES)
                frame = data.tobytes()
                if on_level is not None:
                    level = float(np.sqrt(np.mean(data.astype(np.float32) ** 2))) / 32768.0
                    on_level(min(level, 1.0))
                if endpointer.feed(frame, vad.is_speech(frame, SAMPLE_RATE)) is not None:
                    break

        if self._cancel.is_set():
            return np.empty(0, dtype=np.float32)
        raw = endpointer.audio()
        if not raw:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
