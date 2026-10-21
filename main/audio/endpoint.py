from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum


class EndReason(StrEnum):
    SILENCE = "silence"
    TIMEOUT = "timeout"
    MAX_DURATION = "max_duration"


@dataclass(frozen=True)
class EndpointConfig:
    start_frames: int = 3
    silence_frames: int = 100
    no_speech_frames: int = 333
    max_frames: int = 1000
    pre_roll_frames: int = 10


class Endpointer:
    """Pure frame-based speech endpointing state machine."""

    def __init__(self, config: EndpointConfig | None = None) -> None:
        self.config = config or EndpointConfig()
        self._pre_roll: deque[bytes] = deque(maxlen=self.config.pre_roll_frames)
        self._frames: list[bytes] = []
        self._waiting_frames = 0
        self._speech_run = 0
        self._silence_run = 0
        self._speech_frames = 0
        self.started = False
        self.done = False

    def feed(self, frame: bytes, is_speech: bool) -> EndReason | None:
        if self.done:
            return None

        if not self.started:
            self._pre_roll.append(frame)
            self._waiting_frames += 1
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= self.config.start_frames:
                self.started = True
                self._frames = list(self._pre_roll)
                self._speech_frames = len(self._frames)
            elif self._waiting_frames >= self.config.no_speech_frames:
                self.done = True
                return EndReason.TIMEOUT
            return None

        self._frames.append(frame)
        self._speech_frames += 1
        self._silence_run = 0 if is_speech else self._silence_run + 1
        if self._silence_run >= self.config.silence_frames:
            self.done = True
            return EndReason.SILENCE
        if self._speech_frames >= self.config.max_frames:
            self.done = True
            return EndReason.MAX_DURATION
        return None

    def audio(self) -> bytes:
        return b"".join(self._frames)
