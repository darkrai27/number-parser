from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Reason(StrEnum):
    VALID = "valid"
    WRONG_LENGTH = "wrong_length"
    WRONG_PREFIX = "wrong_prefix"
    NOT_DIGITS = "not_digits"


class ErrorCode(StrEnum):
    FILE_NOT_FOUND = "file_not_found"
    UNSUPPORTED_FILE = "unsupported_file"
    NO_INPUT_DEVICE = "no_input_device"
    AUDIO_READ_ERROR = "audio_read_error"
    NO_SPEECH = "no_speech"
    CANCELLED = "cancelled"
    TRANSCRIPTION_ERROR = "transcription_error"
    NO_NUMBER = "no_number"


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str = "en"
    duration_s: float = 0.0
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    digits: str
    input: str
    interpretation: str
    output: str | None
    is_valid: bool
    reason: Reason


@dataclass(frozen=True)
class RecognitionResult:
    transcript: str
    candidates: tuple[Candidate, ...]
    error: ErrorCode | None = None
    detail: str = ""
