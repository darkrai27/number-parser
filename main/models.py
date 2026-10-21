from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Reason(StrEnum):
    VALID = "valid"
    WRONG_LENGTH = "wrong_length"
    WRONG_PREFIX = "wrong_prefix"
    NOT_DIGITS = "not_digits"


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
    error: str | None = None
