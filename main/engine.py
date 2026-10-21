from __future__ import annotations

from pathlib import Path

from main.asr.base import Transcriber
from main.models import RecognitionResult
from main.numbers.parser import parse_candidates


class RecognitionEngine:
    def __init__(self, transcriber: Transcriber) -> None:
        self._transcriber = transcriber

    def recognize_file(self, path: str | Path) -> RecognitionResult:
        source = Path(path)
        if not source.is_file():
            return RecognitionResult("", (), f"Audio file not found: {source}")

        try:
            transcript = self._transcriber.transcribe_file(source)
        except Exception as exc:
            return RecognitionResult("", (), f"Could not process audio: {exc}")

        candidates = []
        seen: set[str] = set()
        for text in (transcript.text, *transcript.alternatives):
            for candidate in parse_candidates(text):
                if candidate.digits not in seen:
                    seen.add(candidate.digits)
                    candidates.append(candidate)
        if not candidates:
            return RecognitionResult(
                transcript.text,
                (),
                "No telephone number recognized.",
            )
        return RecognitionResult(transcript.text, tuple(candidates))
