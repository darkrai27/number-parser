from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from main.asr.base import Transcriber
from main.audio.microphone import Microphone
from main.models import RecognitionResult, Transcript
from main.numbers.parser import parse_candidates


class Recognizer(Protocol):
    def recognize_file(self, path: str | Path) -> RecognitionResult: ...

    def recognize_microphone(
        self,
        on_level: Callable[[float], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
    ) -> RecognitionResult: ...

    def cancel(self) -> None: ...

    def stop_recording(self) -> None: ...


class RecognitionEngine:
    def __init__(self, transcriber: Transcriber, microphone: Microphone | None = None) -> None:
        self._transcriber = transcriber
        self._microphone = microphone

    def cancel(self) -> None:
        if self._microphone is not None:
            self._microphone.cancel()

    def stop_recording(self) -> None:
        if self._microphone is not None:
            self._microphone.stop()

    def recognize_file(self, path: str | Path) -> RecognitionResult:
        source = Path(path)
        if not source.is_file():
            return RecognitionResult("", (), f"Audio file not found: {source}")

        try:
            transcript = self._transcriber.transcribe_file(source)
        except Exception as exc:
            return RecognitionResult("", (), f"Could not process audio: {exc}")
        return self._result(transcript)

    def recognize_microphone(
        self,
        on_level: Callable[[float], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
    ) -> RecognitionResult:
        if self._microphone is None:
            return RecognitionResult("", (), "Microphone is not available.")

        try:
            audio = self._microphone.record(on_level)
        except Exception as exc:
            return RecognitionResult("", (), f"Could not read microphone: {exc}")
        if audio.size == 0:
            return RecognitionResult("", (), "No speech detected.")

        if on_phase is not None:
            on_phase("transcribing")
        try:
            transcript = self._transcriber.transcribe_audio(audio)
        except Exception as exc:
            return RecognitionResult("", (), f"Could not process audio: {exc}")
        return self._result(transcript)

    @staticmethod
    def _result(transcript: Transcript) -> RecognitionResult:
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
