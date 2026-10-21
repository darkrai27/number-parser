from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from main.asr.base import Transcriber, TranscriptionError, UnsupportedAudioError
from main.audio.microphone import AudioReadError, CaptureCancelled, Microphone, NoInputDeviceError
from main.models import ErrorCode, RecognitionResult, Transcript
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
            return self._failure(ErrorCode.FILE_NOT_FOUND, str(source))

        try:
            transcript = self._transcriber.transcribe_file(source)
        except UnsupportedAudioError as exc:
            return self._failure(ErrorCode.UNSUPPORTED_FILE, str(exc))
        except TranscriptionError as exc:
            return self._failure(ErrorCode.TRANSCRIPTION_ERROR, str(exc))
        except Exception as exc:
            return self._failure(ErrorCode.TRANSCRIPTION_ERROR, str(exc))
        return self._result(transcript)

    def recognize_microphone(
        self,
        on_level: Callable[[float], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
    ) -> RecognitionResult:
        if self._microphone is None:
            return self._failure(ErrorCode.NO_INPUT_DEVICE)

        try:
            audio = self._microphone.record(on_level)
        except CaptureCancelled:
            return self._failure(ErrorCode.CANCELLED)
        except NoInputDeviceError as exc:
            return self._failure(ErrorCode.NO_INPUT_DEVICE, str(exc))
        except AudioReadError as exc:
            return self._failure(ErrorCode.AUDIO_READ_ERROR, str(exc))
        except Exception as exc:
            return self._failure(ErrorCode.AUDIO_READ_ERROR, str(exc))
        if audio.size == 0:
            return self._failure(ErrorCode.NO_SPEECH)

        if on_phase is not None:
            on_phase("transcribing")
        try:
            transcript = self._transcriber.transcribe_audio(audio)
        except TranscriptionError as exc:
            return self._failure(ErrorCode.TRANSCRIPTION_ERROR, str(exc))
        except Exception as exc:
            return self._failure(ErrorCode.TRANSCRIPTION_ERROR, str(exc))
        return self._result(transcript)

    @staticmethod
    def _failure(code: ErrorCode, detail: str = "", transcript: str = "") -> RecognitionResult:
        return RecognitionResult(transcript, (), code, detail)

    @staticmethod
    def _result(transcript: Transcript) -> RecognitionResult:
        if not transcript.text and not transcript.alternatives:
            return RecognitionEngine._failure(ErrorCode.NO_SPEECH)
        candidates = []
        seen: set[str] = set()
        for text in (transcript.text, *transcript.alternatives):
            for candidate in parse_candidates(text):
                if candidate.digits not in seen:
                    seen.add(candidate.digits)
                    candidates.append(candidate)
        if not candidates:
            return RecognitionEngine._failure(ErrorCode.NO_NUMBER, transcript=transcript.text)
        return RecognitionResult(transcript.text, tuple(candidates))
