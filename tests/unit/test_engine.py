from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from main.asr.base import TranscriptionError, UnsupportedAudioError
from main.audio.microphone import AudioReadError, CaptureCancelled, NoInputDeviceError
from main.engine import RecognitionEngine
from main.models import ErrorCode
from tests.fakes import FailingMicrophone, FailingTranscriber, FakeTranscriber


class FakeMicrophone:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio
        self.cancelled = False
        self.stopped = False

    def record(self, on_level: Callable[[float], None] | None = None) -> np.ndarray:
        if on_level is not None:
            on_level(0.5)
        return self.audio

    def cancel(self) -> None:
        self.cancelled = True

    def stop(self) -> None:
        self.stopped = True


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "call.wav"
    path.write_bytes(b"not real audio")
    return path


def test_recognize_file_returns_candidates(tmp_path: Path) -> None:
    result = RecognitionEngine(FakeTranscriber("two oh five")).recognize_file(_audio(tmp_path))

    assert result.error is None
    assert result.transcript == "two oh five"
    assert [candidate.digits for candidate in result.candidates] == ["205"]


def test_missing_file_returns_file_not_found(tmp_path: Path) -> None:
    result = RecognitionEngine(FakeTranscriber("two oh five")).recognize_file(tmp_path / "nope.wav")

    assert result.candidates == ()
    assert result.error is ErrorCode.FILE_NOT_FOUND
    assert "nope.wav" in result.detail


def test_transcript_without_number_returns_no_number(tmp_path: Path) -> None:
    result = RecognitionEngine(FakeTranscriber("hello there")).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error is ErrorCode.NO_NUMBER
    assert result.transcript == "hello there"


def test_empty_transcript_returns_no_speech(tmp_path: Path) -> None:
    result = RecognitionEngine(FakeTranscriber("")).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error is ErrorCode.NO_SPEECH


def test_group_too_large_is_typed(tmp_path: Path) -> None:
    result = RecognitionEngine(FakeTranscriber("one thousand")).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error is ErrorCode.GROUP_TOO_LARGE


def test_grouped_primary_falls_back_to_word_transcript(tmp_path: Path) -> None:
    transcriber = FakeTranscriber(
        "1,500, 524",
        alternatives=("fifteen hundred five hundred twenty four",),
    )

    result = RecognitionEngine(transcriber).recognize_file(_audio(tmp_path))

    assert result.error is None
    found = {candidate.digits for candidate in result.candidates}
    assert "15100524" in found
    assert "1500524" not in found


def test_word_large_unit_rejects_even_with_digit_primary(tmp_path: Path) -> None:
    transcriber = FakeTranscriber("1000", alternatives=("one thousand",))

    result = RecognitionEngine(transcriber).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error is ErrorCode.GROUP_TOO_LARGE
    assert result.transcript == "one thousand"


def test_group_too_large_in_word_pass_rejects_the_result(tmp_path: Path) -> None:
    transcriber = FakeTranscriber(
        "zero zero three zero two one zero one two three four five six seven",
        alternatives=("two thousand",),
    )

    result = RecognitionEngine(transcriber).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error is ErrorCode.GROUP_TOO_LARGE
    assert result.transcript.startswith("zero zero three zero")


def test_group_too_large_keeps_the_primary_transcript(tmp_path: Path) -> None:
    transcriber = FakeTranscriber("one million", alternatives=("two thousand",))

    result = RecognitionEngine(transcriber).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error is ErrorCode.GROUP_TOO_LARGE
    assert result.transcript == "one million"


def test_parser_failure_stays_typed(tmp_path: Path, monkeypatch) -> None:
    def boom(text: str) -> None:
        raise RuntimeError("parser bug")

    monkeypatch.setattr("main.engine.parse_candidates", boom)

    result = RecognitionEngine(FakeTranscriber("two oh five")).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error is ErrorCode.TRANSCRIPTION_ERROR
    assert result.detail == "parser bug"


def test_unsupported_file_is_typed(tmp_path: Path) -> None:
    transcriber = FailingTranscriber(UnsupportedAudioError("bad codec"))

    result = RecognitionEngine(transcriber).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error is ErrorCode.UNSUPPORTED_FILE
    assert result.detail == "bad codec"


def test_transcriber_failure_is_returned_not_raised(tmp_path: Path) -> None:
    result = RecognitionEngine(FailingTranscriber()).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error is ErrorCode.TRANSCRIPTION_ERROR
    assert result.detail == "boom"


def test_typed_transcription_failure_is_mapped(tmp_path: Path) -> None:
    transcriber = FailingTranscriber(TranscriptionError("model exploded"))

    result = RecognitionEngine(transcriber).recognize_file(_audio(tmp_path))

    assert result.error is ErrorCode.TRANSCRIPTION_ERROR
    assert result.detail == "model exploded"


def test_alternative_transcript_adds_candidates(tmp_path: Path) -> None:
    transcriber = FakeTranscriber(
        "two oh five",
        alternatives=("zero zero three zero two one zero one two three four five six seven",),
    )

    result = RecognitionEngine(transcriber).recognize_file(_audio(tmp_path))

    assert result.error is None
    assert "205" in {candidate.digits for candidate in result.candidates}
    assert "00302101234567" in {candidate.digits for candidate in result.candidates}


def test_microphone_result_is_parsed() -> None:
    phases: list[str] = []
    engine = RecognitionEngine(FakeTranscriber("two oh five"), FakeMicrophone(np.ones(160)))

    result = engine.recognize_microphone(on_phase=phases.append)

    assert result.error is None
    assert [candidate.digits for candidate in result.candidates] == ["205"]
    assert phases == ["transcribing"]


def test_missing_microphone_is_typed() -> None:
    result = RecognitionEngine(FakeTranscriber("two oh five")).recognize_microphone()

    assert result.candidates == ()
    assert result.error is ErrorCode.NO_INPUT_DEVICE


def test_empty_recording_returns_no_speech() -> None:
    engine = RecognitionEngine(
        FakeTranscriber("two oh five"), FakeMicrophone(np.empty(0, dtype=np.float32))
    )

    result = engine.recognize_microphone()

    assert result.candidates == ()
    assert result.error is ErrorCode.NO_SPEECH


def test_cancelled_capture_is_typed() -> None:
    engine = RecognitionEngine(
        FakeTranscriber("two oh five"), FailingMicrophone(CaptureCancelled("cancelled"))
    )

    result = engine.recognize_microphone()

    assert result.candidates == ()
    assert result.error is ErrorCode.CANCELLED


def test_no_input_device_is_typed() -> None:
    engine = RecognitionEngine(
        FakeTranscriber("two oh five"), FailingMicrophone(NoInputDeviceError("no default input"))
    )

    result = engine.recognize_microphone()

    assert result.error is ErrorCode.NO_INPUT_DEVICE
    assert result.detail == "no default input"


def test_audio_read_error_is_typed() -> None:
    engine = RecognitionEngine(
        FakeTranscriber("two oh five"), FailingMicrophone(AudioReadError("stream died"))
    )

    result = engine.recognize_microphone()

    assert result.error is ErrorCode.AUDIO_READ_ERROR
    assert result.detail == "stream died"


def test_microphone_failure_is_returned_not_raised() -> None:
    engine = RecognitionEngine(FakeTranscriber("two oh five"), FailingMicrophone())

    result = engine.recognize_microphone()

    assert result.candidates == ()
    assert result.error is ErrorCode.AUDIO_READ_ERROR
    assert result.detail == "no device"


def test_microphone_transcription_failure_is_typed() -> None:
    engine = RecognitionEngine(
        FailingTranscriber(TranscriptionError("mic model failed")),
        FakeMicrophone(np.ones(160)),
    )

    result = engine.recognize_microphone()

    assert result.error is ErrorCode.TRANSCRIPTION_ERROR
    assert result.detail == "mic model failed"


def test_cancel_reaches_microphone() -> None:
    microphone = FakeMicrophone(np.ones(160))
    engine = RecognitionEngine(FakeTranscriber("two oh five"), microphone)

    engine.cancel()

    assert microphone.cancelled


def test_stop_reaches_microphone() -> None:
    microphone = FakeMicrophone(np.ones(160))
    engine = RecognitionEngine(FakeTranscriber("two oh five"), microphone)

    engine.stop_recording()

    assert microphone.stopped
