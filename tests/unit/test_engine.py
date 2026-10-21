from __future__ import annotations

from pathlib import Path

from main.engine import RecognitionEngine
from tests.fakes import FailingTranscriber, FakeTranscriber


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "call.wav"
    path.write_bytes(b"not real audio")
    return path


def test_recognize_file_returns_candidates(tmp_path: Path) -> None:
    result = RecognitionEngine(FakeTranscriber("two oh five")).recognize_file(_audio(tmp_path))

    assert result.error is None
    assert result.transcript == "two oh five"
    assert [candidate.digits for candidate in result.candidates] == ["205"]


def test_missing_file_returns_error(tmp_path: Path) -> None:
    result = RecognitionEngine(FakeTranscriber("two oh five")).recognize_file(tmp_path / "nope.wav")

    assert result.candidates == ()
    assert "not found" in (result.error or "")


def test_transcript_without_number_returns_error(tmp_path: Path) -> None:
    result = RecognitionEngine(FakeTranscriber("hello there")).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error == "No telephone number recognized."


def test_transcriber_failure_is_returned_not_raised(tmp_path: Path) -> None:
    result = RecognitionEngine(FailingTranscriber()).recognize_file(_audio(tmp_path))

    assert result.candidates == ()
    assert result.error == "Could not process audio: boom"


def test_alternative_transcript_adds_candidates(tmp_path: Path) -> None:
    transcriber = FakeTranscriber(
        "two oh five",
        alternatives=("zero zero three zero two one zero one two three four five six seven",),
    )

    result = RecognitionEngine(transcriber).recognize_file(_audio(tmp_path))

    assert result.error is None
    assert "205" in {candidate.digits for candidate in result.candidates}
    assert "00302101234567" in {candidate.digits for candidate in result.candidates}
