from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from main.engine import RecognitionEngine
from tests.fakes import FailingTranscriber, FakeTranscriber


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


class FailingMicrophone:
    def record(self, on_level: Callable[[float], None] | None = None) -> np.ndarray:
        raise RuntimeError("no device")

    def cancel(self) -> None:
        pass

    def stop(self) -> None:
        pass


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


def test_microphone_result_is_parsed() -> None:
    phases: list[str] = []
    engine = RecognitionEngine(FakeTranscriber("two oh five"), FakeMicrophone(np.ones(160)))

    result = engine.recognize_microphone(on_phase=phases.append)

    assert result.error is None
    assert [candidate.digits for candidate in result.candidates] == ["205"]
    assert phases == ["transcribing"]


def test_empty_recording_returns_no_speech() -> None:
    engine = RecognitionEngine(
        FakeTranscriber("two oh five"), FakeMicrophone(np.empty(0, dtype=np.float32))
    )

    result = engine.recognize_microphone()

    assert result.candidates == ()
    assert result.error == "No speech detected."


def test_microphone_failure_is_returned_not_raised() -> None:
    engine = RecognitionEngine(FakeTranscriber("two oh five"), FailingMicrophone())

    result = engine.recognize_microphone()

    assert result.candidates == ()
    assert result.error == "Could not read microphone: no device"


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
