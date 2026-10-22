from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from main.asr.base import TranscriptionError, UnsupportedAudioError
from main.asr.whisper import WhisperTranscriber


class _FakeModel:
    def __init__(
        self,
        primary: str = " 05 ",
        worded: str = " zero five ",
        worded_beam1: str | None = None,
        primary_metrics: dict[str, float] | None = None,
        worded_metrics: dict[str, float] | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.primary = primary
        self.worded = worded
        self.worded_beam1 = worded_beam1
        self.primary_metrics = primary_metrics or {}
        self.worded_metrics = worded_metrics or {}

    def transcribe(self, audio: object, **kwargs: object) -> tuple[list[object], object]:
        self.calls.append(kwargs)
        prompted = bool(kwargs.get("initial_prompt"))
        if prompted:
            text = (
                self.worded_beam1
                if kwargs.get("beam_size") == 1 and self.worded_beam1 is not None
                else self.worded
            )
        else:
            text = self.primary
        metrics = self.worded_metrics if prompted else self.primary_metrics
        segment = SimpleNamespace(
            text=text,
            no_speech_prob=metrics.get("no_speech_prob", 0.1),
            avg_logprob=metrics.get("avg_logprob", -0.2),
            compression_ratio=metrics.get("compression_ratio", 1.0),
        )
        return [segment], SimpleNamespace(language="en", duration=1.5)


def test_transcribe_file_uses_digits_and_word_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(decode_audio=lambda path, sampling_rate: "audio")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    model = _FakeModel()
    transcriber = WhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe_file(Path("call.wav"))

    assert result.text == "05"
    assert result.alternatives == ("zero five",)
    assert len(model.calls) == 2
    assert model.calls[0]["initial_prompt"] is None
    assert model.calls[0]["without_timestamps"] is True
    assert model.calls[1]["without_timestamps"] is False
    assert "zero one two" in str(model.calls[1]["initial_prompt"])


def test_non_number_primary_skips_word_pass() -> None:
    model = _FakeModel(primary=" You ")
    transcriber = WhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe_audio(np.zeros(160, dtype=np.float32))

    assert result.text == "You"
    assert result.alternatives == ()
    assert len(model.calls) == 1


def test_homophone_primary_still_runs_the_word_pass() -> None:
    model = _FakeModel(primary=" oh oh thirty ", worded=" zero zero thirty ")
    transcriber = WhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe_audio(np.zeros(160, dtype=np.float32))

    assert result.text == "oh oh thirty"
    assert result.alternatives == ("zero zero thirty",)
    assert len(model.calls) == 2


def test_digit_only_word_pass_retries_with_greedy_decoding() -> None:
    model = _FakeModel(
        primary=" 2 10 69 36 64 ",
        worded=" 2 10 69 36 64 ",
        worded_beam1=" two ten sixty nine thirty six six four ",
    )
    transcriber = WhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe_audio(np.zeros(160, dtype=np.float32))

    assert result.text == "2 10 69 36 64"
    assert result.alternatives == ("two ten sixty nine thirty six six four",)
    assert len(model.calls) == 3
    assert model.calls[1]["beam_size"] == 5
    assert model.calls[2]["beam_size"] == 1


def test_low_confidence_segment_is_dropped() -> None:
    model = _FakeModel(primary=" twenty twenty ", primary_metrics={"compression_ratio": 55.0})
    transcriber = WhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe_audio(np.zeros(160, dtype=np.float32))

    assert result.text == ""
    assert result.alternatives == ()
    assert len(model.calls) == 1


def test_hallucinated_word_pass_is_dropped() -> None:
    model = _FakeModel(
        primary=" 05 ",
        worded=" twenty twenty ",
        worded_metrics={"compression_ratio": 55.0},
    )
    transcriber = WhisperTranscriber()
    transcriber._model = model

    result = transcriber.transcribe_audio(np.zeros(160, dtype=np.float32))

    assert result.text == "05"
    assert result.alternatives == ()
    assert len(model.calls) == 2


def test_decode_failure_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad_decode(path: str, sampling_rate: int) -> object:
        raise RuntimeError("bad codec")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(decode_audio=bad_decode))

    with pytest.raises(UnsupportedAudioError, match="bad codec"):
        WhisperTranscriber().transcribe_file(Path("bad.wav"))


def test_model_failure_is_typed() -> None:
    class _FailingModel:
        def transcribe(self, audio: object, **kwargs: object) -> object:
            raise RuntimeError("model failed")

    transcriber = WhisperTranscriber()
    transcriber._model = _FailingModel()

    with pytest.raises(TranscriptionError, match="model failed"):
        transcriber.transcribe_audio(np.zeros(160, dtype=np.float32))
