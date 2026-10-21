from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from main.asr.whisper import WhisperTranscriber


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio: object, **kwargs: object) -> tuple[list[object], object]:
        self.calls.append(kwargs)
        text = " zero five " if kwargs.get("initial_prompt") else " 05 "
        return [SimpleNamespace(text=text)], SimpleNamespace(language="en", duration=1.5)


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
    assert "zero one two" in str(model.calls[1]["initial_prompt"])
