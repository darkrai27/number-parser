from __future__ import annotations

import subprocess
import sys

import pytest

from main import __main__ as cli
from main.asr.whisper import WhisperTranscriber
from main.tui.app import RecognitionApp


def test_module_help_lists_cli_options() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--file" in result.stdout
    assert "--model" in result.stdout


def test_file_mode_loads_model_before_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(WhisperTranscriber, "load", lambda self: calls.append("load"))
    monkeypatch.setattr(RecognitionApp, "run", lambda self: calls.append("run"))

    assert cli.main(["--file", "call.wav"]) == 0
    assert calls == ["load", "run"]


def test_model_load_failure_exits_before_tui(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    def fail(self: WhisperTranscriber) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(WhisperTranscriber, "load", fail)
    monkeypatch.setattr(RecognitionApp, "run", lambda self: calls.append("run"))

    assert cli.main(["--file", "call.wav"]) == 1
    assert calls == []
    assert "Could not load speech model: boom" in capsys.readouterr().err
