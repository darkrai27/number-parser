from __future__ import annotations

from pathlib import Path

import pytest

from main.asr.whisper import WhisperTranscriber
from main.engine import RecognitionEngine
from main.models import ErrorCode
from main.numbers.greek import validate

pytestmark = pytest.mark.integration

EXAMPLES = Path(__file__).resolve().parents[2] / "audios" / "examples"
KNOWN_MISRECOGNIZED = {
    "2-10-69-30-6-6-4.mp3",
    "2-10-69-30-6-60-4.mp3",
}
GROUP_TOO_LARGE = {
    "1000023.mp3",
    "1024-125000.mp3",
    "1025-125000.mp3",
    "33000-20-453.mp3",
}

if not EXAMPLES.is_dir():
    pytest.skip("audio fixtures not present", allow_module_level=True)

FIXTURES = sorted(EXAMPLES.glob("*.mp3"))
if not FIXTURES:
    pytest.skip("audio fixtures not present", allow_module_level=True)

_ENGINE = RecognitionEngine(WhisperTranscriber())


def expected_digits(path: Path) -> str:
    overrides = {"1500-524.mp3": "15100524"}
    if path.name in overrides:
        return overrides[path.name]
    return path.stem.replace("(v2)", "").replace("-", "")


def _params() -> list[pytest.ParameterSet]:
    params = []
    for path in FIXTURES:
        marks = []
        if path.name in KNOWN_MISRECOGNIZED:
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    reason="base.en merges '30 6' into '36'",
                )
            )
        params.append(pytest.param(path, marks=marks, id=path.name))
    return params


@pytest.mark.parametrize("path", _params())
def test_example_audio(path: Path) -> None:
    expected = expected_digits(path)
    result = _ENGINE.recognize_file(path)

    if path.name in GROUP_TOO_LARGE:
        assert result.error is ErrorCode.GROUP_TOO_LARGE
        assert result.candidates == ()
        return

    assert result.error is None
    candidate = next((c for c in result.candidates if c.digits == expected), None)
    assert candidate is not None, (
        f"expected {expected}; got {[c.digits for c in result.candidates]}"
    )
    assert candidate.output == expected
    assert candidate.is_valid is validate(expected).is_valid
