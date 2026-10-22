from __future__ import annotations

from pathlib import Path

import pytest

from main.asr.whisper import WhisperTranscriber
from main.engine import RecognitionEngine
from main.models import Candidate, ErrorCode
from main.numbers.greek import validate

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "audios" / "examples"

# Handcrafted input/output pairs. Each value is the expected Candidate.input set
# for the audio fixture: digits grouped as heard, not the concatenated digits.
DOCUMENTED_CASES: dict[str, frozenset[str]] = {
    "136.mp3": frozenset(
        {
            "136",
            "100 36",
            "100 30 6",
            "130 6",
        }
    ),
    "1500-524.mp3": frozenset(
        {
            "15 100 524",
            "15 100 520 4",
            "15 100 500 24",
            "15 100 500 20 4",
        }
    ),
    "30-2-5-58.mp3": frozenset(
        {
            "30 2 5 58",
            "30 2 5 50 8",
            "32 5 58",
            "32 5 50 8",
        }
    ),
    "2-10-69-30-6-60-4.mp3": frozenset(
        {
            "210 69 36 64",
            "2 10 69 36 60 4",
            "2 10 69 30 6 64",
            "2 10 69 30 6 60 4",
            "2 10 60 9 36 64",
            "2 10 60 9 36 60 4",
            "2 10 60 9 30 6 64",
            "2 10 60 9 30 6 60 4",
        }
    ),
    "2-10-69-30-6-60-4(v2).mp3": frozenset(
        {
            "210 69 36 64",
            "2 10 69 36 60 4",
            "2 10 69 30 6 64",
            "2 10 69 30 6 60 4",
            "2 10 60 9 36 64",
            "2 10 60 9 36 60 4",
            "2 10 60 9 30 6 64",
            "2 10 60 9 30 6 60 4",
        }
    ),
    "2-10-69-30-6-6-4.mp3": frozenset(
        {
            # Primary digit grouping wins dedupe; the word pass supplies the
            # three split interpretations below.
            "2 10 69 36 64",
            "2 10 69 30 6 6 4",
            "2 10 60 9 36 6 4",
            "2 10 60 9 30 6 6 4",
        }
    ),
    "2-10-69-30-6-6-4(v2).mp3": frozenset(
        {
            "2 10 69 36 6 4",
            "2 10 69 30 6 6 4",
            "2 10 60 9 36 6 4",
            "2 10 60 9 30 6 6 4",
        }
    ),
    "0-0-30-69-74-0-9-22-52.mp3": frozenset(
        {
            "0 0 30 69 74 0 9 22 52",
            "0 0 30 69 74 0 9 22 50 2",
            "0 0 30 69 74 0 9 20 2 52",
            "0 0 30 69 74 0 9 20 2 50 2",
            "0 0 30 69 70 4 0 9 22 52",
            "0 0 30 69 70 4 0 9 22 50 2",
            "0 0 30 69 70 4 0 9 20 2 52",
            "0 0 30 69 70 4 0 9 20 2 50 2",
            "0 0 30 60 9 74 0 9 22 52",
            "0 0 30 60 9 74 0 9 22 50 2",
            "0 0 30 60 9 74 0 9 20 2 52",
            "0 0 30 60 9 74 0 9 20 2 50 2",
            "0 0 30 60 9 70 4 0 9 22 52",
            "0 0 30 60 9 70 4 0 9 22 50 2",
            "0 0 30 60 9 70 4 0 9 20 2 52",
            "0 0 30 60 9 70 4 0 9 20 2 50 2",
        }
    ),
    "0-0-30-69-700-24-1-3-50-2.mp3": frozenset(
        {
            "0 0 30 69 724 1 3 52",
            "0 0 30 69 724 1 3 50 2",
            "0 0 30 69 720 4 1 3 52",
            "0 0 30 69 720 4 1 3 50 2",
            "0 0 30 69 700 24 1 3 52",
            "0 0 30 69 700 24 1 3 50 2",
            "0 0 30 69 700 20 4 1 3 52",
            "0 0 30 69 700 20 4 1 3 50 2",
            "0 0 30 60 9 724 1 3 52",
            "0 0 30 60 9 724 1 3 50 2",
            "0 0 30 60 9 720 4 1 3 52",
            "0 0 30 60 9 720 4 1 3 50 2",
            "0 0 30 60 9 700 24 1 3 52",
            "0 0 30 60 9 700 24 1 3 50 2",
            "0 0 30 60 9 700 20 4 1 3 52",
            "0 0 30 60 9 700 20 4 1 3 50 2",
        }
    ),
    "0-0-30-69-700-24-135-02.mp3": frozenset(
        {
            "0 0 30 69 724 135 0 2",
            "0 0 30 69 724 130 5 0 2",
            "0 0 30 69 724 100 35 0 2",
            "0 0 30 69 724 100 30 5 0 2",
            "0 0 30 69 720 4 135 0 2",
            "0 0 30 69 720 4 130 5 0 2",
            "0 0 30 69 720 4 100 35 0 2",
            "0 0 30 69 720 4 100 30 5 0 2",
            "0 0 30 69 700 24 135 0 2",
            "0 0 30 69 700 24 130 5 0 2",
            "0 0 30 69 700 24 100 35 0 2",
            "0 0 30 69 700 24 100 30 5 0 2",
            "0 0 30 69 700 20 4 135 0 2",
            "0 0 30 69 700 20 4 130 5 0 2",
            "0 0 30 69 700 20 4 100 35 0 2",
            "0 0 30 69 700 20 4 100 30 5 0 2",
            "0 0 30 60 9 724 135 0 2",
            "0 0 30 60 9 724 130 5 0 2",
            "0 0 30 60 9 724 100 35 0 2",
            "0 0 30 60 9 724 100 30 5 0 2",
            "0 0 30 60 9 720 4 135 0 2",
            "0 0 30 60 9 720 4 130 5 0 2",
            "0 0 30 60 9 720 4 100 35 0 2",
            "0 0 30 60 9 720 4 100 30 5 0 2",
            "0 0 30 60 9 700 24 135 0 2",
            "0 0 30 60 9 700 24 130 5 0 2",
            "0 0 30 60 9 700 24 100 35 0 2",
            "0 0 30 60 9 700 24 100 30 5 0 2",
            "0 0 30 60 9 700 20 4 135 0 2",
            "0 0 30 60 9 700 20 4 130 5 0 2",
            "0 0 30 60 9 700 20 4 100 35 0 2",
            "0 0 30 60 9 700 20 4 100 30 5 0 2",
        }
    ),
    # Dedupe by digits keeps the primary transcript's "697 423 745" grouping
    # instead of the word transcript's "6 97 423 7 45".
    "6-97-400-23-7-40-5.mp3": frozenset(
        {
            "697 423 745",
            "6 97 423 7 40 5",
            "6 97 420 3 7 45",
            "6 97 420 3 7 40 5",
            "6 97 400 23 7 45",
            "6 97 400 23 7 40 5",
            "6 97 400 20 3 7 45",
            "6 97 400 20 3 7 40 5",
            "6 90 7 423 7 45",
            "6 90 7 423 7 40 5",
            "6 90 7 420 3 7 45",
            "6 90 7 420 3 7 40 5",
            "6 90 7 400 23 7 45",
            "6 90 7 400 23 7 40 5",
            "6 90 7 400 20 3 7 45",
            "6 90 7 400 20 3 7 40 5",
        }
    ),
}

KNOWN_ASR_ISSUES = {
    "2-10-69-30-6-60-4.mp3": "base.en hears 'sixty' as 'sixteen' and merges groups",
}
GROUP_TOO_LARGE = {
    "1000023.mp3",
    "1025-125000.mp3",
    "33000-20-453.mp3",
}
if not EXAMPLES.is_dir():
    pytest.skip("audio fixtures not present", allow_module_level=True)

FIXTURES = sorted(EXAMPLES.glob("*.mp3"))
if not FIXTURES:
    pytest.skip("audio fixtures not present", allow_module_level=True)

_ENGINE = RecognitionEngine(WhisperTranscriber())


def _assert_candidate_contract(candidates: tuple[Candidate, ...]) -> None:
    digits = [candidate.digits for candidate in candidates]
    assert len(digits) == len(set(digits))
    for candidate in candidates:
        groups = candidate.input.split()
        assert groups and all(group.isdigit() and len(group) <= 3 for group in groups)
        assert "".join(groups) == candidate.digits
        assert candidate.output == candidate.digits
        validation = validate(candidate.digits)
        assert candidate.is_valid is validation.is_valid
        assert candidate.reason is validation.reason


def _params() -> list[pytest.ParameterSet]:
    params = []
    for path in FIXTURES:
        marks = []
        reason = KNOWN_ASR_ISSUES.get(path.name)
        if reason is not None:
            marks.append(pytest.mark.xfail(strict=True, reason=reason))
        params.append(pytest.param(path, marks=marks, id=path.name))
    return params


def test_every_fixture_has_expectations() -> None:
    missing = sorted(
        path.name
        for path in FIXTURES
        if path.name not in DOCUMENTED_CASES and path.name not in GROUP_TOO_LARGE
    )
    assert not missing, f"fixtures without expectations: {missing}"


@pytest.mark.parametrize("path", _params())
def test_example_audio(path: Path) -> None:
    result = _ENGINE.recognize_file(path)

    if path.name in GROUP_TOO_LARGE:
        assert result.error is ErrorCode.GROUP_TOO_LARGE
        assert result.candidates == ()
        return

    assert result.error is None
    actual = {candidate.input for candidate in result.candidates}
    expected = DOCUMENTED_CASES[path.name]
    assert actual == expected, f"{path.name}: expected {sorted(expected)}; got {sorted(actual)}"
    _assert_candidate_contract(result.candidates)
