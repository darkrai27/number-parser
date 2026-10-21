from __future__ import annotations

import pytest

from main.models import Reason
from main.numbers.parser import parse_candidates


def digits(text: str) -> list[str]:
    return [candidate.digits for candidate in parse_candidates(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("twenty five", ["25", "205"]),
        ("two oh five", ["205"]),
        ("double five", ["55"]),
        ("triple seven", ["777"]),
        ("thirteen", ["13"]),
        ("twenty", ["20"]),
        ("my number is two oh five", ["205"]),
        ("two five and three four", ["25", "34"]),
        ("one hundred twenty three", ["123", "10023", "100203"]),
        ("one hundred and twenty three", ["123", "10023", "100203"]),
        ("two thousand five", ["2005"]),
        ("two thousand and five", ["2005"]),
        ("too five", ["25"]),
        ("won two", ["12"]),
        ("to", []),
        ("zero zero three zero two one zero one two three four five six seven", ["00302101234567"]),
        ("two five and two five", ["25"]),
    ],
)
def test_parse_candidates(text: str, expected: list[str]) -> None:
    assert digits(text) == expected


def test_twenty_five_keeps_both_interpretations() -> None:
    by_digits = {candidate.digits: candidate for candidate in parse_candidates("twenty five")}

    assert by_digits["25"].interpretation == "compound"
    assert by_digits["25"].input == "25"
    assert by_digits["205"].interpretation == "split"
    assert by_digits["205"].input == "20 5"
    assert all(candidate.reason is Reason.WRONG_LENGTH for candidate in by_digits.values())


def test_scaled_phrase_keeps_all_groupings() -> None:
    assert set(digits("seven hundred twenty four")) == {"724", "70024", "700204"}


def test_spoken_group_variants_include_each_split() -> None:
    spoken = "zero zero thirty sixty-nine seven hundred twenty-four one three fifty two"

    found = set(digits(spoken))

    assert {
        "0030697241352",
        "00306097241352",
        "0030609700241352",
        "003069700241352",
    } <= found


def test_full_greek_number_is_valid() -> None:
    (candidate,) = parse_candidates(
        "zero zero three zero two one zero one two three four five six seven"
    )

    assert candidate.is_valid
    assert candidate.output == "00302101234567"
    assert candidate.reason is Reason.VALID


def test_bare_national_number_is_valid() -> None:
    (candidate,) = parse_candidates("two one zero one two three four five six seven")

    assert candidate.is_valid
    assert candidate.output == "2101234567"
    assert candidate.reason is Reason.VALID


def test_wrong_prefix_survives_as_invalid() -> None:
    (candidate,) = parse_candidates(
        "zero zero three zero one two three four five six seven eight nine zero"
    )

    assert not candidate.is_valid
    assert candidate.output == "00301234567890"
    assert candidate.reason is Reason.WRONG_PREFIX


def test_input_keeps_repeated_digits() -> None:
    (candidate,) = parse_candidates(
        "zero zero three zero two one zero one two three four five six seven"
    )

    assert candidate.input == "0 0 3 0 2 1 0 1 2 3 4 5 6 7"


def test_input_expands_double_to_digits() -> None:
    (candidate,) = parse_candidates("double five")

    assert candidate.input == "5 5"


def test_input_is_digits_with_spaces() -> None:
    (candidate,) = parse_candidates("two ten")

    assert candidate.digits == "210"
    assert candidate.input == "2 10"
