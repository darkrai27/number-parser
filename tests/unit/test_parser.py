from __future__ import annotations

import pytest

from main.models import Reason
from main.numbers.parser import GroupTooLargeError, has_number, parse_candidates


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
        ("one hundred twenty three", ["123", "10023", "100203", "1203"]),
        ("one hundred and twenty three", ["123", "10023", "100203", "1203"]),
        ("twelve hundred", ["12100"]),
        ("twenty five hundred", ["20500"]),
        ("twenty hundred", ["20100"]),
        ("fifteen hundred", ["15100"]),
        ("one five hundred", ["1500"]),
        ("too five", ["25"]),
        ("won two", ["12"]),
        ("to", []),
        ("zero zero three zero two one zero one two three four five six seven", ["00302101234567"]),
        ("two five and two five", ["25"]),
    ],
)
def test_parse_candidates(text: str, expected: list[str]) -> None:
    assert digits(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("20 5", {"20 5", "25"}),
        ("75", {"75", "70 5"}),
        (
            "2 10 6 9 30 6 6 4",
            {
                "2 10 6 9 30 6 6 4",
                "2 10 6 9 36 6 4",
            },
        ),
        (
            "2 10 69 30 6 6 4",
            {
                "2 10 69 30 6 6 4",
                "2 10 69 36 6 4",
                "2 10 60 9 30 6 6 4",
                "2 10 60 9 36 6 4",
            },
        ),
        (
            "0 0 30 69 700 24 1 3 50 2",
            {
                "0 0 30 69 700 24 1 3 50 2",
                "0 0 30 69 700 24 1 3 52",
                "0 0 30 69 700 20 4 1 3 50 2",
                "0 0 30 69 700 20 4 1 3 52",
                "0 0 30 69 724 1 3 50 2",
                "0 0 30 69 724 1 3 52",
                "0 0 30 69 720 4 1 3 50 2",
                "0 0 30 69 720 4 1 3 52",
                "0 0 30 60 9 700 24 1 3 50 2",
                "0 0 30 60 9 700 24 1 3 52",
                "0 0 30 60 9 700 20 4 1 3 50 2",
                "0 0 30 60 9 700 20 4 1 3 52",
                "0 0 30 60 9 724 1 3 50 2",
                "0 0 30 60 9 724 1 3 52",
                "0 0 30 60 9 720 4 1 3 50 2",
                "0 0 30 60 9 720 4 1 3 52",
            },
        ),
    ],
)
def test_digit_sequence_ambiguity(text: str, expected: set[str]) -> None:
    assert {candidate.input for candidate in parse_candidates(text)} == expected


def test_digit_ambiguity_can_be_disabled() -> None:
    candidates = parse_candidates("2 10 69 30 6 6 4", digit_ambiguity=False)

    assert [candidate.input for candidate in candidates] == ["2 10 69 30 6 6 4"]


def test_two_ten_keeps_only_two_groups() -> None:
    candidates = parse_candidates("2 10")

    assert [candidate.input for candidate in candidates] == ["2 10"]
    assert all("200" not in candidate.input for candidate in candidates)


def test_twenty_five_keeps_both_interpretations() -> None:
    by_digits = {candidate.digits: candidate for candidate in parse_candidates("twenty five")}

    assert by_digits["25"].interpretation == "compound"
    assert by_digits["25"].input == "25"
    assert by_digits["205"].interpretation == "split"
    assert by_digits["205"].input == "20 5"
    assert all(candidate.reason is Reason.WRONG_LENGTH for candidate in by_digits.values())


def test_hundred_binding_respects_three_digit_groups() -> None:
    by_digits = {
        candidate.digits: candidate for candidate in parse_candidates("twenty five hundred")
    }

    assert by_digits["20500"].input == "20 500"
    assert by_digits["20500"].interpretation == "split"
    assert "2500" not in by_digits
    assert "25100" not in by_digits

    (twelve,) = parse_candidates("twelve hundred")
    assert twelve.digits == "12100"
    assert twelve.input == "12 100"


def test_hundred_absorbs_tens_but_not_unit() -> None:
    by_digits = {
        candidate.digits: candidate for candidate in parse_candidates("one hundred thirty six")
    }

    assert set(by_digits) == {"136", "10036", "100306", "1306"}
    assert by_digits["1306"].input == "130 6"
    assert by_digits["10036"].input == "100 36"
    assert by_digits["100306"].input == "100 30 6"


def test_hundred_prefix_never_emits_a_four_digit_group() -> None:
    candidates = parse_candidates("seven hundred twenty four hundred")

    assert candidates
    for candidate in candidates:
        groups = candidate.input.split()
        assert all(len(group) <= 3 for group in groups), candidate.input


@pytest.mark.parametrize(
    "text",
    [
        "thousand",
        "one thousand",
        "two thousand five",
        "one hundred thousand",
        "sixty nine thousand",
        "one million",
        "two billion",
        "five trillion",
        "millions",
        "1,500",
        "1,500, 524",
        "2200",
        "1500",
        "1000",
        "0030",
    ],
)
def test_large_units_are_rejected(text: str) -> None:
    with pytest.raises(GroupTooLargeError):
        parse_candidates(text)


def test_twenty_two_hundred_keeps_only_split_groups() -> None:
    (candidate,) = parse_candidates("twenty two hundred")

    assert candidate.digits == "20200"
    assert candidate.input == "20 200"


def test_scaled_phrase_keeps_all_groupings() -> None:
    assert set(digits("seven hundred twenty four")) == {"724", "70024", "700204", "7204"}


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


def test_branch_cap_bounds_candidate_count() -> None:
    # Every "twenty five" pair doubles the ambiguity; the cap stops runaway growth.
    assert len(parse_candidates(" ".join(["twenty five"] * 9))) == 256


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("oh oh thirty", True),
        ("for five", True),
        ("double five", True),
        ("one thousand", True),
        ("hello there", False),
        ("", False),
    ],
)
def test_has_number_matches_what_the_parser_reads(text: str, expected: bool) -> None:
    assert has_number(text) is expected
