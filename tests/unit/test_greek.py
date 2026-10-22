from __future__ import annotations

import pytest

from main.models import Reason
from main.numbers.greek import validate


@pytest.mark.parametrize(
    ("raw", "output", "reason"),
    [
        ("00302101234567", "00302101234567", Reason.VALID),
        ("+30 210 123 4567", None, Reason.NOT_DIGITS),
        ("30-210-123-4567", "302101234567", Reason.WRONG_LENGTH),
        ("0030 691 234 5678", "00306912345678", Reason.VALID),
        ("2101234567", "2101234567", Reason.VALID),
        ("6912345678", "6912345678", Reason.VALID),
        ("00301234567890", "00301234567890", Reason.WRONG_PREFIX),
        ("1234567890", "1234567890", Reason.WRONG_PREFIX),
        ("0030210123456", "0030210123456", Reason.WRONG_LENGTH),
        ("003069123456789", "003069123456789", Reason.WRONG_LENGTH),
        ("0030", "0030", Reason.WRONG_LENGTH),
        ("0030abc", None, Reason.NOT_DIGITS),
        ("", None, Reason.NOT_DIGITS),
    ],
)
def test_validate(raw: str, output: str | None, reason: Reason) -> None:
    result = validate(raw)

    assert result.output == output
    assert result.reason == reason
    assert result.is_valid is (reason is Reason.VALID)
