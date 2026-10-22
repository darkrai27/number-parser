from __future__ import annotations

import re
from dataclasses import dataclass

from main.models import Reason

_SEPARATORS = re.compile(r"[\s\-\.\(\)/]")
_COUNTRY_CODE = "0030"
_NATIONAL = re.compile(r"^(?:2\d{9}|69\d{8})$")


@dataclass(frozen=True)
class GreekValidation:
    output: str | None
    reason: Reason

    @property
    def is_valid(self) -> bool:
        return self.reason is Reason.VALID


def validate(raw: str) -> GreekValidation:
    """Clean a candidate and check it against the Greek numbering rules."""
    value = _SEPARATORS.sub("", raw.strip())
    if not value.isdigit():
        return GreekValidation(None, Reason.NOT_DIGITS)

    national = value[len(_COUNTRY_CODE) :] if value.startswith(_COUNTRY_CODE) else value

    if len(national) != 10:
        return GreekValidation(value, Reason.WRONG_LENGTH)

    if not _NATIONAL.fullmatch(national):
        return GreekValidation(value, Reason.WRONG_PREFIX)
    return GreekValidation(value, Reason.VALID)
