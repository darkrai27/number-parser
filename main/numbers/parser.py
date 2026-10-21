"""Turn a noisy English transcript into plausible telephone-number candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass

from main.models import Candidate
from main.numbers.greek import validate

_UNITS = {
    "zero": 0,
    "oh": 0,
    "nought": 0,
    "nil": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_HOMOPHONES = {
    "to": 2,
    "too": 2,
    "for": 4,
    "won": 1,
    "ate": 8,
    "free": 3,
    "o": 0,
}
_SCALES = {"hundred": 100}
_LARGE_UNITS = frozenset(
    {
        "thousand",
        "thousands",
        "million",
        "millions",
        "billion",
        "billions",
        "trillion",
        "trillions",
        "quadrillion",
        "quadrillions",
    }
)
_WORDS = re.compile(r"[a-z0-9]+")
_MAX_STATES = 256  # ponytail: hard cap on ambiguity branches; revisit if real input overflows


class GroupTooLargeError(ValueError):
    """A spoken group would require more than three digits."""


@dataclass(frozen=True)
class _Token:
    text: str
    kind: str  # unit | teen | tens | hundred | large | digits
    value: int
    homophone: bool = False
    weak: bool = False


def _token(word: str) -> _Token | None:
    if word.isdigit():
        return _Token(word, "digits", int(word))
    if word in _LARGE_UNITS:
        return _Token(word, "large", 0)
    if word in _UNITS:
        return _Token(word, "unit", _UNITS[word])
    if word in _TEENS:
        return _Token(word, "teen", _TEENS[word])
    if word in _TENS:
        return _Token(word, "tens", _TENS[word])
    if word in _SCALES:
        return _Token(word, word, _SCALES[word])
    if word in _HOMOPHONES:
        return _Token(word, "unit", _HOMOPHONES[word], homophone=True, weak=True)
    return None


def _tokenize(text: str) -> list[_Token | None]:
    words = _WORDS.findall(text.lower())
    tokens: list[_Token | None] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word == "point":
            break
        if word == "and" and tokens and tokens[-1] is not None and tokens[-1].kind in _SCALES:
            index += 1
            continue
        if word in {"double", "triple"}:
            following = _token(words[index + 1]) if index + 1 < len(words) else None
            if following is not None and following.kind == "unit":
                phrase = f"{word} {words[index + 1]}"
                repeated = _Token(
                    phrase,
                    "unit",
                    following.value,
                    homophone=following.homophone,
                    weak=following.weak,
                )
                tokens.append(repeated)
                tokens.extend(
                    _Token("", "unit", following.value, following.homophone, following.weak)
                    for _ in range((2 if word == "double" else 3) - 1)
                )
                index += 2
                continue
            index += 1
            continue
        token = _token(word)
        tokens.append(token)
        index += 1
    return tokens


@dataclass(frozen=True)
class _Variant:
    digits: str
    text: str
    flags: frozenset[str]


def _dedupe(states: list[_Variant]) -> list[_Variant]:
    seen: set[str] = set()
    unique: list[_Variant] = []
    for variant in states:
        if variant.digits not in seen:
            seen.add(variant.digits)
            unique.append(variant)
    return unique[:_MAX_STATES]


@dataclass(frozen=True)
class _Parsed:
    value: int
    variants: tuple[_Variant, ...]
    next_index: int


def _small(tokens: list[_Token], index: int) -> _Parsed:
    token = tokens[index]
    if token.kind == "tens" and index + 1 < len(tokens) and tokens[index + 1].kind == "unit":
        unit = tokens[index + 1]
        flags: set[str] = {"compound"}
        if token.homophone or unit.homophone:
            flags.add("homophone")
        total = token.value + unit.value
        return _Parsed(
            total,
            (
                _Variant(str(total), str(total), frozenset(flags)),
                _Variant(
                    f"{token.value}{unit.value}",
                    f"{token.value} {unit.value}",
                    frozenset(flags | {"split"}),
                ),
            ),
            index + 2,
        )

    digits = token.text if token.kind == "digits" else str(token.value)
    flags = frozenset({"homophone"}) if token.homophone else frozenset()
    return _Parsed(
        int(token.text) if token.kind == "digits" else token.value,
        (_Variant(digits, digits, flags),),
        index + 1,
    )


def _hundred(
    lead_text: str,
    lead_digits: str,
    base: int,
    remainder: _Parsed | None,
    flags: frozenset[str],
) -> tuple[_Variant, ...]:
    def joined(*parts: str) -> str:
        return " ".join(part for part in parts if part)

    if remainder is None:
        return (
            _Variant(f"{lead_digits}{base}", joined(lead_text, str(base)), flags | {"compound"}),
        )

    variants = [
        _Variant(
            f"{lead_digits}{base + remainder.value}",
            joined(lead_text, str(base + remainder.value)),
            flags | {"compound"},
        )
    ]
    variants.extend(
        _Variant(
            f"{lead_digits}{base}{part.digits}",
            joined(lead_text, str(base), part.text),
            flags | part.flags | {"split"},
        )
        for part in remainder.variants
    )
    return tuple(variants)


def _scale(tokens: list[_Token], index: int, prefix: _Parsed | None) -> _Parsed:
    token = tokens[index]
    if token.kind != "hundred":
        raise GroupTooLargeError

    start = index + 1
    remainder = None
    if start < len(tokens) and tokens[start].kind in {"unit", "teen", "tens", "digits"}:
        remainder = _small(tokens, start)
    next_index = remainder.next_index if remainder is not None else start

    prefix_flags = frozenset()
    if prefix is not None and any("homophone" in variant.flags for variant in prefix.variants):
        prefix_flags = frozenset({"homophone"})

    if prefix is None:
        variants = _hundred("", "", token.value, remainder, prefix_flags)
        value = token.value + (remainder.value if remainder is not None else 0)
        return _Parsed(value, variants, next_index)

    if prefix.value * token.value <= 999:
        variants = _hundred("", "", prefix.value * token.value, remainder, prefix_flags)
        value = prefix.value * token.value + (remainder.value if remainder is not None else 0)
        return _Parsed(value, variants, next_index)

    split = next((variant for variant in prefix.variants if "split" in variant.flags), None)
    if split is not None:
        chunks = split.text.split()
        unit = int(chunks[-1])
        lead_text = " ".join(chunks[:-1])
        lead_digits = split.digits[: -len(str(unit))]
        variants = _hundred(lead_text, lead_digits, unit * 100, remainder, prefix_flags)
        value = unit * 100 + (remainder.value if remainder is not None else 0)
        return _Parsed(value, variants, next_index)

    variants = tuple(
        variant
        for part in prefix.variants
        for variant in _hundred(part.text, part.digits, token.value, remainder, prefix_flags)
    )
    value = token.value + (remainder.value if remainder is not None else 0)
    return _Parsed(value, variants, next_index)


def _item(tokens: list[_Token], index: int) -> _Parsed:
    token = tokens[index]
    parsed = _scale(tokens, index, None) if token.kind in _SCALES else _small(tokens, index)
    while parsed.next_index < len(tokens) and tokens[parsed.next_index].kind in _SCALES:
        parsed = _scale(tokens, parsed.next_index, parsed)
    return parsed


def _parse_run(tokens: list[_Token]) -> list[_Variant]:
    states: list[_Variant] = [_Variant("", "", frozenset())]
    index = 0
    while index < len(tokens):
        item = _item(tokens, index)
        states = _dedupe(
            [
                _Variant(
                    f"{state.digits}{variant.digits}",
                    f"{state.text} {variant.text}".strip(),
                    state.flags | variant.flags,
                )
                for state in states
                for variant in item.variants
            ]
        )
        index = item.next_index
    return states


def _interpretation(flags: frozenset[str]) -> str:
    for name in ("split", "homophone", "compound"):
        if name in flags:
            return name
    return "direct"


def parse_candidates(text: str) -> tuple[Candidate, ...]:
    """Extract every plausible digit sequence from an English transcript."""
    tokens = _tokenize(text)
    if any(token is not None and token.kind == "large" for token in tokens):
        raise GroupTooLargeError

    runs: list[list[_Token]] = []
    current: list[_Token] = []
    for token in tokens:
        if token is None:
            if current:
                runs.append(current)
                current = []
            continue
        current.append(token)
    if current:
        runs.append(current)

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for run in runs:
        if all(token.weak for token in run):
            continue
        parsed = _parse_run(run)
        for variant in parsed:
            if not variant.digits or variant.digits in seen:
                continue
            seen.add(variant.digits)
            result = validate(variant.digits)
            candidates.append(
                Candidate(
                    digits=variant.digits,
                    input=variant.text,
                    interpretation=_interpretation(variant.flags),
                    output=result.output,
                    is_valid=result.is_valid,
                    reason=result.reason,
                )
            )
    return tuple(candidates)
