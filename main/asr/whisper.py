from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from main.asr.base import TranscriptionError, UnsupportedAudioError
from main.models import Transcript
from main.numbers.parser import has_number

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import TranscriptionInfo

# Keep spoken numbers as words so the parser can preserve ambiguous groupings.
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
    "sixty seventy eighty ninety hundred thousand million billion trillion"
)
_MAX_NO_SPEECH = 0.6
_MIN_LOGPROB = -1.0
_MAX_COMPRESSION = 2.4

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_TEENS = (
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_WORD_GROUPS = re.compile(r"\d[\d,]*")


def _group_words(value: int) -> str:
    if value < 10:
        return _ONES[value]
    if value < 20:
        return _TEENS[value - 10]
    if value < 100:
        tens, unit = divmod(value, 10)
        return _TENS[tens] + (f" {_ONES[unit]}" if unit else "")
    hundreds, remainder = divmod(value, 100)
    words = f"{_ONES[hundreds]} hundred"
    return f"{words} {_group_words(remainder)}" if remainder else words


def _word_hint(text: str) -> str | None:
    """Turn digit groups from the digit pass into a number-word decoding hint."""
    groups: list[str] = []
    for match in _WORD_GROUPS.findall(text):
        raw = match.rstrip(",").replace(",", "")
        if not raw.isdigit() or len(raw) > 3 or (raw != "0" and raw.startswith("0")):
            continue
        groups.append(_group_words(int(raw)))
    return " ".join(groups) or None


class WhisperTranscriber:
    """faster-whisper adapter. The model is loaded on first use, not at import."""

    def __init__(self, model_name: str = "base.en") -> None:
        self._model_name = model_name
        self._model: WhisperModel | None = None

    def _model_instance(self) -> WhisperModel:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
        return self._model

    def load(self) -> None:
        """Load the model before the TUI starts.

        HuggingFace's download path creates a tqdm multiprocessing lock. Doing that
        inside a Textual worker hits closed file descriptors, so force it early.
        """
        self._model_instance()

    def transcribe_file(self, path: Path) -> Transcript:
        from faster_whisper import decode_audio

        try:
            audio = decode_audio(str(path), sampling_rate=16000)
        except Exception as exc:
            raise UnsupportedAudioError(str(exc)) from exc
        return self.transcribe_audio(audio)

    def transcribe_audio(self, audio: np.ndarray) -> Transcript:
        try:
            model = self._model_instance()
            # Segment timestamps are unused and can inject numeric artifacts into
            # the digit pass, such as an extra leading zero.
            primary, info = self._transcribe(model, audio, None, without_timestamps=True)
            worded = ""
            if has_number(primary):
                hotwords = _word_hint(primary)
                worded, _ = self._transcribe(model, audio, _NUMBER_WORDS, hotwords=hotwords)
                if worded and not any(character.isalpha() for character in worded):
                    # The prompted pass sometimes normalizes back to digits.
                    # Greedy decoding keeps the spoken number words instead.
                    worded, _ = self._transcribe(
                        model, audio, _NUMBER_WORDS, beam_size=1, hotwords=hotwords
                    )
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(str(exc)) from exc

        texts = [text for text in (primary, worded) if text]
        alternatives = (worded,) if worded and worded != primary else ()
        return Transcript(
            text=texts[0] if texts else "",
            language=info.language or "en",
            duration_s=info.duration or 0.0,
            alternatives=alternatives,
        )

    @staticmethod
    def _transcribe(
        model: WhisperModel,
        audio: np.ndarray,
        initial_prompt: str | None,
        *,
        beam_size: int = 5,
        without_timestamps: bool = False,
        hotwords: str | None = None,
    ) -> tuple[str, TranscriptionInfo]:
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=beam_size,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            without_timestamps=without_timestamps,
            hotwords=hotwords,
        )
        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.no_speech_prob <= _MAX_NO_SPEECH
            and segment.avg_logprob >= _MIN_LOGPROB
            and segment.compression_ratio <= _MAX_COMPRESSION
        ).strip()
        return text, info
