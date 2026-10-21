from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from main.models import Transcript

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

# Keep spoken numbers as words so the parser can preserve ambiguous groupings.
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
    "sixty seventy eighty ninety hundred thousand"
)


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

        audio = decode_audio(str(path), sampling_rate=16000)
        model = self._model_instance()
        primary, info = self._transcribe(model, audio, None)
        worded, _ = self._transcribe(model, audio, _NUMBER_WORDS)

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
        model: WhisperModel, audio: object, initial_prompt: str | None
    ) -> tuple[str, object]:
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )
        return " ".join(segment.text.strip() for segment in segments).strip(), info
