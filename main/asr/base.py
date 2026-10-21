from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from main.models import Transcript


class UnsupportedAudioError(Exception):
    """The audio file could not be decoded."""


class TranscriptionError(Exception):
    """The ASR model failed to transcribe audio."""


class Transcriber(Protocol):
    def transcribe_file(self, path: Path) -> Transcript: ...

    def transcribe_audio(self, audio: np.ndarray) -> Transcript: ...
