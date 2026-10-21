from __future__ import annotations

from pathlib import Path
from typing import Protocol

from main.models import Transcript


class Transcriber(Protocol):
    def transcribe_file(self, path: Path) -> Transcript: ...
