from __future__ import annotations

from pathlib import Path

from main.models import Transcript


class FakeTranscriber:
    def __init__(self, *texts: str, alternatives: tuple[str, ...] = ()) -> None:
        self._texts = list(texts) or [""]
        self._alternatives = alternatives
        self.calls = 0

    def transcribe_file(self, path: Path) -> Transcript:
        self.calls += 1
        text = self._texts[0]
        if len(self._texts) > 1:
            self._texts.pop(0)
        return Transcript(text=text, alternatives=self._alternatives)


class FailingTranscriber:
    def transcribe_file(self, path: Path) -> Transcript:
        raise RuntimeError("boom")
