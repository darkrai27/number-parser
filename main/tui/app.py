from __future__ import annotations

import asyncio
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Static

from main.engine import RecognitionEngine
from main.models import RecognitionResult

_NO_FILE_MESSAGE = (
    "Live microphone mode is not implemented yet.\nRun: python -m main --file PATH\n\nQ quit"
)


class RecognitionApp(App[None]):
    """Minimal TUI: start, transcribe a file, show candidates, retry or quit."""

    TITLE = "Omilia Challenge"
    CSS = """
    Screen { padding: 1 2; }
    #title { text-style: bold; }
    #status { height: auto; margin: 1 0; }
    DataTable { height: 1fr; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("space", "start", "Start"),
        Binding("r", "retry", "Retry"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, engine: RecognitionEngine, file_path: str | None = None) -> None:
        super().__init__()
        self.engine = engine
        self.file_path = file_path
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Static("Omilia Challenge — telephone number recognition", id="title", markup=False)
        yield Static("", id="status", markup=False)
        yield DataTable(id="results")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("#", "Input", "Output", "Status", "Reason")
        table.display = False
        if self.file_path:
            self._start()
        else:
            self._set_status(_NO_FILE_MESSAGE)

    def action_start(self) -> None:
        self._start()

    def action_retry(self) -> None:
        self._start()

    def _start(self) -> None:
        if self._busy:
            return
        if not self.file_path:
            self._set_status(_NO_FILE_MESSAGE)
            return
        self.run_worker(self._recognize(), exclusive=True)

    async def _recognize(self) -> None:
        self._busy = True
        table = self.query_one(DataTable)
        table.clear()
        table.display = False
        self._set_status("Transcribing…")
        try:
            result = await asyncio.to_thread(self.engine.recognize_file, self.file_path)
        except Exception as exc:
            result = RecognitionResult("", (), f"Recognition failed: {exc}")
        finally:
            self._busy = False
        self._show(result)

    def _show(self, result: RecognitionResult) -> None:
        if result.error:
            self._set_status(f"{result.error}\n\nR retry | Q quit")
            return

        table = self.query_one(DataTable)
        rows = sorted(result.candidates, key=lambda candidate: not candidate.is_valid)
        for index, candidate in enumerate(rows[:50], start=1):
            status = "VALID" if candidate.is_valid else "invalid"
            reason = "" if candidate.is_valid else candidate.reason.value.replace("_", " ")
            table.add_row(
                str(index),
                candidate.input,
                candidate.output or "—",
                status,
                reason,
            )
        table.display = True

        note = f"\nShowing 50 of {len(rows)}" if len(rows) > 50 else ""
        self._set_status(f'Transcript: "{result.transcript}"\nR retry | Q quit{note}')

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)
