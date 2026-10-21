from __future__ import annotations

import asyncio
import contextlib
import time
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Static

from main.engine import Recognizer
from main.models import ErrorCode, RecognitionResult

_START_MESSAGE = "Press SPACE to start speaking.\nQ quit"
_ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.FILE_NOT_FOUND: "Audio file not found",
    ErrorCode.UNSUPPORTED_FILE: "Unsupported audio file",
    ErrorCode.NO_INPUT_DEVICE: "Microphone is not available",
    ErrorCode.AUDIO_READ_ERROR: "Could not read microphone",
    ErrorCode.NO_SPEECH: "No speech detected",
    ErrorCode.CANCELLED: "Capture cancelled",
    ErrorCode.TRANSCRIPTION_ERROR: "Could not process audio",
    ErrorCode.NO_NUMBER: "No telephone number recognized",
}


def format_error(error: ErrorCode, detail: str = "") -> str:
    """Map a typed engine failure to user-facing English copy."""
    message = _ERROR_MESSAGES[error]
    return f"{message}: {detail}" if detail else f"{message}."


class RecognitionApp(App[None]):
    """TUI: start, record or load audio, show candidates, retry or quit."""

    TITLE = "Omilia Challenge"
    CSS = """
    Screen { padding: 1 2; }
    #title { text-style: bold; }
    #status { height: auto; margin: 1 0; }
    DataTable { height: 1fr; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("space", "start", "Start"),
        Binding("c,C", "cancel", "Cancel"),
        Binding("r", "retry", "Retry"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    def __init__(self, engine: Recognizer, file_path: str | None = None) -> None:
        super().__init__()
        self.engine = engine
        self.file_path = file_path
        self._busy = False
        self._listening = False
        self._timer = None
        self._listening_started = 0.0
        self._last_level = 0.0

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
            self._set_status(_START_MESSAGE)

    def action_start(self) -> None:
        if self._listening:
            self.engine.stop_recording()
            self._listening = False
            self._set_status("Processing…")
            return
        self._start()

    def action_retry(self) -> None:
        self._start()

    def action_cancel(self) -> None:
        if not self._listening:
            return
        self._listening = False
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.engine.cancel()
        self._set_status("Cancelling…")

    def action_quit(self) -> None:
        self.engine.cancel()
        self.exit()

    def on_unmount(self) -> None:
        self.engine.cancel()

    def _start(self) -> None:
        if self._busy:
            return
        if self.file_path:
            self.run_worker(self._recognize_file(), exclusive=True)
        else:
            self._listening = True
            self.run_worker(self._recognize_microphone(), exclusive=True)

    async def _prepare(self) -> None:
        self._busy = True
        table = self.query_one(DataTable)
        table.clear()
        table.display = False

    async def _recognize_file(self) -> None:
        await self._prepare()
        self._set_status("Transcribing…")
        try:
            result = await asyncio.to_thread(self.engine.recognize_file, self.file_path)
        except Exception as exc:
            result = RecognitionResult("", (), ErrorCode.TRANSCRIPTION_ERROR, str(exc))
        finally:
            self._busy = False
        self._show(result)

    async def _recognize_microphone(self) -> None:
        await self._prepare()
        self._listening_started = time.monotonic()
        self._last_level = 0.0
        self._set_status("Listening…")
        self._timer = self.set_interval(0.1, self._tick)
        try:
            result = await asyncio.to_thread(
                self.engine.recognize_microphone, self._on_level, self._on_phase
            )
        except Exception as exc:
            result = RecognitionResult("", (), ErrorCode.TRANSCRIPTION_ERROR, str(exc))
        finally:
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            self._listening = False
            self._busy = False
        self._show(result)

    def _on_level(self, level: float) -> None:
        self._last_level = level
        with contextlib.suppress(RuntimeError):
            self.call_from_thread(self._tick)

    def _on_phase(self, phase: str) -> None:
        if phase == "transcribing":
            with contextlib.suppress(RuntimeError):
                self.call_from_thread(self._show_transcribing)

    def _show_transcribing(self) -> None:
        self._listening = False
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._set_status("Transcribing…")

    def _tick(self) -> None:
        if not self._busy:
            return
        elapsed = time.monotonic() - self._listening_started
        filled = min(20, int(self._last_level * 20))
        meter = "█" * filled + "░" * (20 - filled)
        self._set_status(f"Listening… {elapsed:.1f}s\n[{meter}]\nSPACE stop | C cancel | Q quit")

    def _show(self, result: RecognitionResult) -> None:
        if result.error is ErrorCode.CANCELLED:
            if self.is_running:
                self._set_status(_START_MESSAGE)
            return
        if result.error:
            self._set_status(f"{format_error(result.error, result.detail)}\n\nR retry | Q quit")
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
