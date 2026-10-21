from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from textual.widgets import DataTable, Static

from main.engine import RecognitionEngine
from main.tui.app import RecognitionApp
from tests.fakes import FakeTranscriber


class _Mic:
    def __init__(self) -> None:
        self.cancelled = False

    def record(self, on_level: Callable[[float], None] | None = None) -> np.ndarray:
        if on_level is not None:
            on_level(0.5)
        return np.ones(160, dtype=np.float32)

    def cancel(self) -> None:
        self.cancelled = True

    def stop(self) -> None:
        pass


class _BlockingMic:
    def __init__(self) -> None:
        self.stopped = threading.Event()
        self.cancelled = False

    def record(self, on_level: Callable[[float], None] | None = None) -> np.ndarray:
        self.stopped.wait(5.0)
        return np.ones(160, dtype=np.float32)

    def cancel(self) -> None:
        self.cancelled = True

    def stop(self) -> None:
        self.stopped.set()


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "call.wav"
    path.write_bytes(b"not real audio")
    return path


async def _paused(pilot) -> None:  # type: ignore[no-untyped-def]
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


async def test_file_result_is_shown(tmp_path: Path) -> None:
    app = RecognitionApp(RecognitionEngine(FakeTranscriber("two oh five")), str(_audio(tmp_path)))

    async with app.run_test(size=(100, 30)) as pilot:
        await _paused(pilot)

        table = app.query_one(DataTable)
        assert table.display
        assert table.row_count == 1
        row = table.get_row_at(0)
        assert str(row[1]) == "2 0 5"
        assert str(row[2]) == "205"
        assert str(row[3]) == "invalid"
        assert str(row[4]) == "wrong length"
        assert "Transcript:" in str(app.query_one("#status", Static).content)


async def test_valid_rows_come_first(tmp_path: Path) -> None:
    transcript = (
        "twenty five then zero zero three zero two one zero one two three four five six seven"
    )
    app = RecognitionApp(RecognitionEngine(FakeTranscriber(transcript)), str(_audio(tmp_path)))

    async with app.run_test(size=(100, 30)) as pilot:
        await _paused(pilot)

        table = app.query_one(DataTable)
        assert table.row_count == 3
        assert str(table.get_row_at(0)[3]) == "VALID"
        assert str(table.get_row_at(0)[4]) == ""
        assert str(table.get_row_at(1)[2]) == "25"


async def test_retry_runs_again(tmp_path: Path) -> None:
    transcriber = FakeTranscriber("two oh five", "six seven")
    app = RecognitionApp(RecognitionEngine(transcriber), str(_audio(tmp_path)))

    async with app.run_test(size=(100, 30)) as pilot:
        await _paused(pilot)
        table = app.query_one(DataTable)
        assert str(table.get_row_at(0)[1]) == "2 0 5"

        await pilot.press("r")
        await _paused(pilot)

        assert str(table.get_row_at(0)[1]) == "6 7"


async def test_no_number_hides_table(tmp_path: Path) -> None:
    app = RecognitionApp(RecognitionEngine(FakeTranscriber("hello there")), str(_audio(tmp_path)))

    async with app.run_test(size=(100, 30)) as pilot:
        await _paused(pilot)

        assert not app.query_one(DataTable).display
        assert "No telephone number recognized." in str(app.query_one("#status", Static).content)


async def test_space_starts_microphone_flow() -> None:
    app = RecognitionApp(RecognitionEngine(FakeTranscriber("two oh five"), _Mic()))

    async with app.run_test(size=(100, 30)) as pilot:
        assert "Press SPACE" in str(app.query_one("#status", Static).content)

        await pilot.press("space")
        await _paused(pilot)

        table = app.query_one(DataTable)
        assert table.display
        assert str(table.get_row_at(0)[1]) == "2 0 5"
        assert "Transcript:" in str(app.query_one("#status", Static).content)


async def test_quit_cancels_microphone() -> None:
    microphone = _Mic()
    app = RecognitionApp(RecognitionEngine(FakeTranscriber("two oh five"), microphone))

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("q")

    assert microphone.cancelled


async def test_space_stops_recording() -> None:
    microphone = _BlockingMic()
    app = RecognitionApp(RecognitionEngine(FakeTranscriber("two oh five"), microphone))

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("space")
        for _ in range(20):
            if app._listening:
                break
            await pilot.pause(0.01)
        assert app._listening

        await pilot.press("space")
        await _paused(pilot)

        assert microphone.stopped.is_set()
        assert str(app.query_one(DataTable).get_row_at(0)[1]) == "2 0 5"
