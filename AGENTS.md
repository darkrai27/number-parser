# AGENTS.md — Omilia Challenge

English-only TUI that recognizes spoken telephone numbers, lists every plausible digit
variant, and flags valid Greek numbers. Local/offline after the ASR model is cached.

Read `docs/constitution.md` plus the spec for the area before changing behavior:
`SPEC-001.md` (TUI/orchestration), `SPEC-002.md` (audio → ASR → candidates → validation).

## Status

Foundation stage: specs/config exist; `main/` is a runnable placeholder. Recognition is
not implemented yet. Replace the placeholder as the first implementation step.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- Python 3.11.x is required (`>=3.11,<3.12`).
- First real transcription downloads the model (~150 MB) to `~/.cache/huggingface`.
- Audio decoding uses PyAV, installed with faster-whisper; no external `ffmpeg` binary.
- macOS asks for microphone permission on the first live run.

## Commands

```bash
python -m main                     # live microphone + TUI
python -m main --file sample.wav   # deterministic path, no microphone
pytest                            # unit tests; no mic/model/network
pytest tests/test_cli.py::test_module_help_lists_cli_options   # single test
pytest -m integration             # opt-in: real audio/model
ruff check .                      # lint
ruff format .                     # format
ruff check . && ruff format --check . && pytest      # full verification
```

Run verification in that order before handing work back. Do not claim a UI change works
without running the `--file` path or the live app.

## Target layout

```text
main/__main__.py   CLI (--file, --model); no domain logic
main/models.py     shared frozen dataclasses (Transcript, Candidate, RecognitionResult)
main/engine.py     orchestrates capture -> transcribe -> parse -> validate
main/tui/          Textual screens; fake-engine pilot tests
main/audio/        capture + VAD; mono 16 kHz float32
main/asr/          Transcriber protocol + faster-whisper adapter (lazy import)
main/numbers/      parser + Greek validator; pure and synchronous
tests/unit/       parser/validator/pilot tests, no hardware
tests/integration/ real audio/model tests, marked `integration`
```

Dependency direction: `tui → engine → audio/asr`; `numbers` imports only stdlib and
`main.models`. Keep it that way—it is what makes the core testable without a microphone.

## Non-obvious rules

- Valid Greek number = `0030` + 10 digits starting with `2` or `69`. Normalize `0030`,
  `+30`, `30`; a bare 10-digit number is shown but invalid (`missing_country_code`).
- "twenty five" must yield both `25` (compound) and `205` (split `20` + `5`). Never
  collapse ambiguity; never drop invalid candidates.
- Parser/validator are deterministic and pure: no I/O, clock, randomness, locale, or
  Textual imports.
- Unit tests must not touch a mic, model, network, or real audio. Fake the boundaries;
  mark real-audio/model tests `@pytest.mark.integration`.
- Run audio/model/transcription off the Textual event loop (`asyncio.to_thread` or a
  worker); no blocking handler.
- English only in UI strings, errors, identifiers, specs, and docs.
- Logs go to stderr—stdout belongs to Textual. No `print()` in TUI code.
- No cloud STT, no API keys, no persisted audio, no telephony features without asking.
- Errors are typed and actionable; `Ctrl+C` must release the microphone cleanly.

## Definition of done

Spec updated for behavior/UX changes, tests at the lowest useful layer, full verification
above passes, and the `--file` demo (plus live mic for UI work) has been exercised.
