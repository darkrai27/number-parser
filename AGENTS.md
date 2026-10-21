# AGENTS.md — Omilia Challenge

English-only TUI that recognizes spoken telephone numbers, lists every plausible digit
variant, and flags valid Greek numbers. Local/offline after the ASR model is cached.

Read `docs/constitution.md` plus the spec for the area before changing behavior:
`docs/SPEC-001.md` (TUI/orchestration), `docs/SPEC-002.md` (audio → ASR → candidates →
validation).

## Status

File and live-microphone pipelines are implemented end to end: PyAV decode or
sounddevice/webrtcvad capture → two faster-whisper passes (default digits + number-word
prompt) → parser → Textual results table. Both `python -m main` and
`python -m main --file PATH` work.

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
python -m main                     # live microphone TUI
python -m main --file sample.wav   # deterministic file path
pytest                            # unit tests; no mic/model/network
pytest tests/unit/test_parser.py::test_twenty_five_keeps_both_interpretations  # single test
pytest -m integration             # opt-in: real audio/model; skips if audios/ is absent
ruff check .                      # lint
ruff format .                     # format
ruff check . && ruff format --check . && pytest      # full verification
```

Run verification in that order before handing work back. Do not claim a UI change works
without running the `--file` path or the live app.

## Layout

```text
main/__main__.py   CLI (--file, --model); no domain logic
main/models.py     shared frozen dataclasses (Transcript, Candidate, RecognitionResult)
main/engine.py     file/microphone -> transcribe -> parse -> result
main/tui/          Textual app + results table; fake-engine pilot tests
main/asr/          Transcriber protocol + faster-whisper adapter (lazy import)
main/audio/        sounddevice capture + webrtcvad endpointing
main/numbers/      parser + Greek validator; pure and synchronous
tests/unit/        parser/validator/engine/endpoint/pilot tests, no hardware
tests/integration/ real audio/model tests, marked `integration`
```

Dependency direction: `tui → engine → audio/asr`; `numbers` imports only stdlib and
`main.models`. Keep it that way—it is what makes the core testable without a microphone.

## Non-obvious rules

- Valid Greek number = 10 national digits starting with `2` or `69`, with or without the
  country code. Accept `0030`, `+30`, `30`, and bare numbers, but never add `0030` that
  was not spoken; `Output` preserves the spoken prefix.
- "twenty five" must yield both `25` (compound) and `205` (split `20` + `5`), and
  "seven hundred twenty four" must yield `724`, `70024`, `700204`. Never collapse
  ambiguity; never drop invalid candidates.
- Each word-derived group is at most 3 digits. `twelve hundred` → `12 100`; `twenty five
  hundred` → `20 500`; never `1200`, `2500`, or `25 100`. `thousand` and larger units
  are rejected (`GroupTooLargeError` → `group_too_large`), not silently ignored.
- Whisper is decoded twice: default for accurate digits, number-word prompt for grouping.
  Parse primary and `Transcript.alternatives`, then dedupe by digits.
- Parser/validator are deterministic and pure: no I/O, clock, randomness, locale, or
  Textual imports.
- Unit tests must not touch a mic, model, network, or real audio. Fake the boundaries;
  mark real-audio/model tests `@pytest.mark.integration`.
- Run audio/model/transcription off the Textual event loop (`asyncio.to_thread` or a
  worker); no blocking handler.
- Preload the Whisper model in the CLI **before** `RecognitionApp.run()`. Loading it
  inside a Textual worker makes tqdm/multiprocessing fail with `bad value(s) in
  fds_to_keep`. See `WhisperTranscriber.load()`.
- Microphone capture is 16 kHz mono, 30 ms frames, webrtcvad mode 3. Endpointing: 3 voiced
  frames to start, 3 s silence to stop, 10 s no-speech timeout, 30 s max, 300 ms pre-roll.
- Whisper filters segments with `no_speech_prob > 0.6`, `avg_logprob < -1.0`, or
  `compression_ratio > 2.4`. The word-form pass runs only when the default transcript
  contains a digit or number word; empty transcripts map to `no_speech`.
- Level/phase callbacks come from the capture thread; the TUI must marshal them with
  `call_from_thread`. `Q`/`Ctrl+C` calls `engine.cancel()` to release the stream; `C`
  cancels live capture and returns to the start screen without exiting.
- English only in UI strings, errors, identifiers, specs, and docs.
- Logs go to stderr—stdout belongs to Textual. No `print()` in TUI code.
- No cloud STT, no API keys, no persisted audio, no telephony features without asking.
- Errors are typed and actionable; `Ctrl+C` must release the microphone cleanly.

## Definition of done

Spec updated for behavior/UX changes, tests at the lowest useful layer, full verification
above passes, and the `--file` demo (plus live mic for UI work) has been exercised.
