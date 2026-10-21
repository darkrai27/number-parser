# Omilia Challenge

English-only terminal application that recognizes a spoken telephone number, lists every
plausible digit variant, and flags which variants are valid Greek numbers.

Valid Greek numbers are 10-digit national numbers starting with `2` or `69`. The `0030`
country code is optional; `0030`, `+30`, and `30` are accepted, and output never adds a
prefix that was not spoken.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first real transcription downloads the `base.en` model (~150 MB). Audio decoding is
handled by PyAV (installed with faster-whisper), so no external `ffmpeg` binary is needed.
Live capture requires microphone permission.

## Usage

```bash
python -m main                 # live microphone TUI
python -m main --file call.wav # process a file instead of the microphone
python -m main --model small.en
```

In live mode, press `SPACE` and speak the number in English. Recording stops after three
seconds of silence, or press `SPACE` again to stop immediately. `R` retries, `Q` quits.

## Docs

- `docs/SPEC-001.md` — TUI and orchestration
- `docs/SPEC-002.md` — recognition engine, parser, and validation
- `docs/constitution.md` — non-negotiable project rules
- `AGENTS.md` — commands and conventions for coding agents

## Verification

```bash
ruff check . && ruff format --check . && pytest   # unit suite
pytest -m integration                             # real TTS audio + model
```
