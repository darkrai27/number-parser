# Omilia Challenge

English-only terminal application that recognizes a spoken telephone number, lists every
plausible digit variant, and flags which variants are valid Greek numbers.

Valid Greek numbers are `0030` followed by a 10-digit national number starting with `2`
or `69`. Country-code variants `0030`, `+30`, and `30` are normalized; a bare national
number is shown but marked invalid.

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
python -m main                 # live microphone + TUI
python -m main --file call.wav # process a file instead of the microphone
python -m main --model small.en
```

## Docs

- `SPEC-001.md` — TUI and orchestration
- `SPEC-002.md` — recognition engine, parser, and validation
- `docs/constitution.md` — non-negotiable project rules
- `AGENTS.md` — commands and conventions for coding agents

## Verification

```bash
ruff check . && ruff format --check . && pytest
```
