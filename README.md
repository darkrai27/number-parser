# Omilia Challenge

English-only terminal application that recognizes a spoken telephone number, lists every
plausible digit variant, and flags which variants are valid Greek numbers.

Valid Greek numbers are 10-digit national numbers starting with `2` or `69`. The `0030`
country code is optional; only `0030` and bare national form are accepted, and output
never adds a prefix that was not spoken.

Spoken groups are limited to three digits: `twelve hundred` → `12 100`, `twenty five
hundred` → `20 500`. Utterances containing `thousand` or larger units are rejected.
Digit sequences keep the same ambiguity: `69` → `69` / `609`, `20 5` → `205` / `25`,
and `2 10` stays `2 10`, never `200 10`.

## Setup

Requires Python 3.11.x. Python 3.12 is not supported.

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`.[dev]` installs runtime dependencies plus pytest and ruff. For runtime only:

```bash
python -m pip install -e .
```

Verify the install:

```bash
python -m main --help
pytest
```

The first real transcription downloads `base.en` (~150 MB) to
`~/.cache/huggingface`. Audio decoding uses PyAV, so no external `ffmpeg` binary is
needed. Live capture prompts for microphone permission on first use; on macOS check
System Settings → Privacy & Security → Microphone.

Run the real-audio tests with:

```bash
pytest -m integration
```

They use the fixtures in `audios/examples/` and the downloaded model.

## Usage

```bash
python -m main                  # live microphone TUI
python -m main --file audio.mp3 # process a file instead of the microphone
python -m main --model small.en
```

In live mode, press `SPACE` and speak the number in English. Recording stops after three
seconds of silence, or press `SPACE` again to stop immediately. Press `C` while listening
to cancel and return to the start screen. `R` retries, `Q` quits.

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
