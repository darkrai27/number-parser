# Constitution — Omilia Challenge

Non-negotiable rules for this repository. If a task conflicts with one of these, stop
and ask before changing it. Specs may add detail; they may not weaken these rules.

## 1. Scope

- English speech only. The application and its UI are English only.
- The only domain is telephone-number recognition and Greek-number validation.
- No telephony integration, dialing, recording history, or cloud services.

## 2. Architecture boundaries

- `main.tui` renders and orchestrates; it does not parse numbers, access audio devices,
  or call faster-whisper directly.
- `main.numbers` (parser + validator) is pure: stdlib and shared dataclasses only. No I/O,
  network, clock, randomness, locale, or Textual imports.
- `main.audio` emits mono 16 kHz float32 samples; `main.asr` exposes a `Transcriber`
  protocol and a faster-whisper adapter behind a lazy import.
- Blocking audio/model/ASR work runs outside the Textual event loop.
- The parser/validator must be usable without installing audio or ASR extras at import
  time (lazy imports at the boundaries).

## 3. Ambiguity is data

- Never silently pick one interpretation of an utterance.
- Return every plausible candidate, including invalid ones, with a reason.
- "twenty five" produces both `25` (compound) and `205` (split, `20` + `5`).
- Deduplicate by digits only, preserving the first source fragment and discovery order.

## 4. Validation

- Canonical Greek form: `0030` + 10 national digits.
- National number is valid only if it starts with `2` or `69`.
- Normalize `0030`, `+30`, and `30` as the country code; a missing country code is
  invalid (`missing_country_code`) but the candidate is still displayed.
- Do not invent a country code or pad/truncate digits to make a number valid.

## 5. Privacy and offline operation

- Audio is processed locally and is not persisted by default.
- No network calls at runtime after the ASR model is cached; no API keys in the repo.
- Do not log transcripts or audio paths unless an explicit debug flag is enabled.

## 6. Testability and determinism

- Unit tests never touch a microphone, the network, or a real model.
- Use fake `AudioSource`/`Transcriber` implementations and pure parser/validator tests.
- Tests that need real audio or a downloaded model are marked `integration` and opt-in.
- Parser/validator behavior is specified by table-driven tests.

## 7. UX

- English-only copy, no tracebacks, and every error says what the user can do next.
- Keyboard-only operation; `SPACE` start, `R` retry, `Q` quit, `Ctrl+C` clean exit.
- Live feedback is required while listening and while transcribing.
- At least 80x24 terminals must remain usable.

## 8. Dependencies and tooling

- Python 3.11.x, Textual, faster-whisper, sounddevice, webrtcvad-wheels, numpy.
- Runtime dependencies are declared in `pyproject.toml`; do not add one without stating
  why in the task summary and updating `AGENTS.md` if it changes daily workflow.
- `ruff` is the formatter and linter. Do not introduce a second formatter.
- No generated code, migrations, or build artifacts are committed.

## 9. Definition of done

1. Read the active spec and this constitution before changing behavior.
2. Update the spec first when behavior or UX changes.
3. Add or update tests at the lowest useful layer.
4. Run `ruff check . && ruff format --check . && pytest`.
5. Run the `--file` demo for end-to-end changes; use the live microphone for UI changes.
6. Report commands run and any verification that could not be performed.
