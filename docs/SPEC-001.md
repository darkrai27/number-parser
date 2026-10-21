# Spec 001 — TUI for telephone number recognition

Status: accepted for the foundation
Depends on: SPEC-002 (recognition engine)
Related: `docs/constitution.md`

## Context and goal

A simple English-only TUI, started from the terminal. It asks the user to speak a
telephone number, shows live feedback while recording, then displays every plausible
number found in the utterance and whether it is a valid Greek number.

The TUI owns presentation and orchestration only. Transcription, candidate extraction,
and validation belong to SPEC-002 and are reached through the engine interface, so the
UI can be tested with a fake engine (no microphone, no model).

## Actors

- Final user: a person who wants to recognize a telephone number from a voice recording.
- Developer/operator: runs the app from a terminal, grants microphone permission.

## User stories

- H1: As a user I want a simple TUI so that I can recognize a telephone number from speech.
- H2: As a user I want clear status feedback and a clean results table so that I understand
  what the application is doing and which candidates are valid.

## Functional requirements (EARS)

- RF-1: When the application starts, the system shall display a start screen with the
  instruction "Press SPACE to start speaking", a "Start recognition" action, and a
  "Quit" action.
- RF-2: When the user activates start, the system shall request the default microphone and
  enter the listening state.
- RF-3: While listening, the system shall show a level meter, the elapsed recording time,
  and the status "Listening…".
- RF-4: While listening, the system shall update the level meter at least 10 times per
  second without blocking keyboard input.
- RF-5: When the engine reports that speech ended, the system shall enter a processing
  state with a spinner and the status "Transcribing…".
- RF-6: When the recognition result is available, the system shall display a table with the
  columns `#`, `Input`, `Output`, `Status`, and `Reason`.
- RF-7: The `Input` column shall contain digits only, separated by spaces, and reflect the
  candidate's grouping (for example `two ten` → `2 10`, split `twenty five` → `20 5`).
- RF-8: The `Status` column shall contain exactly `VALID` or `invalid`.
- RF-9: The `Reason` column shall be empty for valid numbers and contain the validation
  reason for invalid numbers.
- RF-10: The system shall show the parsed digits in `Output` for every candidate,
  including invalid rows, and list valid rows before invalid rows.
- RF-11: When the result contains no candidate, the system shall display
  "No telephone number recognized", the transcript, and the available actions.
- RF-12: If an operation fails, the system shall display an actionable error panel and
  shall return to the start screen without printing a traceback.
- RF-13: When results or an error are displayed, the system shall allow `R` to start a new
  recognition and `Q` to quit.
- RF-14: When the user presses `Ctrl+C` in any state, the system shall exit cleanly, stop
  recording, and release the audio device.
- RF-15: When the application is started with `--file PATH`, the system shall skip
  microphone capture, process that file through the same pipeline, and display the same
  results table.

## Non-functional requirements

- The UI, status messages, errors, and help text are English only.
- Keyboard-only operation is sufficient; a mouse must not be required.
- The layout must remain usable at 80x24 and degrade gracefully on resize.
- Blocking work (audio, model load, transcription) must run off the Textual event loop;
  no event handler may block for more than ~100 ms.
- The ASR model is loaded lazily once per process; the UI shows "Loading model…" while
  the first load happens.
- No audio is written to disk and no network request is made at runtime after the model
  has been downloaded.
- Startup to the start screen is under 1 second, excluding model download/load.
- Errors are specific and actionable (for example, "Microphone permission denied",
  "Unsupported audio file", "No speech detected").

## Edge cases

- Microphone permission denied or no input device.
- Silence, very short speech, clipped first word, or speech longer than 30 seconds.
- Background noise opens the VAD but no number is spoken.
- Multiple telephone numbers in one utterance.
- More than 50 candidates: show the first 50 and a "+N more" note.
- Empty, corrupt, unsupported, stereo, or non-16 kHz audio file.
- ASR returns an empty transcript or a transcript without digits.
- Model missing and the first download fails or is offline.
- Terminal resize while recording or while the table is shown.
- Repeated `R` presses and `Ctrl+C` during transcription or model load.

## Out of scope

- Editing a candidate, copying to clipboard, dialing, or storing history.
- Settings screens, themes, configurable keybindings, or persistence.
- Languages other than English.
- Cloud transcription, GPU setup, or packaged binaries.
- Candidate search/filtering; the table is small and fully visible by scrolling.

## Finalization criteria

- RF-1 through RF-15 have automated Textual pilot tests using a fake engine (no mic/model).
- Parser and validator unit tests from SPEC-002 pass.
- `ruff check .`, `ruff format --check .`, and `pytest` pass.
- Manual demo: live microphone run and `python -m main --file <fixture>.wav` both show a
  results table and allow retry and quit.

## Decisions recorded

- TUI framework: Textual (pilot tests cover the state machine).
- Input: live microphone by default; `--file` for deterministic demos and tests.
- Ambiguity is data, not an error: every candidate is shown, valid or not.
- Country code variants `0030`, `+30`, and `30` are accepted; a bare 10-digit national
  number is valid. `Output` preserves the spoken prefix and never adds a missing `0030`.
- Keyboard shortcuts: `SPACE` start, `R` retry, `Q` quit, `Ctrl+C` hard exit.

## Open questions

None blocking. Candidate ranking beyond "valid first, then discovery order" can be
revisited after the first manual demo.
