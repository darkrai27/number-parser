# Spec 002 — Voice recognition engine

Status: accepted for the foundation
Related: SPEC-001 (TUI), `docs/constitution.md`

## Context and goal

An English-only, local voice recognition engine for telephone numbers. It accepts a live
microphone stream or an audio file, detects when speech starts and ends, transcribes it,
extracts every plausible digit sequence, and validates candidates against the Greek
numbering rules. It returns a typed result to the TUI; it does not render anything.

Phone numbers are spoken as a sequence of numbers, each with at most three digits (for
example `2 10 69 30 6 6 4`); the candidate is the concatenation of those numbers.
Different groupings and ASR noise make several readings plausible. The engine must
preserve ambiguity instead of silently choosing one interpretation. For example,
"twenty five" can be heard as `25` (compound) or `20 5` → `205` (split), and
"seven hundred twenty four" as `724`, `70024`, or `700204`. All are candidates and all
are displayed. Only candidates that normalize to `0030` + a valid 10-digit Greek
national number are marked valid.

Whisper often normalizes spoken numbers to digits (`0-0-30-69-724-1352`), destroying the
grouping evidence. The adapter therefore performs two decodes: one default pass for the
most accurate digit string, and one pass with a number-word prompt to recover the spoken
words. Candidates from both transcripts are unioned.

## Actors

- Final user: speaks the number after the TUI prompts.
- Calling module: the TUI (SPEC-001), which consumes `RecognitionResult`.
- Operator: optionally supplies a file and selects the model.

## User stories

- H1: As a user I want a voice recognition engine so that it recognizes all plausible
  telephone numbers from my recording.
- H2: As a user I want to see every plausible variation and whether it is a valid Greek
  number, so that I can judge recognition errors myself.

## Pipeline

```
AudioSource (microphone | file)
  -> mono 16 kHz float32 samples
  -> VAD endpointing (speech start / speech end)
  -> Transcribed twice (default digits + number-word prompt)
  -> Candidate parser (number words, digits, homophones, scale ambiguity)
  -> Greek validator (normalize + validate)
  -> RecognitionResult
```

The TUI calls the engine; the engine never imports TUI code.

## Data contracts

```python
@dataclass(frozen=True)
class Transcript:
    text: str
    language: str          # "en"
    duration_s: float
    alternatives: tuple[str, ...] = ()  # word-form pass, when different

@dataclass(frozen=True)
class Candidate:
    digits: str            # digits only, as heard; may omit the country code
    input: str             # digits only, space-separated, candidate grouping
    interpretation: str    # "direct" | "compound" | "split" | "homophone"
    output: str | None     # cleaned digits for every parsed candidate; None only if no digits
    is_valid: bool
    reason: str            # "valid" | "wrong_length" | "wrong_prefix" | "not_digits"

@dataclass(frozen=True)
class RecognitionResult:
    transcript: str
    candidates: tuple[Candidate, ...]
    error: ErrorCode | None  # None on success; typed failure otherwise
    detail: str = ""         # diagnostic text; never a traceback
```

`ErrorCode` is a stable machine-readable enum shared by the engine and the TUI:

```python
class ErrorCode(StrEnum):
    FILE_NOT_FOUND = "file_not_found"
    UNSUPPORTED_FILE = "unsupported_file"
    NO_INPUT_DEVICE = "no_input_device"
    AUDIO_READ_ERROR = "audio_read_error"
    NO_SPEECH = "no_speech"
    CANCELLED = "cancelled"
    TRANSCRIPTION_ERROR = "transcription_error"
    NO_NUMBER = "no_number"
```

`detail` carries the failing path or exception message for diagnostics. The TUI owns the
user-facing copy; the engine never returns presentation prose.

## Error taxonomy

| Code | Trigger |
| --- | --- |
| `file_not_found` | `--file` path does not exist or is not a file. |
| `unsupported_file` | The audio file cannot be decoded. |
| `no_input_device` | No default input device, or the device cannot be opened. |
| `audio_read_error` | The microphone stream fails while capturing. |
| `no_speech` | Capture ended with no samples to transcribe. |
| `cancelled` | The user cancelled capture with `C`; buffered audio is discarded. |
| `transcription_error` | Model inference or transcript processing fails. |
| `no_number` | The transcript contains no digits or number words. |

Model loading is a CLI startup step, not an engine call, so it is not an `ErrorCode`.
The CLI reports `Could not load speech model: {detail}` on stderr and exits with status 1.

## Functional requirements (EARS)

- RF-1: When recognition starts, the engine shall capture mono 16 kHz audio from the
  default input device, or decode `--file PATH` when supplied.
- RF-2: While capturing, the engine shall classify 20–30 ms frames with a VAD and shall
  consider speech started after three consecutive voiced frames.
- RF-3: When 3.0 seconds of silence follows speech, the engine shall stop capturing and
  process the buffered speech.
- RF-4: If no speech is detected within 10 seconds, the engine shall stop and return a
  `no_speech` result.
- RF-5: If speech continues for 30 seconds, the engine shall stop and process what was
  captured.
- RF-6: When speech ends, the engine shall transcribe it twice in English with
  faster-whisper using the configured model (default `base.en`, CPU, `int8`): once with
  default decoding and once with a number-word `initial_prompt`. The second pass runs
  only if the first contains a digit or number word. Segments with
  `no_speech_prob > 0.6`, `avg_logprob < -1.0`, or `compression_ratio > 2.4` are discarded.
  Both resulting transcripts are returned, with the word-form pass in
  `Transcript.alternatives`.
- RF-7: When transcription completes, the engine shall extract every plausible telephone
  number candidate from every transcript.
- RF-8: The parser shall support digits, units, teens, tens, hundreds, `oh`/`o`/`zero`/
  `nought`/`nil`, `double X`, `triple X`, hyphens, and common ASR homophones
  (`two/to/too`, `four/for`, `one/won`, `eight/ate`, `three/free`). Each word-derived
  group is limited to three digits.
- RF-9: When a tens word is immediately followed by a unit word, the parser shall emit
  both the compound candidate and the split candidate (for example "twenty five" →
  `25` and `205`).
- RF-10: When a hundreds phrase cannot form a group of at most three digits, the parser
  shall keep the prefix and `100` as separate groups (`twelve hundred` → `12 100`,
  `twenty five hundred` → `20 500`). When it can, compound and split candidates are
  emitted (`seven hundred twenty four` → `724`, `70024`, `700204`).
- RF-10a: If an utterance contains `thousand` or a larger unit (`million`, `billion`,
  `trillion`, ...), the parser shall raise `GroupTooLargeError`; the engine shall return
  `group_too_large`.
- RF-11: The parser shall deduplicate candidates by digits across all transcripts, keep
  the first source fragment, and never discard a candidate because it is invalid.
- RF-12: If no transcript contains digits, the engine shall return a `no_number` result
  with the primary transcript and an empty candidate list.
- RF-13: The validator shall normalize separators and the optional country-code variants
  `0030`, `+30`, and `30` to `0030` + national number; a bare 10-digit national number
  is accepted.
- RF-14: The validator shall mark a candidate valid when the national number is exactly
  10 digits and starts with `2` or `69`, whether or not the country code was spoken.
- RF-15: The validator shall mark every other candidate invalid with a specific reason:
  `wrong_length`, `wrong_prefix`, or `not_digits`.
- RF-16: If audio input, decoding, or transcription fails, the engine shall return a
  `RecognitionResult` carrying an `ErrorCode` (`file_not_found`, `unsupported_file`,
  `no_input_device`, `audio_read_error`, `transcription_error`, or
  `group_too_large`) and a diagnostic `detail` instead of raising into the TUI. Silence,
  user cancellation, and an empty candidate set are typed outcomes (`no_speech`,
  `cancelled`, `no_number`), not exceptions. Model loading happens in the CLI before the
  TUI starts; a load failure is reported to stderr and exits with status 1.

## Parsing rules

- Number words are combined using normal English rules within a group:
  `twenty five` → 25, `one hundred twenty three` → 123.
- Adjacent groups are concatenated: `two five` → 25, `zero zero three zero` → 0030.
- The tens+unit rule (RF-9) emits both compound and split forms for every adjacent pair,
  including inside larger phrases: `sixty nine` → `69`, `609`.
- Each word-derived group is limited to three digits.
- The hundreds rule (RF-10) binds `hundred` to the preceding number when the product is
  at most 999: `seven hundred` → `700`.
- If the prefix is a tens+unit pair and the product would exceed 999, `hundred` binds to
  the unit: `twenty five hundred` → `20 500` (`20500`), never `2500` or `25 100`.
- Otherwise the prefix and `100` stay separate: `twelve hundred` → `12 100` (`12100`),
  never `1200`.
- Remainders still produce compound and split variants when they fit:
  `seven hundred twenty four` → `724`, `70024`, `700204`.
- `thousand` and larger units (`million`, `billion`, `trillion`, ...) are rejected:
  they cannot form a group of at most three digits.
- Candidates are generated by composing every item variant across the utterance, then
  deduplicating by digits. The branch count is capped at 256.
- Dual transcripts are parsed independently and unioned; the default transcript is shown
  in the TUI status line.
- Each candidate carries an `input` string: digits only, space-separated, reflecting that
  candidate's grouping (`twenty five` compound → `25`, split → `20 5`).
- `double five` → 55, `triple seven` → 777.
- Homophones produce separate candidates only when they change the digit sequence.
- Filler words (`my`, `number`, `is`, `call`, `me`, `at`, `plus`, `country`, `code`) are
  ignored, not treated as digits.
- A transcript may contain multiple numbers; candidates from all maximal digit runs are
  returned.
- "point" and everything after a decimal marker is out of scope and ignored.

## Greek validation rules

```
input: 0030 2101234567 / +30 2101234567 / 30 2101234567 / 2101234567
strip: spaces, "-", ".", "(", ")", "/"
prefix:
  "0030"              -> national = rest
  "+30"               -> national = rest
  "30" (total 12)     -> national = rest
  otherwise           -> national = whole input
valid national: ^(2\d{9}|69\d{8})$
output: cleaned input digits, preserving the spoken prefix
```

A bare 10-digit national number is valid and its output stays 10 digits; `0030` is not
added. `output` is the cleaned digits for every parsed candidate, including invalid
lengths. A country code followed by anything other than 10 digits is invalid
(`wrong_length`).

## Non-functional requirements

- Local/offline: no cloud API, no network calls after the model is cached; no API keys.
- Default model `base.en`; overridable with `--model`. Model is loaded once and reused.
- Parser and validator are pure, synchronous, deterministic, and free of I/O, clock,
  locale, randomness, and global state.
- The engine writes no audio to disk by default; optional debug saving requires an
  explicit flag.
- Logs go to stderr; stdout belongs to the TUI.
- Target latency: at most ~8 seconds from end of speech to result for a 15-second clip
  with `base.en` on a modern laptop, covering both ASR passes (guideline, not a CI gate).
- Memory budget: under ~1 GB with `base.en` on CPU.
- Errors are typed and actionable; no traceback reaches the user.

## Edge cases

- Silence, noise-only input, clipped start, very long silence, speech over 30 seconds.
- Background noise that opens the VAD but contains no number.
- Multiple numbers in one utterance.
- `0030` spoken as words, `+30`, `30`, or omitted entirely.
- Leading zeros, repeated digits, `double oh`, `triple zero`.
- More or fewer than 10 national digits; national numbers starting with `1`, `69`, `2`.
- ASR homophones and word-boundary errors (`twenty five` vs `twenty-five`).
- Hundreds phrases that would exceed three digits (`twelve hundred`, `twenty five
  hundred`) and utterances containing `thousand` or larger units.
- Whisper returning digits (`0-0-30-69-724-1352`) or words for the same audio; both must
  be parsed, and the word pass must recover grouping evidence.
- Empty transcript, transcript without digits, unsupported sample rate, stereo input.
- User cancels live capture before or during speech; buffered audio is discarded.
- Model not cached and the machine is offline; corrupted audio file.

## Out of scope

- Greek-language speech or language detection; English only.
- Speaker diarization, speaker identification, or noise suppression models.
- Streaming partial transcripts; processing starts after the utterance ends.
- Cloud STT, GPU tuning, telephony integration, or dialing.
- Confidence calibration and automatic candidate ranking beyond valid-first ordering.
- Editing or correcting a candidate in the UI.

## Finalization criteria

- Table-driven parser tests cover units, teens, tens, hundreds, `double`/`triple`,
  homophones, country-code variants, multiple numbers, tens+unit ambiguity,
  hundred-group ambiguity, and the three-digit group limit.
- `thousand` and larger units are rejected with `group_too_large` and a TUI error message.
- Validator tests cover every reason code and the `2` / `69` prefix rule.
- Engine tests use fake transcripts and typed failures, including `Transcript.alternatives`;
  every `ErrorCode` is asserted; no microphone or model download.
- Real-audio tests run `pytest -m integration` against `audios/examples/` (not committed)
  and skip when the directory is absent.
- `ruff check .`, `ruff format --check .`, and `pytest` pass.

## Decisions recorded

- STT: faster-whisper, local, `base.en` default, CPU `int8`.
- ASR ambiguity: two decodes per file — default for accurate digits, number-word
  `initial_prompt` for grouping — but the word pass runs only when the default transcript
  contains a number. Low-confidence segments are filtered to prevent silence/noise
  hallucinations.
- VAD: `webrtcvad` mode 3 frame classification with 3-second silence endpointing and a
  10-second no-speech timeout.
- Audio: live microphone by default; `--file` bypasses capture.
- Cancellation: `C` stops a live capture, discards buffered audio, and returns to the
  start screen without exiting; cancellation is a typed `cancelled` outcome.
- Ambiguity: emit compound and split interpretations for tens+unit and hundreds, plus
  homophones; never silently coerce to one answer.
- Group size: each word-derived group is at most three digits. Hundreds that would
  overflow are split; `thousand` and larger units are rejected as `group_too_large`.
- Validation: country-code variants are accepted; a bare 10-digit national number is
  valid. Output preserves the spoken prefix and never adds `0030`.

## Open questions

- Is `free` → `3` too aggressive in a phone-number context? Keep it, marked as a
  homophone, and revisit after real recordings.
- The second ASR pass doubles transcription time; revisit if real-time feedback needs it
  to be conditional.
