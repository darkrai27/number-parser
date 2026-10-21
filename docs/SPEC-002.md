# Spec 002 — Voice recognition engine

Status: accepted for the foundation
Related: SPEC-001 (TUI), `docs/constitution.md`

## Context and goal

An English-only, local voice recognition engine for telephone numbers. It accepts a live
microphone stream or an audio file, detects when speech starts and ends, transcribes it,
extracts every plausible digit sequence, and validates candidates against the Greek
numbering rules. It returns a typed result to the TUI; it does not render anything.

Phone numbers are spoken with different groupings, and ASR output is noisy. The engine
must preserve ambiguity instead of silently choosing one interpretation. For example,
"twenty five" can be heard as `25` (compound) or `20 5` → `205` (split); both are
candidates and both are displayed. Only candidates that normalize to
`0030` + a valid 10-digit Greek national number are marked valid.

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
  -> Transcriber (faster-whisper, English)
  -> Candidate parser (number words, digits, homophones, ambiguity)
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

@dataclass(frozen=True)
class Candidate:
    digits: str            # digits only, as heard; may omit the country code
    heard_as: str          # transcript fragment that produced it
    interpretation: str    # "direct" | "compound" | "split" | "homophone"
    normalized: str | None # "0030XXXXXXXXXX" when the country code is present
    is_valid: bool
    reason: str            # "valid" or a short machine-readable reason

@dataclass(frozen=True)
class RecognitionResult:
    transcript: str
    candidates: tuple[Candidate, ...]
    error: str | None      # None on success; typed message otherwise
```

## Functional requirements (EARS)

- RF-1: When recognition starts, the engine shall capture mono 16 kHz audio from the
  default input device, or decode `--file PATH` when supplied.
- RF-2: While capturing, the engine shall classify 20–30 ms frames with a VAD and shall
  consider speech started after three consecutive voiced frames.
- RF-3: When 1.0 second of silence follows speech, the engine shall stop capturing and
  process the buffered speech.
- RF-4: If no speech is detected within 10 seconds, the engine shall stop and return a
  `no_speech` result.
- RF-5: If speech continues for 30 seconds, the engine shall stop and process what was
  captured.
- RF-6: When speech ends, the engine shall transcribe it in English with faster-whisper,
  using the configured model (default `base.en`, CPU, `int8`).
- RF-7: When transcription completes, the engine shall extract every plausible telephone
  number candidate from the transcript.
- RF-8: The parser shall support digits, units, teens, tens, hundreds, and thousands,
  `oh`/`o`/`zero`/`nought`/`nil`, `double X`, `triple X`, hyphens, and common ASR
  homophones (`two/to/too`, `four/for`, `one/won`, `eight/ate`, `three/free`).
- RF-9: When a tens word is immediately followed by a unit word, the parser shall emit
  both the compound candidate and the split candidate (for example "twenty five" →
  `25` and `205`).
- RF-10: The parser shall deduplicate candidates by digits, keep the first source
  fragment, and never discard a candidate because it is invalid.
- RF-11: If the transcript contains no digits, the engine shall return a `no_number`
  result with the transcript and an empty candidate list.
- RF-12: The validator shall normalize separators and the country-code variants `0030`,
  `+30`, and `30` to `0030` + national number.
- RF-13: The validator shall mark a candidate valid only when the national number is
  exactly 10 digits and starts with `2` or `69`.
- RF-14: The validator shall mark every other candidate invalid with a specific reason:
  `missing_country_code`, `wrong_length`, `wrong_prefix`, or `not_digits`.
- RF-15: If audio input, decoding, model loading, or transcription fails, the engine
  shall return a typed error (`no_input_device`, `audio_read_error`, `unsupported_file`,
  `model_load_error`, `transcription_error`) instead of raising into the TUI.

## Parsing rules

- Number words are combined using normal English rules within a group:
  `twenty five` → 25, `one hundred twenty three` → 123.
- Adjacent groups are concatenated: `two five` → 25, `zero zero three zero` → 0030.
- The tens+unit split rule (RF-9) applies only to adjacent tens and unit words, not to
  teens or scales, to avoid combinatorial noise.
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
  "0030" -> national = rest
  "+30"  -> national = rest
  "30"   -> national = rest
  other  -> missing_country_code
valid national: ^(2\d{9}|69\d{8})$
canonical: "0030" + national
```

A bare 10-digit national number remains visible but is invalid because the country code
was not spoken. `0030` followed by anything other than 10 digits is invalid.

## Non-functional requirements

- Local/offline: no cloud API, no network calls after the model is cached; no API keys.
- Default model `base.en`; overridable with `--model`. Model is loaded once and reused.
- Parser and validator are pure, synchronous, deterministic, and free of I/O, clock,
  locale, randomness, and global state.
- The engine writes no audio to disk by default; optional debug saving requires an
  explicit flag.
- Logs go to stderr; stdout belongs to the TUI.
- Target latency: at most ~5 seconds from end of speech to result for a 15-second clip
  with `base.en` on a modern laptop (guideline, not a CI gate).
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
- Empty transcript, transcript without digits, unsupported sample rate, stereo input.
- Model not cached and the machine is offline; corrupted audio file.

## Out of scope

- Greek-language speech or language detection; English only.
- Speaker diarization, speaker identification, or noise suppression models.
- Streaming partial transcripts; processing starts after the utterance ends.
- Cloud STT, GPU tuning, telephony integration, or dialing.
- Confidence calibration and automatic candidate ranking beyond valid-first ordering.
- Editing or correcting a candidate in the UI.

## Finalization criteria

- Table-driven parser tests cover units, teens, tens, scales, `double`/`triple`,
  homophones, country-code variants, multiple numbers, and the tens+unit ambiguity.
- Validator tests cover every reason code and the `2` / `69` prefix rule.
- Engine integration tests use a fake `Transcriber` and a small WAV fixture; they do not
  require a microphone or a model download.
- Real-audio/model tests are marked `integration` and are opt-in.
- `ruff check .`, `ruff format --check .`, and `pytest` pass.

## Decisions recorded

- STT: faster-whisper, local, `base.en` default, CPU `int8`.
- VAD: `webrtcvad` frame classification with a 10-second no-speech timeout.
- Audio: live microphone by default; `--file` bypasses capture.
- Ambiguity: emit compound and split interpretations for tens+unit, plus homophones;
  never silently coerce to one answer.
- Validation: country code variants are normalized, but the country code must be present.
- Multiple ASR n-best decodes are out of scope for the first iteration; ambiguity comes
  from the deterministic parser.

## Open questions

- Should a valid 10-digit national number with no spoken country code be offered as a
  "possible" normalized candidate? Deferred; current decision is to show it invalid.
- Is `free` → `3` too aggressive in a phone-number context? Keep it, marked as a
  homophone, and revisit after real recordings.
