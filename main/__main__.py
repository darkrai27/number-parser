from __future__ import annotations

import argparse
import sys

from main import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main",
        description="Recognize a spoken telephone number and flag valid Greek numbers.",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="process an audio file instead of the microphone",
    )
    parser.add_argument(
        "--model",
        default="base.en",
        help="faster-whisper model name (default: base.en)",
    )
    parser.add_argument("--version", action="version", version=f"main {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from main.asr.whisper import WhisperTranscriber
    from main.engine import RecognitionEngine
    from main.tui.app import RecognitionApp

    transcriber = WhisperTranscriber(args.model)
    if args.file:
        print(f"Loading speech model {args.model}…", file=sys.stderr, flush=True)
        try:
            transcriber.load()
        except Exception as exc:
            print(f"Could not load speech model: {exc}", file=sys.stderr)
            return 1

    engine = RecognitionEngine(transcriber)
    RecognitionApp(engine, file_path=args.file).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
