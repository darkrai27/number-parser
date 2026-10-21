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
    build_parser().parse_args(argv)
    print(
        "main: foundation scaffold only; the recognition pipeline is not implemented yet.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
