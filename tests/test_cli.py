from __future__ import annotations

import subprocess
import sys


def test_module_help_lists_cli_options() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--file" in result.stdout
    assert "--model" in result.stdout
