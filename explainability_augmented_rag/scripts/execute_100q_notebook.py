#!/usr/bin/env python3
"""Execute the Ollama 100Q benchmark report notebook and save executed copy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "13_ollama_100q_benchmark_report.ipynb"
OUT = ROOT / "notebooks" / "13_ollama_100q_benchmark_report_executed.ipynb"


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(NB),
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode == 0:
        print(f"Executed notebook: {NB}")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
