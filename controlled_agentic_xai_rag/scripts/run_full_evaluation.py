"""Run full evaluation suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmds = [
        [sys.executable, "-m", "controlled_agentic_xai_rag.evaluation.eval_retrieval", "--limit", "20", "--bm25-only"],
        [sys.executable, "-m", "controlled_agentic_xai_rag.evaluation.acceptance"],
        [sys.executable, "-m", "controlled_agentic_xai_rag.evaluation.expert_review_export"],
    ]
    for cmd in cmds:
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
