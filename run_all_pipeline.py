"""Run full XAI-RAG pipeline (notebooks 1-4 logic) with FAST mode."""
import os
os.environ.setdefault("XAI_FAST", "1")

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

steps = [
    ("Notebook 1: models + explainability", [sys.executable, str(ROOT.parent / "_run_nb1.py")]),
    ("Notebook 2: stability + cards", [sys.executable, str(ROOT.parent / "_run_nb2.py")]),
    ("Notebook 3: RAG KB + PDFs", [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
     "--execute", "--inplace", "--ExecutePreprocessor.timeout=7200", "03_rag_knowledge_base.ipynb"]),
    ("Notebook 4: LLM eval", [sys.executable, str(ROOT.parent / "_run_nb4.py")]),
]

for name, cmd in steps:
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"FAILED: {name} (exit {r.returncode})", flush=True)
        sys.exit(r.returncode)
print("\nPipeline complete.", flush=True)
