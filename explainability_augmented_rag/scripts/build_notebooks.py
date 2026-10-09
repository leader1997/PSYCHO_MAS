#!/usr/bin/env python3
"""Build all Phase 2 notebooks (01-12)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

NOTEBOOKS = [
    ("01_build_corpus.ipynb", "Build Corpus", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'retrieval/build_corpus.py'])"),
    ("02_standardize_xai_cards.ipynb", "Standardize XAI Cards", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'xai/build_xai_cards.py'])"),
    ("03_query_generation.ipynb", "Query Generation", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'generation/query_builder.py'])"),
    ("04_retriever_ablation.ipynb", "Retriever Ablation", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'retrieval/retriever_eval.py', '--limit', '10'])"),
    ("05_evidence_synthesis.ipynb", "Evidence Synthesis", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'generation/evidence_synthesis.py', '--limit', '3'])"),
    ("06_generate_narratives.ipynb", "Generate Narratives", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'generation/narrative_generator.py', '--limit', '3'])"),
    ("07_safety_validation.ipynb", "Safety Validation", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'safety/safety_validator.py'])"),
    ("08_faithfulness_checking.ipynb", "Faithfulness Checking", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'safety/faithfulness_checker.py'])"),
    ("09_build_rag_benchmark.ipynb", "Build RAG Benchmark", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'evaluation/build_benchmark.py'])"),
    ("10_ragas_evaluation.ipynb", "RAGAS Evaluation", "import subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'evaluation.ragas_evaluation', '--limit', '5'])"),
    ("11_full_system_ablation.ipynb", "Full System Ablation", "import subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'evaluation.ablation_evaluation'])"),
    ("12_error_analysis.ipynb", "Error Analysis", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'evaluation/error_analysis.py'])\nsubprocess.check_call([sys.executable, 'evaluation/generate_paper_outputs.py'])"),
    ("expert_review_analysis.ipynb", "Expert Review Analysis", "import subprocess, sys\nsubprocess.check_call([sys.executable, 'evaluation/expert_review_analysis.py'])"),
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for fname, title, run_code in NOTEBOOKS:
        cells = [
            md(f"# {title}\n\nExplainability-Augmented RAG Phase 2\n"),
            md("## Run"),
            code(f"import os\nos.chdir(r'{ROOT}')\n" + run_code),
        ]
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "cells": cells,
        }
        out = NB_DIR / fname
        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
