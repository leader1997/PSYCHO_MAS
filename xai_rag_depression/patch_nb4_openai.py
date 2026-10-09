#!/usr/bin/env python3
"""Patch 04 notebook: Ollama default generation, GPT judge."""

import json
from pathlib import Path

NB = Path(__file__).parent / "04_llm_generation_evaluation.ipynb"

INTRO = """# Notebook 4 - LLM Generation and System Evaluation

Compares **four RAG systems** using:

- **Generation (default):** local **Ollama** chat models (`llama3`, `gemma2`, `mistral`, …)
- **LLM-as-judge (default):** **OpenAI GPT** (`XAI_OPENAI_JUDGE_MODEL`)

| System | Context |
|--------|---------|
| S0 baseline | dataset + prediction |
| S1 rag_xai | + local XAI + literature |
| S2 rag_agg_xai | + aggregated XAI + literature |
| S3 full | + local + aggregated XAI + literature |

**Env:** `XAI_LLM_BACKEND=ollama` (default), `XAI_JUDGE_BACKEND=openai`, `XAI_OLLAMA_MODEL`, `OPENAI_API_KEY`, `XAI_CASE_LIMIT`.  
Or run `python run_nb4.py`."""

SETUP_MD = """## 1. Setup — Ollama (default) + GPT judge

| Role | Default backend | Override |
|------|-----------------|----------|
| Generation | **Ollama** (`llama3`, …) | `XAI_LLM_BACKEND=openai` |
| Judge | **GPT** | `XAI_JUDGE_BACKEND=ollama` |

Ollama preferred models: `llama3.1`, `llama3`, `gemma2`, `mistral`, `phi4`, `qwen2.5` (embedding models excluded)."""

SETUP_CODE = """import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import llm_client

OUTPUT_DIR = Path("outputs")
KB_DIR = OUTPUT_DIR / "kb"

LLM_OK = llm_client.is_available()
JUDGE_OK = llm_client.is_judge_available()

if LLM_OK:
    ollama_models = llm_client.list_ollama_models()
    print(f"Ollama chat models: {ollama_models}")
    print(
        f"Generation ({llm_client.backend_name()}): {llm_client.model_name()} | "
        f"Judge ({llm_client.judge_backend_name()}): {llm_client.judge_model_name()}"
    )
else:
    print(
        "LLM unavailable. Start Ollama (`ollama serve`, `ollama pull llama3`) or set "
        "XAI_LLM_BACKEND=openai with OPENAI_API_KEY."
    )

def generate(system, user, temperature=0.2):
    return llm_client.chat(system, user, temperature=temperature)
"""


def to_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0]["source"] = to_source(INTRO)
    cells[1]["source"] = to_source(SETUP_MD)
    cells[2]["source"] = to_source(SETUP_CODE)
    cells[2]["outputs"] = []
    cells[2]["execution_count"] = None

    for c in cells:
        if c["cell_type"] != "code":
            if c["cell_type"] == "markdown" and "## 4. LLM-as-judge" in "".join(c.get("source", [])):
                c["source"] = to_source("## 4. LLM-as-judge metrics (GPT default)")
            continue
        src = "".join(c.get("source", []))
        src = src.replace("OPENAI_OK", "LLM_OK")
        src = src.replace(
            '"llm_backend": "openai",\n                "llm_model": llm_client.model_name(),\n                "judge_model": llm_client.judge_model_name(),',
            '"llm_backend": llm_client.backend_name(),\n                "llm_model": llm_client.model_name(),\n                "judge_backend": llm_client.judge_backend_name(),\n                "judge_model": llm_client.judge_model_name(),',
        )
        if "if LLM_OK:" in src or "if OPENAI_OK:" in src:
            src = src.replace(
                'print("Skipped live generation (OPENAI_API_KEY not set).")',
                'print("Skipped live generation (LLM unavailable).")',
            )
            src = src.replace(
                "if OPENAI_OK and example_answers:",
                "if LLM_OK and example_answers:",
            )
            c["source"] = to_source(src)
            if "rows.append(row)" in src:
                c["outputs"] = []
                c["execution_count"] = None

    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Patched {NB}")


if __name__ == "__main__":
    main()
