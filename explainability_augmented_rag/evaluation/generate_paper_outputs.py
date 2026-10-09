"""Generate paper-ready outputs (WP14)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import load_config, path  # noqa: E402


def _read_csv(name: str) -> pd.DataFrame:
    p = path("results_dir") / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def generate_methods_section() -> str:
    cfg = load_config()
    return f"""# RAG Methodology (Draft)

## Overview
We implemented an Explainability-Augmented RAG pipeline for **non-clinical** depression/anxiety **screening communication** using HMS 2024–2025 survey features. The system does **not** diagnose mental illness.

## Knowledge base
- Section-aware chunking (~{cfg['corpus']['chunk_target_tokens']} tokens, overlap {cfg['corpus']['chunk_overlap_tokens']})
- Metadata: document ID, title, year, page, section, source type, domain tags
- Corpus includes peer-reviewed literature PDFs and project artifacts (XAI cards, ethics)

## Retrieval
Four strategies compared: dense ({cfg['retrieval']['embedding_model']}), BM25, hybrid RRF, hybrid + cross-encoder reranker ({cfg['retrieval']['cross_encoder_model']}).

## Generation pipeline
1. Query construction from case profile + XAI factors (4 query types)
2. Evidence synthesis (structured JSON, chunk-linked claims)
3. Safe narrative generation (structured JSON, mandatory disclaimer)
4. Safety validator (8 checks + revision loop)
5. Claim-level faithfulness checker

## Evaluation
- 100-question RAG benchmark with remapped gold chunk IDs
- RAGAS six metrics (Batal et al. 2026 aligned) across systems S1–S6
- Full ablation A1–A8 on de-identified cases
- Human expert review protocol (template provided)

## LLM configuration
- Generation/judge: {cfg['llm']['generation_model']}
- Embeddings: {cfg['retrieval']['embedding_model']}
- Random seed: {cfg['project']['random_seed']}
"""


def generate_results_section() -> str:
    parts = ["# RAG Results (Draft)\n"]
    ret = _read_csv("retriever_ablation_summary.csv")
    if not ret.empty:
        parts.append("## Retriever ablation\n```\n" + ret.to_string(index=False) + "\n```\n")
    ragas = _read_csv("ragas_summary_table.csv")
    if not ragas.empty:
        parts.append("## RAGAS summary\n```\n" + ragas.to_string(index=False) + "\n```\n")
    abl = _read_csv("ablation_summary_table.csv")
    if not abl.empty:
        parts.append("## System ablation\n```\n" + abl.to_string(index=False) + "\n```\n")
    safety = _read_csv("safety_reports.csv")
    if not safety.empty:
        rate = (safety["overall_safety"] == "pass").mean()
        parts.append(f"## Safety pass rate: {rate:.1%}\n")
    cite = _read_csv("citation_precision_results.csv")
    if not cite.empty:
        parts.append("## Citation precision\n```\n" + cite.to_string(index=False) + "\n```\n")
    parts.append("""
## Limitations
- Screening communication only; not validated for clinical deployment
- XAI factors are model contributors, not causal explanations
- Retrieval precision remains a bottleneck for some benchmark categories
- Human expert review is required before any real-world use
""")
    return "\n".join(parts)


def main() -> int:
    out_dir = path("paper_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_methods_rag_section.md").write_text(generate_methods_section(), encoding="utf-8")
    (out_dir / "paper_results_rag_section.md").write_text(generate_results_section(), encoding="utf-8")

    tables = {}
    for name, key in [
        ("retriever_ablation_summary.csv", "retriever"),
        ("ragas_summary_table.csv", "ragas"),
        ("ablation_summary_table.csv", "ablation"),
        ("safety_reports.csv", "safety"),
        ("citation_precision_results.csv", "faithfulness"),
    ]:
        df = _read_csv(name)
        if not df.empty:
            tables[key] = df

    if tables:
        xlsx = out_dir / "tables_for_paper.xlsx"
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            for sheet, df in tables.items():
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
        print(f"Tables -> {xlsx}")

    print(f"Paper sections -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
