"""Run acceptance gates G1–G6."""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from local_agentic_xai_rag.config_loader import path, validate_phase2_paths
from local_agentic_xai_rag.llm.ollama_client import MODEL, ping


REQUIRED_NARRATIVE_FIELDS = [
    "case_id", "screening_summary", "model_explanation", "evidence_grounding",
    "recommended_next_step", "limitations", "citations", "safety_status",
]


def run_gates() -> dict:
    gates = {}
    res = path("results_dir")

    # G5 local
    gates["G5_ollama_local"] = ping() and MODEL == "llama3"

    # G6 results exist
    gates["G6_results_logged"] = res.exists() and any(res.iterdir())

    # G1 final narratives schema
    fn = res / "final_narratives.jsonl"
    if fn.exists():
        lines = [json.loads(l) for l in fn.read_text(encoding="utf-8").splitlines() if l.strip()]
        gates["G1_n_cases"] = len(lines)
        gates["G1_valid_schema"] = all(
            all(f in n for f in REQUIRED_NARRATIVE_FIELDS) for n in lines
        ) and len(lines) >= 3
    else:
        gates["G1_valid_schema"] = False

    # G2 disclaimer rate
    if fn.exists():
        lines = [json.loads(l) for l in fn.read_text(encoding="utf-8").splitlines() if l.strip()]
        disc = sum(1 for n in lines if "not a clinical diagnosis" in json.dumps(n).lower())
        gates["G2_disclaimer_rate"] = disc / len(lines) if lines else 0
        gates["G2_pass"] = gates["G2_disclaimer_rate"] >= 0.9
    else:
        gates["G2_pass"] = False

    # G3 unsupported/contradicted claims
    cf = res / "claim_level_faithfulness.csv"
    if cf.exists():
        df = pd.read_csv(cf)
        bad = df[df["support_status"].isin(["unsupported", "contradicted"])]
        gates["G3_bad_claims"] = len(bad)
        gates["G3_pass"] = len(bad) == 0
    else:
        gates["G3_pass"] = False

    # G4 retrieval eval ran
    gates["G4_retrieval_eval"] = (res / "retrieval_eval_summary.csv").exists()

    gates["phase2_paths_ok"] = len(validate_phase2_paths()) == 0

    all_pass = (
        gates.get("G1_valid_schema")
        and gates.get("G2_pass")
        and gates.get("G3_pass")
        and gates.get("G4_retrieval_eval")
        and gates.get("G5_ollama_local")
        and gates.get("G6_results_logged")
        and gates.get("phase2_paths_ok")
    )
    gates["ALL_PASS"] = all_pass
    return gates


def main() -> int:
    gates = run_gates()
    out = path("results_dir") / "acceptance_gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gates, indent=2), encoding="utf-8")
    for k, v in gates.items():
        print(f"{k}: {v}")
    return 0 if gates.get("ALL_PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
