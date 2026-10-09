"""Export expert review template from final narratives."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from local_agentic_xai_rag.config_loader import ensure_dirs, path


def export_template() -> Path:
    ensure_dirs()
    narr_path = path("results_dir") / "final_narratives.jsonl"
    if not narr_path.exists():
        raise FileNotFoundError(f"Run pipeline first: {narr_path}")

    rows = []
    for line in narr_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n = json.loads(line)
        rows.append(
            {
                "case_id": n.get("case_id"),
                "screening_summary": n.get("screening_summary"),
                "recommended_next_step": n.get("recommended_next_step"),
                "limitations": n.get("limitations"),
                "clinical_appropriateness_1_5": "",
                "non_diagnostic_language_1_5": "",
                "faithfulness_to_evidence_1_5": "",
                "usefulness_of_next_step_1_5": "",
                "overall_acceptability_1_5": "",
                "comments": "",
            }
        )
    out = path("paper_outputs") / "expert_review_template.xlsx"
    pd.DataFrame(rows).to_excel(out, index=False)
    print(f"Wrote {out}")
    return out


def main() -> int:
    export_template()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
