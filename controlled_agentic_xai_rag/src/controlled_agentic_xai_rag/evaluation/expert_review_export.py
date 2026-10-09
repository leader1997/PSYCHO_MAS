"""Expert review template export."""

from __future__ import annotations

import json

import pandas as pd

from controlled_agentic_xai_rag.config_loader import ensure_dirs, path


def export_template() -> None:
    ensure_dirs()
    finals = path("results_dir") / "final_narratives.jsonl"
    rows = []
    if finals.exists():
        for line in finals.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({
                    "case_id": r.get("case_id"),
                    "screening_summary": r.get("screening_summary", "")[:500],
                    "clarity_1_5": "",
                    "safety_1_5": "",
                    "usefulness_1_5": "",
                    "trustworthiness_1_5": "",
                    "appropriateness_1_5": "",
                    "reviewer_notes": "",
                })
    if not rows:
        rows = [{"case_id": "example", "screening_summary": "", "clarity_1_5": ""}]
    out = path("paper_outputs") / "expert_review_template.xlsx"
    pd.DataFrame(rows).to_excel(out, index=False)
    print(f"Wrote {out}")


def main() -> None:
    export_template()


if __name__ == "__main__":
    main()
