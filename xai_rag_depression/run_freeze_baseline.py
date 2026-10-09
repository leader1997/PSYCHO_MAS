#!/usr/bin/env python3
"""Freeze current pipeline outputs as Baseline v1 (FAST mode)."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
VERSIONS = OUTPUT / "versions"
V1_DIR = VERSIONS / "v1_fast"

KEY_ARTIFACTS = [
    "model_performance.csv",
    "global_importance_per_model.csv",
    "aggregated_importance.csv",
    "stability_scores.csv",
    "faithfulness.csv",
    "local_explanations.json",
    "rag_ablation_evaluation.csv",
    "rag_evaluation.csv",
    "llm_example_answers.json",
    "xai_method_per_model.csv",
    "lr_and_perm_extras.json",
]

KEY_DIRS = ["cards", "kb"]


def main() -> int:
    V1_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []

    for name in KEY_ARTIFACTS:
        src = OUTPUT / name
        if src.exists():
            shutil.copy2(src, V1_DIR / name)
            copied.append(name)
        else:
            missing.append(name)

    for dname in KEY_DIRS:
        src = OUTPUT / dname
        if src.exists():
            dst = V1_DIR / dname
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied.append(f"{dname}/")

    registry_path = VERSIONS / "VERSION_REGISTRY.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"versions": []}

    entry = {
        "version": "v1",
        "label": "Baseline v1 — FAST mode",
        "description": "Current FAST-mode pipeline (60 trees, 4 DL epochs, 300 SHAP rows)",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "env": {"XAI_FAST": "1"},
        "snapshot_dir": str(V1_DIR.relative_to(ROOT)),
        "artifacts_copied": copied,
        "artifacts_missing": missing,
    }
    registry["versions"] = [v for v in registry["versions"] if v.get("version") != "v1"]
    registry["versions"].append(entry)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # Summary table for paper
    rows = []
    for v in registry["versions"]:
        rows.append({
            "Version": v["version"],
            "Description": v.get("label", v.get("description", "")),
            "Frozen": v.get("frozen_at", "")[:10],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(VERSIONS / "version_roadmap.csv", index=False)

    print(f"Frozen Baseline v1 -> {V1_DIR}")
    print(f"Copied {len(copied)} items; missing: {missing or 'none'}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
