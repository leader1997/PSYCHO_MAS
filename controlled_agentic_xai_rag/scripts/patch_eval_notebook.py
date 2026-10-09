"""Patch 08_scientific_evaluation.ipynb fixes."""

from __future__ import annotations

import json
from pathlib import Path


def get_src(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def set_src(cell: dict, text: str) -> None:
    lines = text.splitlines(keepends=True)
    cell["source"] = lines if lines else [text]


ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks" / "08_scientific_evaluation.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# Setup cell
setup = get_src(nb["cells"][2])
extra = (
    "import os\n"
    "os.chdir(ROOT)\n"
    "\n"
    'METRIC_COLS = ["precision@5", "recall@5", "recall@10", "hit@5", "hit@10", "mrr", "ndcg@10"]\n'
    "retrieval_detail = pd.DataFrame()\n"
    "retrieval_summary_flat = pd.DataFrame()\n"
    "qp_df = pd.DataFrame()\n"
    "report_df = pd.DataFrame()\n"
    "bench = pd.DataFrame()\n"
    "pred_metrics = {}\n"
)
if "os.chdir(ROOT)" not in setup:
    setup = setup.replace("load_config.cache_clear()\n", extra + "load_config.cache_clear()\n")
set_src(nb["cells"][2], setup)

# Prediction cell
pred = get_src(nb["cells"][8])
pred = pred.replace(
    'display(pred_df.round(4) if "auc" in pred_metrics else pred_df)\n',
    "num_cols = {k: round(float(v), 4) for k, v in pred_metrics.items() if isinstance(v, (int, float, np.floating))}\n"
    "display(pd.DataFrame([num_cols]) if num_cols else pred_df)\n",
)
set_src(nb["cells"][8], pred)

# Retrieval cells — use shared gold resolution
src10 = get_src(nb["cells"][10])
if "resolve_gold_chunks" not in src10:
    import_block = (
        "from controlled_agentic_xai_rag.evaluation.benchmark_gold import resolve_gold_chunks, load_corpus_metadata\n"
        "from controlled_agentic_xai_rag.evaluation.eval_retrieval import eval_retrieval_ablation\n\n"
    )
    src10 = src10.replace(
        "from controlled_agentic_xai_rag.evaluation.metrics import",
        import_block + "from controlled_agentic_xai_rag.evaluation.metrics import",
    )
    # Drop inline parse_gold_ids / eval_retriever_on_benchmark if present
    start = src10.find("def parse_gold_ids")
    end = src10.find("\n\nbench_path")
    if start != -1 and end != -1:
        src10 = src10[:start] + src10[end + 2 :]
    src10 = src10.replace(
        "if RUN_RETRIEVAL_EVAL:\n",
        "if RUN_RETRIEVAL_EVAL:\n"
        "    meta = load_corpus_metadata()\n"
        "    def eval_retriever_on_benchmark(retriever, bench: pd.DataFrame, chunks_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:\n"
        "        rows = []\n"
        "        for _, row in bench.iterrows():\n"
        "            q = str(row.get(\"question\", \"\")).strip()\n"
        "            relevant = resolve_gold_chunks(row, chunks_df, meta)\n"
        "            if not q or not relevant:\n"
        "                continue\n"
        "            hits = retriever.retrieve(q, k=k)\n"
        "            retrieved = [h.chunk_id for h in hits]\n"
        "            rows.append({\n"
        "                \"question_id\": row.get(\"question_id\", \"\"),\n"
        "                \"category\": row.get(\"category\", \"\"),\n"
        "                \"difficulty\": row.get(\"difficulty\", \"\"),\n"
        "                \"answer_type\": row.get(\"answer_type_master\", row.get(\"answer_type\", \"\")),\n"
        "                \"retriever\": retriever.name,\n"
        "                \"n_gold\": len(relevant),\n"
        "                \"precision@5\": precision_at_k(relevant, retrieved, 5),\n"
        "                \"recall@5\": recall_at_k(relevant, retrieved, 5),\n"
        "                \"recall@10\": recall_at_k(relevant, retrieved, 10),\n"
        "                \"hit@5\": float(any(c in relevant for c in retrieved[:5])),\n"
        "                \"hit@10\": float(any(c in relevant for c in retrieved[:10])),\n"
        "                \"mrr\": mrr(relevant, retrieved),\n"
        "                \"ndcg@10\": ndcg_at_k(relevant, retrieved, 10),\n"
        "            })\n"
        "        return pd.DataFrame(rows)\n\n",
    )
    src10 = src10.replace("corpus_ids = set(chunks[\"chunk_id\"].astype(str))\n", "")
    src10 = src10.replace("eval_retriever_on_benchmark(r, bench, corpus_ids, k=K_RETRIEVAL)", "eval_retriever_on_benchmark(r, bench, chunks, k=K_RETRIEVAL)")
src10 = src10.replace(
    '    metric_cols = ["precision@5", "recall@5", "recall@10", "hit@5", "hit@10", "mrr", "ndcg@10"]\n',
    "",
)
src10 = src10.replace("metric_cols", "METRIC_COLS")
if "Resolved 0/" not in src10:
    src10 = src10.replace(
        "if not retrieval_detail.empty:",
        "n_resolved = len(retrieval_detail)\n"
        "print(f\"Retrieval benchmark: {n_resolved} questions with resolvable gold (target {RETRIEVAL_LIMIT}).\")\n"
        "if not retrieval_detail.empty:",
        1,
    )
set_src(nb["cells"][10], src10)

src11 = get_src(nb["cells"][11]).replace("metric_cols", "METRIC_COLS")
set_src(nb["cells"][11], src11)

# Query planning
qp_src = get_src(nb["cells"][13])
if "def load_case_card" not in qp_src:
    helper = (
        "\n\n"
        "def load_case_card(case_id: str) -> dict:\n"
        "    try:\n"
        "        return get_xai_card(case_id)\n"
        "    except Exception:\n"
        "        fc_path = cfg_path(\"results_dir\") / \"formatted_cases\" / f\"{case_id}.json\"\n"
        "        if fc_path.exists():\n"
        "            fc = json.loads(fc_path.read_text(encoding=\"utf-8\"))\n"
        "            return {\n"
        "                \"case_id\": case_id,\n"
        "                \"prediction_probability\": fc.get(\"prediction_probability\", 0.5),\n"
        "                \"top_positive_factors\": fc.get(\"top_xai_contributors\", []),\n"
        "                \"top_protective_factors\": fc.get(\"protective_factors\", []),\n"
        "            }\n"
        "        return {\"case_id\": case_id, \"top_positive_factors\": [], \"top_protective_factors\": []}\n"
    )
    qp_src = qp_src.replace(
        "from controlled_agentic_xai_rag.inputs.loader import get_xai_card\n",
        "from controlled_agentic_xai_rag.inputs.loader import get_xai_card\n" + helper,
    )
    qp_src = qp_src.replace("        card = get_xai_card(cid)", "        card = load_case_card(cid)")
set_src(nb["cells"][13], qp_src)

# Report quality — honest pass rates with n
rep_src = get_src(nb["cells"][15])
if "pass_rates_with_n" not in rep_src:
    rep_src = rep_src.replace(
        "from controlled_agentic_xai_rag.validators.safety_rules import check_safety_rules",
        "from controlled_agentic_xai_rag.evaluation.benchmark_gold import PRESET_CASE_IDS\n"
        "from controlled_agentic_xai_rag.evaluation.eval_report_batch import pass_rates_with_n\n"
        "from controlled_agentic_xai_rag.validators.safety_rules import check_safety_rules",
    )
    rep_src = rep_src.replace(
        "reports = latest_by_case(load_jsonl(finals_path))",
        "reports = {k: v for k, v in latest_by_case(load_jsonl(finals_path)).items() if k in PRESET_CASE_IDS}",
    )
    rep_src = rep_src.replace(
        "    pass_rates = report_df[pass_cols].mean().rename(\"pass_rate\").reset_index().rename(columns={\"index\": \"check\"})\n"
        "    pass_rates[\"pass_rate_pct\"] = (pass_rates[\"pass_rate\"] * 100).round(1)\n"
        "    display(pass_rates)\n"
        "    pass_rates.to_csv(OUT_DIR / \"table5b_report_pass_rates.csv\", index=False)\n",
        "    pass_rates = pass_rates_with_n(report_df)\n"
        "    if not pass_rates.empty:\n"
        "        pass_rates[\"pass_rate_pct\"] = (pass_rates[\"pass_rate\"] * 100).round(1)\n"
        "        display(pass_rates)\n"
        "        pass_rates.to_csv(OUT_DIR / \"table5b_report_pass_rates.csv\", index=False)\n"
        "    else:\n"
        "        print(f\"Report pass rates unavailable (n={len(report_df)}).\")\n",
    )
    rep_src = rep_src.replace(
        "    axes[0].bar(pass_rates[\"check\"], pass_rates[\"pass_rate_pct\"], color=\"#55A868\")\n",
        "    if not pass_rates.empty:\n"
        "        axes[0].bar(pass_rates[\"check\"], pass_rates[\"pass_rate_pct\"], color=\"#55A868\")\n",
    )
set_src(nb["cells"][15], rep_src)

# Claims cell — shared loader + header fix
set_src(
    nb["cells"][16],
    '''from controlled_agentic_xai_rag.evaluation.benchmark_gold import load_claims_csv, ensure_claims_csv_header

claims_path = cfg_path("results_dir") / "claim_level_faithfulness.csv"
ensure_claims_csv_header(claims_path)

if claims_path.exists():
    claims = load_claims_csv(claims_path)
    if claims.empty or "support_status" not in claims.columns:
        print("claim_level_faithfulness.csv has no usable rows.")
    else:
        faith_summary = claims.groupby("support_status").size().reset_index(name="count")
        display(faith_summary)
        faith_summary.to_csv(OUT_DIR / "table6_citation_faithfulness.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"fully_supported": "#55A868", "partially_supported": "#CCB974", "unsupported": "#C44E52", "contradicted": "#8B0000"}
        ax.bar(faith_summary["support_status"], faith_summary["count"],
               color=[colors.get(s, "#999999") for s in faith_summary["support_status"]])
        ax.set_title("Citation faithfulness (LLM-as-judge)")
        ax.set_ylabel("Claim count")
        ax.tick_params(axis="x", rotation=20)
        savefig("fig8_citation_faithfulness.png")
        plt.show()
else:
    print("No claim_level_faithfulness.csv — run citation checker on saved reports for this figure.")
''',
)

# Limitations / gaps cell (insert before master if missing)
limitations_md = (
    "---\n"
    "## 7. Evaluation limitations (honest reporting)\n"
    "\n"
    "Pass rates and aggregate scores must be read with sample size. Small N does not imply clinical validation."
)
limitations_code = '''from controlled_agentic_xai_rag.evaluation.eval_report_batch import evaluation_gaps

qp_n = len(list((cfg_path("results_dir") / "query_plans").glob("*.json")))
limitations = evaluation_gaps(report_df, len(retrieval_detail), qp_n)
limitations_df = pd.DataFrame(limitations)
display(limitations_df)
limitations_df.to_csv(OUT_DIR / "table_evaluation_limitations.csv", index=False)
print(f"Retrieval N={len(retrieval_detail)}, report N={len(report_df[report_df.get('found', False)])}, query plans={qp_n}")
'''
has_lim = any("evaluation_limitations" in get_src(c) for c in nb["cells"] if c["cell_type"] == "code")
if not has_lim:
    nb["cells"].insert(18, {"cell_type": "markdown", "metadata": {}, "source": [limitations_md]})
    nb["cells"].insert(19, {"cell_type": "code", "metadata": {}, "outputs": [], "source": limitations_code.splitlines(keepends=True)})

# Master table
master_idx = next(
    i for i, c in enumerate(nb["cells"])
    if c["cell_type"] == "code" and ("master.append" in get_src(c) or "master_table" in get_src(c))
)
master_src = get_src(nb["cells"][master_idx]).replace("metric_cols", "METRIC_COLS")
master_src = master_src.replace(
    "if not retrieval_detail.empty:\n",
    "if not retrieval_detail.empty and not retrieval_summary_flat.empty:\n",
    1,
)
if "n_cases" not in master_src:
    master_src = master_src.replace(
        'master.append({"component": "Report quality", "metric": "Safety pass rate", "value": round(report_df["safety_pass"].mean(), 3)})',
        'found_reports = report_df[report_df.get("found", False)]\n'
        '    if not found_reports.empty:\n'
        '        master.append({"component": "Report quality", "metric": "Safety pass rate", "value": round(found_reports["safety_pass"].mean(), 3), "n": len(found_reports)})',
    )
set_src(nb["cells"][master_idx], master_src)

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Patched {nb_path} ({nb_path.stat().st_size} bytes)")
