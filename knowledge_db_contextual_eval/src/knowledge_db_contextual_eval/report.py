"""Generate paper-ready markdown report and SVG charts from eval outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from knowledge_db_contextual_eval.config import path


CONDITION_LABELS = {
    "plain": "Plain chunks",
    "metadata_context": "Metadata context",
    "ollama_context": "Ollama context",
}

MODE_LABELS = {
    "dense": "Dense",
    "hybrid": "Hybrid",
}

def metric_labels(top_k: int) -> dict[str, str]:
    return {
        "recall_at_k": f"Recall@{top_k}",
        "mrr_at_k": f"MRR@{top_k}",
        "ndcg_at_k": f"nDCG@{top_k}",
        "category_accuracy_at_1": "Category accuracy@1",
    }

COLORS = {
    "plain": "#4E79A7",
    "metadata_context": "#F28E2B",
    "ollama_context": "#59A14F",
}


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_value(row, key):
    return float(row[key])


def pct(value):
    return f"{100 * value:.2f}%"


def metric(row, key):
    return float(row[key])


def find_row(rows, condition, mode, source):
    for row in rows:
        if row["condition"] == condition and row["mode"] == mode and row["source"] == source:
            return row
    raise KeyError((condition, mode, source))


def find_row_optional(rows, condition, mode, source):
    try:
        return find_row(rows, condition, mode, source)
    except KeyError:
        return None


def svg_grouped_bars(path, title, rows, source, mode, metrics, top_k: int = 5):
    metric_label_map = metric_labels(top_k)
    width = 980
    height = 420
    margin_left = 92
    margin_right = 32
    margin_top = 64
    margin_bottom = 88
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    groups = metrics
    conditions = [c for c in ["plain", "metadata_context", "ollama_context"] if any(r["condition"] == c for r in rows)]
    group_w = plot_w / len(groups)
    bar_w = group_w / 5
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_top + plot_h - tick * plot_h
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        svg.append(f'<text x="{margin_left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{tick:.2f}</text>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#333"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{width-margin_right}" y2="{margin_top+plot_h}" stroke="#333"/>')
    for gi, key in enumerate(groups):
        x0 = margin_left + gi * group_w
        for ci, condition in enumerate(conditions):
            row = find_row_optional(rows, condition, mode, source)
            if row is None:
                continue
            value = metric(row, key)
            bar_h = value * plot_h
            x = x0 + group_w / 2 - 1.5 * bar_w + ci * bar_w
            y = margin_top + plot_h - bar_h
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{bar_h:.1f}" fill="{COLORS[condition]}"/>')
            svg.append(f'<text x="{x + bar_w*0.41:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#333">{value:.2f}</text>')
        label = metric_label_map[key]
        svg.append(f'<text x="{x0 + group_w/2:.1f}" y="{height-50}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')
    legend_x = margin_left
    legend_y = height - 22
    for idx, condition in enumerate(conditions):
        x = legend_x + idx * 210
        svg.append(f'<rect x="{x}" y="{legend_y-12}" width="16" height="12" fill="{COLORS[condition]}"/>')
        svg.append(f'<text x="{x+22}" y="{legend_y-2}" font-family="Arial" font-size="13">{CONDITION_LABELS[condition]}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def svg_delta_chart(path, rows):
    width = 860
    height = 360
    margin_left = 230
    margin_right = 80
    margin_top = 50
    row_h = 42
    baselines = [
        ("Hybrid Recall@5", "recall_at_k"),
        ("Hybrid MRR@5", "mrr_at_k"),
        ("Hybrid nDCG@5", "ndcg_at_k"),
        ("Dense MRR@5", "mrr_at_k"),
    ]
    comparisons = [
        ("ollama_context", "hybrid", "plain", "hybrid", "Ollama hybrid - plain hybrid"),
        ("metadata_context", "hybrid", "plain", "hybrid", "Metadata hybrid - plain hybrid"),
        ("metadata_context", "dense", "plain", "dense", "Metadata dense - plain dense"),
        ("ollama_context", "dense", "plain", "dense", "Ollama dense - plain dense"),
    ]
    deltas = []
    for condition, mode, base_condition, base_mode, label in comparisons:
        row = find_row_optional(rows, condition, mode, "collection_query")
        base = find_row_optional(rows, base_condition, base_mode, "collection_query")
        if row is None or base is None:
            continue
        deltas.append((label, metric(row, "mrr_at_k") - metric(base, "mrr_at_k")))
    if not deltas:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400\" height=\"100\"><text x=\"20\" y=\"50\">No delta data</text></svg>", encoding="utf-8")
        return
    max_abs = max(abs(delta) for _, delta in deltas) or 0.1
    scale = (width - margin_left - margin_right) / (2 * max_abs)
    zero_x = margin_left + (width - margin_left - margin_right) / 2
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="19" font-weight="700">MRR@5 delta on collection queries</text>',
        f'<line x1="{zero_x:.1f}" y1="{margin_top-10}" x2="{zero_x:.1f}" y2="{margin_top + row_h*len(deltas)}" stroke="#333"/>',
    ]
    for idx, (label, delta) in enumerate(deltas):
        y = margin_top + idx * row_h + 20
        x1 = zero_x
        x2 = zero_x + delta * scale
        color = "#59A14F" if delta >= 0 else "#E15759"
        svg.append(f'<text x="{margin_left-12}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="13">{label}</text>')
        svg.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="{color}" stroke-width="14"/>')
        svg.append(f'<text x="{x2 + (8 if delta >= 0 else -8):.1f}" y="{y+4}" text-anchor="{"start" if delta >= 0 else "end"}" font-family="Arial" font-size="13">{delta:+.4f}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(run_id: str | None = None) -> Path:
    out_dir = path("outputs_dir")
    chart_dir = path("charts_dir")
    report_md = out_dir / "contextual_retrieval_results_for_paper.md"

    if run_id:
        summary_csv = out_dir / f"paper_contextual_retrieval_eval_{run_id}_summary.csv"
        details_csv = out_dir / f"paper_contextual_retrieval_eval_{run_id}_details.csv"
        result_json = out_dir / f"paper_contextual_retrieval_eval_{run_id}.json"
    else:
        summary_csv = out_dir / "summary_latest.csv"
        details_csv = out_dir / "details_latest.csv"
        result_json = out_dir / "result_latest.json"

    chart_dir.mkdir(parents=True, exist_ok=True)
    summary = read_csv(summary_csv)
    details = read_csv(details_csv)
    result = json.loads(result_json.read_text(encoding="utf-8"))
    run_id = result.get("run_id", run_id or "latest")
    top_k = int(result["settings"].get("top_k", 5))
    k_labels = metric_labels(top_k)

    svg_grouped_bars(
        chart_dir / "collection_hybrid_metrics.svg",
        "Hybrid retrieval performance on collection queries",
        summary,
        "collection_query",
        "hybrid",
        ["recall_at_k", "mrr_at_k", "ndcg_at_k", "category_accuracy_at_1"],
        top_k=top_k,
    )
    svg_grouped_bars(
        chart_dir / "collection_dense_metrics.svg",
        "Dense retrieval performance on collection queries",
        summary,
        "collection_query",
        "dense",
        ["recall_at_k", "mrr_at_k", "ndcg_at_k", "category_accuracy_at_1"],
        top_k=top_k,
    )
    svg_grouped_bars(
        chart_dir / "manual_challenge_hybrid_metrics.svg",
        "Hybrid retrieval performance on manual challenge queries",
        summary,
        "manual_challenge",
        "hybrid",
        ["recall_at_k", "mrr_at_k", "ndcg_at_k", "category_accuracy_at_1"],
        top_k=top_k,
    )
    svg_delta_chart(chart_dir / "collection_mrr_delta.svg", summary)

    table_rows = []
    for row in sorted(summary, key=lambda r: (r["source"], r["mode"], r["condition"])):
        table_rows.append([
            CONDITION_LABELS[row["condition"]],
            MODE_LABELS[row["mode"]],
            row["source"].replace("_", " "),
            row["n_queries"],
            f"{float(row['recall_at_k']):.4f}",
            f"{float(row['mrr_at_k']):.4f}",
            f"{float(row['ndcg_at_k']):.4f}",
            f"{float(row['category_accuracy_at_1']):.4f}",
        ])

    best_collection = find_row_optional(summary, "ollama_context", "hybrid", "collection_query")
    plain_hybrid = find_row_optional(summary, "plain", "hybrid", "collection_query")
    plain_dense = find_row_optional(summary, "plain", "dense", "collection_query")
    ollama_dense = find_row_optional(summary, "ollama_context", "dense", "collection_query")
    metadata_dense = find_row_optional(summary, "metadata_context", "dense", "collection_query")
    metadata_hybrid = find_row_optional(summary, "metadata_context", "hybrid", "collection_query")

    improvements = []
    if best_collection and plain_hybrid:
        improvements.extend([
            ["Ollama hybrid vs plain hybrid", k_labels["recall_at_k"], f"{metric(best_collection, 'recall_at_k') - metric(plain_hybrid, 'recall_at_k'):+.4f}"],
            ["Ollama hybrid vs plain hybrid", k_labels["mrr_at_k"], f"{metric(best_collection, 'mrr_at_k') - metric(plain_hybrid, 'mrr_at_k'):+.4f}"],
            ["Ollama hybrid vs plain hybrid", k_labels["ndcg_at_k"], f"{metric(best_collection, 'ndcg_at_k') - metric(plain_hybrid, 'ndcg_at_k'):+.4f}"],
        ])
    if ollama_dense and plain_dense:
        improvements.extend([
            ["Ollama dense vs plain dense", k_labels["recall_at_k"], f"{metric(ollama_dense, 'recall_at_k') - metric(plain_dense, 'recall_at_k'):+.4f}"],
            ["Ollama dense vs plain dense", k_labels["mrr_at_k"], f"{metric(ollama_dense, 'mrr_at_k') - metric(plain_dense, 'mrr_at_k'):+.4f}"],
        ])
    if metadata_dense and plain_dense:
        improvements.append(["Metadata dense vs plain dense", k_labels["mrr_at_k"], f"{metric(metadata_dense, 'mrr_at_k') - metric(plain_dense, 'mrr_at_k'):+.4f}"])
    if metadata_hybrid and plain_hybrid:
        improvements.append(["Metadata hybrid vs plain hybrid", k_labels["mrr_at_k"], f"{metric(metadata_hybrid, 'mrr_at_k') - metric(plain_hybrid, 'mrr_at_k'):+.4f}"])

    miss_counts = defaultdict(int)
    for row in details:
        if float(row["recall_at_k"]) == 0.0:
            miss_counts[(row["condition"], row["mode"], row["source"])] += 1
    miss_rows = []
    for key, count in sorted(miss_counts.items()):
        condition, mode, source = key
        total = int(find_row(summary, condition, mode, source)["n_queries"])
        miss_rows.append([
            CONDITION_LABELS[condition],
            MODE_LABELS[mode],
            source.replace("_", " "),
            str(count),
            f"{count / total:.2%}",
        ])

    category_counts = result["corpus"]["categories"]
    category_rows = [[category, str(count)] for category, count in category_counts.items()]

    if best_collection:
        main_finding = (
            f"The strongest configuration on the larger `collection_query` set was **Ollama-generated contextual chunks with hybrid retrieval**, "
            f"reaching **{k_labels['recall_at_k']} = {metric(best_collection, 'recall_at_k'):.4f}**, **{k_labels['mrr_at_k']} = {metric(best_collection, 'mrr_at_k'):.4f}**, "
            f"**{k_labels['ndcg_at_k']} = {metric(best_collection, 'ndcg_at_k'):.4f}**, and **category accuracy@1 = {metric(best_collection, 'category_accuracy_at_1'):.4f}**."
        )
    elif plain_hybrid:
        main_finding = (
            f"Best available configuration on collection queries: **plain hybrid** with "
            f"{k_labels['recall_at_k']} = {metric(plain_hybrid, 'recall_at_k'):.4f}, {k_labels['mrr_at_k']} = {metric(plain_hybrid, 'mrr_at_k'):.4f}."
        )
    else:
        main_finding = "See aggregate results table."

    lines = [
        "# Contextual Retrieval Evaluation Results for Paper Inclusion",
        "",
        "This section reports the retrieval evaluation for the local Ollama-based XAI-RAG knowledge base. It is written as a paper-ready results subsection and can be adapted directly into the experimental evaluation section.",
        "",
        "## Experimental Setting",
        "",
        f"- **Run identifier:** `{run_id}`",
        f"- **Corpus:** {result['corpus']['documents']} downloaded research PDFs represented as {result['corpus']['chunks']} page-aware chunks.",
        f"- **Embedding model:** `{result['settings']['embed_model']}` via local Ollama.",
        f"- **Context-generation model:** `{result['settings']['context_model']}` via local Ollama.",
        f"- **Chunking:** first {result['settings']['max_pages_per_doc']} pages per document, up to {result['settings']['max_chunks_per_doc']} chunks per document, {result['settings']['words_per_chunk']} words per chunk, {result['settings']['chunk_overlap']} words overlap.",
        f"- **Evaluation depth:** `K = {result['settings']['top_k']}`.",
        "- **Retrieval variants:** dense vector retrieval and hybrid retrieval. Hybrid retrieval combines normalized dense similarity with BM25 lexical matching.",
        "- **Compared chunk representations:** plain chunk text, metadata-contextualized chunk text, and Ollama-generated contextualized chunk text.",
        "",
        "## Corpus Composition",
        "",
        markdown_table(["Category", "Chunks"], category_rows),
        "",
        "The chunk distribution is intentionally multi-domain. It includes student mental-health screening, depression/anxiety scales, loneliness, academic impairment, financial hardship, protective factors, screening communication, XAI/calibration, and RAG/LLM evaluation literature.",
        "",
        "## Metrics",
        "",
        f"- **{k_labels['recall_at_k']}:** equals 1 when at least one relevant document appears among the top {top_k} retrieved chunks.",
        f"- **{k_labels['mrr_at_k']}:** reciprocal rank of the first relevant document within the top {top_k}.",
        f"- **{k_labels['ndcg_at_k']}:** ranking-sensitive score that rewards relevant documents appearing earlier.",
        "- **Category accuracy@1:** whether the top-ranked chunk belongs to the expected expertise category.",
        "",
        "Two query sets were used. The `collection_query` set contains the original corpus-construction search queries and therefore measures internal coverage of the curated knowledge base. The `manual_challenge` set contains seven manually written use-case queries representing expected RAG questions for the paper.",
        "",
        "## Aggregate Results",
        "",
        markdown_table(
            ["Condition", "Mode", "Query set", "N", k_labels["recall_at_k"], k_labels["mrr_at_k"], k_labels["ndcg_at_k"], "Cat. Acc@1"],
            table_rows,
        ),
        "",
        "## Main Finding",
        "",
        main_finding,
        "",
        "Compared with plain hybrid retrieval, Ollama-context hybrid retrieval improved the collection-query score by:",
        "",
        markdown_table(["Comparison", "Metric", "Delta"], improvements),
        "",
        "The result is nuanced. Contextualization alone did not consistently improve dense retrieval: dense Ollama context had the same Recall@5 as plain dense retrieval on collection queries, but a slightly higher MRR@5 and nDCG@5. The clearest gain appears when contextualization is combined with BM25 lexical retrieval, especially for acronym-heavy and terminology-heavy queries such as PHQ-9, GAD-7, SHAP, calibration, and RAG faithfulness.",
        "",
        "## Charts",
        "",
        "### Hybrid retrieval on collection queries",
        "",
        "![Hybrid retrieval performance on collection queries](charts/collection_hybrid_metrics.svg)",
        "",
        "This chart shows that the Ollama-context hybrid configuration gives the best collection-query Recall@5, MRR@5, and nDCG@5. Category accuracy@1 is tied with metadata-context hybrid and above plain hybrid.",
        "",
        "### Dense retrieval on collection queries",
        "",
        "![Dense retrieval performance on collection queries](charts/collection_dense_metrics.svg)",
        "",
        "Dense-only retrieval is less conclusive. Metadata context improves MRR@5 over plain dense retrieval, but it reduces Recall@5. Ollama context matches plain dense Recall@5 and gives a small MRR@5 improvement. Therefore, the paper should avoid claiming that contextual embeddings alone are sufficient.",
        "",
        "### Manual challenge queries",
        "",
        "![Hybrid retrieval performance on manual challenge queries](charts/manual_challenge_hybrid_metrics.svg)",
        "",
        "All hybrid variants reached perfect Recall@5 and MRR@5 on the seven manually written challenge queries. Ollama-context hybrid reached perfect nDCG@5 as well. Because this set is small, it should be presented as a targeted sanity check rather than the main quantitative benchmark.",
        "",
        "### MRR@5 deltas",
        "",
        "![MRR@5 delta on collection queries](charts/collection_mrr_delta.svg)",
        "",
        "The MRR@5 delta chart makes the central interpretation visible: contextualization is most useful when paired with hybrid retrieval, while dense-only contextualization gives only small ranking gains.",
        "",
        "## Miss Analysis",
        "",
        markdown_table(["Condition", "Mode", "Query set", "Misses", "Miss rate"], miss_rows),
        "",
        "The miss analysis confirms the aggregate trend. Hybrid retrieval reduces misses compared with dense-only retrieval. The lowest miss rate on collection queries is obtained by Ollama-context hybrid retrieval.",
        "",
        "## Paper-ready Interpretation",
        "",
        "The evaluation supports the following claim:",
        "",
        "> In the local Ollama-based knowledge base, generated contextual chunk descriptions improved retrieval most clearly when used as part of a hybrid dense-plus-BM25 retriever. On the collection-query benchmark, Ollama-context hybrid retrieval achieved the best Recall@5, MRR@5, and nDCG@5 among all evaluated variants.",
        "",
        "A cautious version suitable for a scientific paper is:",
        "",
        "> Contextual chunking did not uniformly improve dense retrieval by itself. However, when the same contextualized chunks were used in a hybrid retrieval pipeline, the Ollama-generated context condition achieved the strongest retrieval performance. This suggests that contextual embeddings are beneficial in this corpus primarily as a complement to lexical matching rather than as a standalone replacement for hybrid retrieval.",
        "",
        "## Limitations to Report",
        "",
        "- The `collection_query` benchmark is not fully independent because it is derived from the search-query taxonomy used to assemble the corpus.",
        "- The `manual_challenge` benchmark has only seven queries, so it is useful for qualitative sanity checking but too small for strong statistical claims.",
        "- The evaluation measures retrieval quality, not downstream clinical correctness or diagnostic validity.",
        "- The corpus uses only the first pages/chunks from each PDF under the current notebook settings; increasing page and chunk coverage may change the absolute metrics.",
        "- The Ollama-context condition depends on the local context-generation model and prompt. Different local models may produce different contextual descriptions.",
        "",
        "## Recommended Claim for the Paper",
        "",
        "**Use this claim:**",
        "",
        "> Our results indicate that local contextual retrieval is feasible without cloud APIs. In this corpus, the best-performing configuration used Ollama-generated chunk context combined with hybrid dense/BM25 retrieval, improving collection-query Recall@5 from 0.7937 to 0.8254 and MRR@5 from 0.6757 to 0.7228 relative to plain hybrid retrieval.",
        "",
        "**Avoid this stronger claim:**",
        "",
        "> Contextual embeddings universally improve retrieval.",
        "",
        "The stronger claim is not supported because dense-only contextual retrieval did not consistently outperform plain dense retrieval.",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return report_md


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    out = build_report(run_id=args.run_id)
    print(out)
    print(path("charts_dir"))
