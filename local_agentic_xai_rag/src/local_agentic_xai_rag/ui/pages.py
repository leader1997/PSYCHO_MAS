"""Streamlit page renderers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from local_agentic_xai_rag.config_loader import cfg_get, ensure_dirs, path, validate_phase2_paths
from local_agentic_xai_rag.llm.ollama_client import MODEL, ping
from local_agentic_xai_rag.main import load_xai_cards, run_case, save_results
from local_agentic_xai_rag.ui.helpers import (
    config_summary,
    corpus_stats,
    figure_paths,
    load_gates,
    load_jsonl,
    load_results_csv,
)


DISCLAIMER = (
    "**Advisory screening only — not a clinical diagnosis.** "
    "All outputs require qualified human review before any action."
)


def render_sidebar() -> str:
    st.sidebar.title("Local XAI-RAG")
    st.sidebar.caption(f"Ollama `{MODEL}` only")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Run pipeline",
            "Case reports",
            "Retrieval explorer",
            "Evaluation dashboard",
            "Figures & paper",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    ok = ping()
    st.sidebar.metric("Ollama", "Online" if ok else "Offline")
    missing = validate_phase2_paths()
    st.sidebar.metric("Phase 2 data", "OK" if not missing else f"{len(missing)} missing")
    return page


def page_overview() -> None:
    st.title("Local Agentic XAI-RAG")
    st.markdown(DISCLAIMER)
    st.markdown(
        """
        Human-in-the-loop pipeline: **case + XAI card → query planning agent → hybrid retrieval
        → evidence agent → advisory report → safety rules + claim verification → expert review.**
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    stats = corpus_stats()
    c1.metric("Corpus chunks", f"{stats['chunks']:,}")
    c2.metric("Documents", stats["documents"])
    c3.metric("Cases", len(load_xai_cards()))
    c4.metric("Model", MODEL)

    st.subheader("Architecture")
    fig = path("figures_dir") / "fig01_architecture.png"
    if fig.exists():
        st.image(str(fig), use_container_width=True)
    else:
        st.code(
            "Case+XAI → Query Planning Agent → Hybrid Retrieval (BM25+llama3)\n"
            "→ Evidence Agent → Report Agent → Safety + Claims → Human Review",
            language=None,
        )

    st.subheader("Acceptance gates (G1–G6)")
    gates = load_gates()
    rows = [
        ("G1", "Valid final narratives (3 cases)", gates.get("G1_valid_schema")),
        ("G2", f"Disclaimer ≥ 90% ({gates.get('G2_disclaimer_rate', 0):.0%})", gates.get("G2_pass")),
        ("G3", f"No bad claims ({gates.get('G3_bad_claims', 0)})", gates.get("G3_pass")),
        ("G4", "Retrieval eval (100Q)", gates.get("G4_retrieval_eval")),
        ("G5", "Ollama llama3 local", gates.get("G5_ollama_local")),
        ("G6", "Results logged", gates.get("G6_results_logged")),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Gate", "Criterion", "Pass"]).assign(
            Pass=lambda d: d["Pass"].map({True: "✓", False: "✗", None: "?"})
        ),
        hide_index=True,
        use_container_width=True,
    )
    if gates.get("ALL_PASS"):
        st.success("All acceptance gates passed.")
    else:
        st.warning("Some gates not passed — run pipeline and evaluation from the sidebar pages.")

    with st.expander("Configuration"):
        st.json(config_summary())


def page_run_pipeline() -> None:
    st.title("Run pipeline")
    st.markdown(DISCLAIMER)

    missing = validate_phase2_paths()
    if missing:
        st.error("Missing Phase 2 paths:\n" + "\n".join(f"- {m}" for m in missing))
        return
    if not ping():
        st.error("Ollama is not reachable. Start Ollama and ensure `llama3` is pulled.")
        return

    cards = load_xai_cards()
    case_ids = [c["case_id"] for c in cards]

    col1, col2 = st.columns(2)
    with col1:
        selected = st.multiselect("Cases", case_ids, default=case_ids)
    with col2:
        bm25_only = st.checkbox("BM25 only (fast, skip FAISS build)", value=True)
        save_to_disk = st.checkbox("Save results to results/", value=True)
        use_query_planning = st.checkbox("Use XAI query planning agent", value=True)
        templates_only = st.checkbox("Template queries only (skip agent)", value=False)

    if st.button("Run selected cases", type="primary", disabled=not selected):
        ensure_dirs()
        progress = st.progress(0, text="Starting pipeline…")
        results = []
        for i, cid in enumerate(selected):
            progress.progress((i) / len(selected), text=f"Processing {cid}…")
            with st.spinner(f"Running {cid} (evidence + report + validation)…"):
                try:
                    result = run_case(
                        cid,
                        bm25_only=bm25_only,
                        use_query_planning=use_query_planning,
                        templates_only=templates_only,
                    )
                    if save_to_disk:
                        save_results(result)
                    results.append(result)
                except Exception as exc:
                    st.error(f"{cid} failed: {exc}")
        progress.progress(1.0, text="Done")
        st.session_state["last_pipeline_results"] = results
        st.success(f"Completed {len(results)} case(s).")

    if "last_pipeline_results" in st.session_state:
        for r in st.session_state["last_pipeline_results"]:
            with st.expander(f"{r['case_id']} — safety={r['safety']['overall_safety']}", expanded=True):
                if r.get("query_plan"):
                    st.markdown("**Query plan (agent)**")
                    st.json(r["query_plan"])
                st.markdown("**Screening summary**")
                st.write(r["report"].get("screening_summary"))
                st.markdown("**Recommended next step**")
                st.write(r["report"].get("recommended_next_step"))
                st.markdown("**Retrieved chunks**")
                st.dataframe(pd.DataFrame(r["hits"])[["chunk_id", "title", "score", "method"]], hide_index=True)
                if len(r["claims"]):
                    st.markdown("**Claim verification**")
                    st.dataframe(r["claims"], hide_index=True)


def page_case_reports() -> None:
    st.title("Case reports")
    st.markdown(DISCLAIMER)

    narratives = load_jsonl("final_narratives.jsonl")
    if not narratives:
        st.info("No saved reports yet. Run the pipeline first.")
        return

    by_id = {n["case_id"]: n for n in narratives}
    case_id = st.selectbox("Case", list(by_id.keys()))
    n = by_id[case_id]

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Query plan", "Report", "Evidence", "Safety", "Claims"])

    with tab0:
        plans = load_jsonl("query_plans.jsonl")
        plan = next((p for p in plans if p.get("case_id") == case_id), None)
        if plan:
            st.json(plan)
            st.markdown("**Internal questions**")
            st.dataframe(pd.DataFrame(plan.get("internal_questions") or []), hide_index=True)
            st.markdown("**Retrieval queries**")
            st.dataframe(pd.DataFrame(plan.get("retrieval_queries") or []), hide_index=True)
        else:
            st.info("No query plan saved — re-run pipeline with query planning agent enabled.")

    with tab1:
        st.subheader("Advisory report")
        for field in (
            "screening_summary",
            "model_explanation",
            "evidence_grounding",
            "recommended_next_step",
            "limitations",
            "safety_status",
        ):
            st.markdown(f"**{field.replace('_', ' ').title()}**")
            val = n.get(field, "")
            if isinstance(val, (dict, list)):
                st.json(val)
            else:
                st.write(val)
        cites = n.get("citations") or []
        if cites:
            st.markdown("**Citations**")
            st.dataframe(pd.DataFrame(cites), hide_index=True)

    with tab2:
        evidence_lines = load_jsonl("evidence_summaries.jsonl")
        ev = next((e for e in evidence_lines if e.get("case_id") == case_id), None)
        if ev:
            st.json(ev)
        else:
            st.info("No evidence summary saved for this case.")

    with tab3:
        val_df = load_results_csv("validation_reports.csv")
        if val_df is not None and case_id in val_df["case_id"].values:
            st.dataframe(val_df[val_df["case_id"] == case_id].T, use_container_width=True)
        else:
            st.info("No validation report for this case.")

    with tab4:
        claims = load_results_csv("claim_level_faithfulness.csv")
        if claims is not None and case_id in claims["case_id"].values:
            sub = claims[claims["case_id"] == case_id]
            st.dataframe(sub, hide_index=True, use_container_width=True)
            st.bar_chart(sub["support_status"].value_counts())
        else:
            st.info("No claim-level results for this case.")


def page_retrieval() -> None:
    st.title("Retrieval explorer")
    st.markdown("Query the shared Phase 2 corpus with BM25 or hybrid (BM25 + llama3 dense).")

    missing = validate_phase2_paths()
    if missing:
        st.error("Phase 2 corpus not available.")
        return

    bm25_only = st.checkbox("BM25 only", value=True, key="ret_bm25")
    query = st.text_area(
        "Query",
        value="How is loneliness related to depression among college students?",
        height=80,
    )
    top_k = st.slider("Top K", 3, 15, cfg_get("retrieval", "top_k", default=8))

    if st.button("Retrieve", type="primary"):
        from local_agentic_xai_rag.retrieval.hybrid import get_retriever

        with st.spinner("Retrieving…"):
            retriever = get_retriever(bm25_only=bm25_only)
            hits = retriever.retrieve(query, k=top_k)
        rows = [h.to_dict() for h in hits]
        st.dataframe(pd.DataFrame(rows)[["rank", "chunk_id", "title", "page", "score", "method"]], hide_index=True)
        for h in hits:
            with st.expander(f"[{h.chunk_id}] {h.title[:60]}…"):
                st.write(h.text[:2000])

    st.divider()
    st.subheader("Gold-chunk benchmark summary")
    summary = load_results_csv("retrieval_eval_summary.csv")
    if summary is not None:
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.bar_chart(summary.set_index("retriever")[["precision@5", "recall@5", "recall@10"]])
    else:
        st.caption("Run retrieval evaluation from the Evaluation dashboard.")


def page_evaluation() -> None:
    st.title("Evaluation dashboard")

    col1, col2 = st.columns(2)
    with col1:
        eval_limit = st.number_input("Retrieval eval limit (questions)", 10, 100, 100)
        bm25_hybrid_fast = st.checkbox("Fast hybrid slot (BM25)", value=True)
    with col2:
        bench_limit = st.number_input("System comparison benchmark subset", 0, 20, 20)
        bm25_cmp = st.checkbox("BM25 only for comparison", value=True)

    if st.button("Run retrieval eval"):
        from local_agentic_xai_rag.evaluation.retrieval_eval import run_retrieval_eval

        with st.spinner("Running retrieval eval…"):
            run_retrieval_eval(limit=int(eval_limit), bm25_only_hybrid=bm25_hybrid_fast)
        st.success("Saved retrieval_eval_summary.csv")
        st.rerun()

    if st.button("Run system comparison (B1/B2/B3)"):
        from local_agentic_xai_rag.evaluation.system_comparison import run_comparison

        st.warning("This may take 15–30+ minutes (many Ollama calls).")
        with st.spinner("Running system comparison…"):
            run_comparison(bm25_only=bm25_cmp, benchmark_limit=int(bench_limit))
        st.success("Saved system_comparison.csv")
        st.rerun()

    if st.button("Run acceptance gates"):
        from local_agentic_xai_rag.evaluation.acceptance import run_gates

        gates = run_gates()
        out = path("results_dir") / "acceptance_gates.json"
        out.write_text(json.dumps(gates, indent=2), encoding="utf-8")
        st.json(gates)

    st.divider()
    st.subheader("Retrieval metrics")
    summary = load_results_csv("retrieval_eval_summary.csv")
    if summary is not None:
        st.dataframe(summary, hide_index=True)
    else:
        st.caption("No retrieval eval results yet.")

    st.subheader("System comparison (cases)")
    cmp_df = load_results_csv("system_comparison.csv")
    if cmp_df is not None:
        src = cmp_df.get("source", pd.Series(["case"] * len(cmp_df)))
        cases = cmp_df[src == "case"] if "source" in cmp_df.columns else cmp_df
        st.dataframe(cases.groupby("system").mean(numeric_only=True).round(3))
    else:
        st.caption("No system comparison results yet.")

    st.subheader("Claim faithfulness (all cases)")
    claims = load_results_csv("claim_level_faithfulness.csv")
    if claims is not None:
        st.bar_chart(claims["support_status"].value_counts())
        st.dataframe(claims, hide_index=True, use_container_width=True)


def page_figures() -> None:
    st.title("Figures & paper outputs")
    figs = figure_paths()
    if figs:
        cols = st.columns(2)
        for i, fp in enumerate(figs):
            cols[i % 2].image(str(fp), caption=fp.name, use_container_width=True)
    else:
        st.info("No figures yet. Click **Build figures** below.")

    if st.button("Build all figures + Excel tables"):
        from local_agentic_xai_rag.visualization.build_all_figures import build_all_figures, build_tables_xlsx

        ensure_dirs()
        build_all_figures()
        build_tables_xlsx()
        st.success("Figures and tables updated.")
        st.rerun()

    xlsx = path("paper_outputs") / "tables_for_paper.xlsx"
    if xlsx.exists():
        st.subheader("Paper tables")
        xl = pd.ExcelFile(xlsx)
        sheet = st.selectbox("Sheet", xl.sheet_names)
        st.dataframe(pd.read_excel(xlsx, sheet_name=sheet), hide_index=True, use_container_width=True)

    for md_name in ("methods_section.md", "results_section.md", "limitations.md"):
        md_path = path("paper_outputs") / md_name
        if md_path.exists():
            with st.expander(md_name):
                st.markdown(md_path.read_text(encoding="utf-8"))
