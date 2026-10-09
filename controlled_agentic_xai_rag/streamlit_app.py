"""Streamlit dashboard."""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Allow running without pip install -e . (e.g. streamlit on a different Python)
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

st.set_page_config(page_title="Controlled XAI-RAG", layout="wide")
st.title("Controlled Agentic XAI-RAG")
st.caption("Non-clinical screening communication — not a diagnostic tool.")

results = ROOT / "results"

_DEPS_OK = True
_DEPS_ERROR = ""
try:
    from controlled_agentic_xai_rag.pipeline import PipelineOptions, run_case
    from controlled_agentic_xai_rag.llm.ollama_client import (
        embeddings_available,
        get_embed_model,
        get_model,
        list_embedding_models,
        list_installed_models,
        ping,
    )
except ImportError as exc:
    _DEPS_OK = False
    _DEPS_ERROR = str(exc)

if "last_pipeline_result" not in st.session_state:
    st.session_state.last_pipeline_result = None
if "last_run_options" not in st.session_state:
    st.session_state.last_run_options = None

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Run pipeline", "Pipeline trace", "Final report"]
)


def _feature_table(profile: dict) -> pd.DataFrame:
    """All HMS features with raw values and tertile levels."""
    try:
        from controlled_agentic_xai_rag.prediction.hms_predictor import FEATURE_DISPLAY, FEATURES
    except ImportError:
        FEATURES = list((profile.get("feature_levels") or {}).keys())
        FEATURE_DISPLAY = {}

    levels = profile.get("feature_levels") or {}
    raw = profile.get("raw_values") or {}
    rows = []
    for feature in FEATURES:
        rows.append(
            {
                "feature": feature,
                "display_name": FEATURE_DISPLAY.get(feature, feature.replace("_", " ").title()),
                "raw_value": raw.get(feature),
                "level": levels.get(feature, "—"),
            }
        )
    return pd.DataFrame(rows)


def _factors_dataframe(factors: list) -> pd.DataFrame:
    if not factors:
        return pd.DataFrame()
    return pd.DataFrame(factors)


@st.cache_data(show_spinner="Loading live HMS prediction…")
def _preview_live_inputs(case_id: str, hms_test_index: int | None) -> dict:
    from controlled_agentic_xai_rag.prediction.hms_predictor import resolve_live_case

    result = resolve_live_case(
        preset=case_id if hms_test_index is None else case_id,
        test_index=hms_test_index,
    )
    return {
        "source": "live_hms_prediction",
        "card": result.card,
        "profile": result.profile,
        "prediction": result.to_audit_dict(),
    }


def _preview_seeded_inputs(case_id: str) -> dict:
    from controlled_agentic_xai_rag.inputs.loader import get_case_profile, get_xai_card

    return {
        "source": "seeded_inputs",
        "card": get_xai_card(case_id),
        "profile": get_case_profile(case_id),
        "prediction": None,
    }


def _render_pipeline_inputs(
    card: dict,
    profile: dict,
    *,
    source: str,
    prediction: dict | None = None,
    run_options: dict | None = None,
    expanded: bool = True,
) -> None:
    """Show full pipeline inputs: profile features, XAI factors, and raw JSON."""
    with st.expander("Pipeline inputs — XAI card + case profile", expanded=expanded):
        st.markdown(f"**Input source:** `{source}`")

        if run_options:
            st.markdown("**Run configuration**")
            opt_cols = st.columns(3)
            opt_cols[0].write(f"**Model:** `{run_options.get('ollama_model', '—')}`")
            opt_cols[1].write(f"**Embed:** `{run_options.get('embed_model', '—')}`")
            opt_cols[2].write(f"**Retrieval:** hybrid (FAISS + BM25)")

        prob = profile.get("predicted_probability", card.get("prediction_probability"))
        mcols = st.columns(4)
        mcols[0].metric("Case ID", profile.get("case_id") or card.get("case_id", "—"))
        mcols[1].metric("Probability", f"{prob:.4f}" if prob is not None else "—")
        mcols[2].metric("Predicted class", profile.get("predicted_class", card.get("prediction_label", "—")))
        mcols[3].metric(
            "HMS test index",
            prediction.get("test_index") if prediction else "—",
        )

        if prediction:
            st.markdown("**Live prediction audit**")
            st.json(prediction)

        st.markdown("#### All survey features (25)")
        st.dataframe(_feature_table(profile), width="stretch", hide_index=True)

        fx1, fx2 = st.columns(2)
        with fx1:
            st.markdown("#### XAI — risk-increasing factors")
            pos = card.get("top_positive_factors") or []
            if pos:
                st.dataframe(_factors_dataframe(pos), width="stretch", hide_index=True)
            else:
                st.caption("_(none in top 5)_")
        with fx2:
            st.markdown("#### XAI — protective factors")
            prot = card.get("top_protective_factors") or []
            if prot:
                st.dataframe(_factors_dataframe(prot), width="stretch", hide_index=True)
            else:
                st.caption("_(none in top 5)_")

        warning = card.get("important_warning")
        if warning:
            st.warning(warning)

        st.markdown("#### Model metadata")
        meta_cols = st.columns(3)
        meta_cols[0].write(f"**Model:** {card.get('model_name', '—')}")
        meta_cols[1].write(f"**Label:** {card.get('prediction_label', '—')}")
        meta_cols[2].write(f"**Calibration:** {card.get('calibration_status', '—')}")

        json_cols = st.columns(2)
        with json_cols[0]:
            st.markdown("**Case profile (full JSON)**")
            st.json(profile)
        with json_cols[1]:
            st.markdown("**XAI card (full JSON)**")
            st.json(card)


def _json_block(data) -> None:
    if isinstance(data, pd.DataFrame):
        if data.empty:
            st.caption("_(empty)_")
        else:
            st.dataframe(data, width="stretch")
    else:
        st.json(data)


def _hits_dataframe(hits: list) -> pd.DataFrame:
    if not hits:
        return pd.DataFrame()
    df = pd.DataFrame(hits)
    preferred = ["chunk_id", "rank", "title", "page", "rrf_score", "query_id", "text"]
    cols = [c for c in preferred if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra].copy()
    if "text" in df.columns:
        df["text"] = df["text"].astype(str).str[:200] + "…"
    return df


def _display_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    text = str(value).strip()
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parts = []
                for k, v in parsed.items():
                    if isinstance(v, (dict, list)):
                        parts.append(f"**{k}**: {json.dumps(v)}")
                    else:
                        parts.append(f"**{k}**: {v}")
                return "\n\n".join(parts)
            if isinstance(parsed, list):
                return "\n".join(f"- {json.dumps(item) if isinstance(item, (dict, list)) else item}" for item in parsed)
        except json.JSONDecodeError:
            pass
    return text


def _before_after(before, after, before_label: str = "Before", after_label: str = "After") -> None:
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(f"**{before_label}**")
        _json_block(before)
    with col_a:
        st.markdown(f"**{after_label}**")
        _json_block(after)


def _render_pipeline_trace(result: dict) -> None:
    """Show every pipeline stage output with before/after where applicable."""
    st.markdown(f"**Case:** `{result.get('case_id')}` · **Model:** `{result.get('ollama_model', '—')}`")
    if result.get("prediction"):
        with st.expander("Step −1 — HMS Prediction (CatBoost + TreeSHAP)", expanded=True):
            st.caption("Real HMS test-row features → classification probability → SHAP XAI card")
            pred = result["prediction"]
            st.metric("Prediction probability", f"{pred.get('prediction_probability', 0):.4f}")
            st.metric("HMS test index", pred.get("test_index", "—"))
            st.json(pred)

    inputs = result.get("inputs") or {}
    if inputs.get("xai_card") and inputs.get("case_profile"):
        _render_pipeline_inputs(
            inputs["xai_card"],
            inputs["case_profile"],
            source=inputs.get("source", "unknown"),
            prediction=result.get("prediction"),
            run_options={
                "ollama_model": result.get("ollama_model"),
                **(st.session_state.last_run_options or {}),
            },
            expanded=True,
        )

    # ── Step 0: Case Formatter ──────────────────────────────────────────
    with st.expander("Step 0 — Case Formatter", expanded=True):
        st.caption("Input: de-identified case profile + XAI card → deterministic formatted JSON")
        _before_after(
            {
                "xai_card": result.get("inputs", {}).get("xai_card"),
                "case_profile": result.get("inputs", {}).get("case_profile"),
            },
            result.get("formatted_case"),
            before_label="Input (XAI card + case profile)",
            after_label="Output (formatted case)",
        )

    # ── Step 1: Query Planning ────────────────────────────────────────
    with st.expander("Step 1 — Query Planning Agent (LLM)", expanded=True):
        st.caption("XAI context → report questions + hybrid retrieval queries")
        plan = result.get("query_plan") or {}
        questions = plan.get("report_questions") or plan.get("internal_questions") or []
        if questions:
            st.markdown("**Report questions (backbone of the recommendation report)**")
            for i, q in enumerate(questions, start=1):
                st.markdown(f"{i}. {q}")
        st.markdown("**Full query plan**")
        _json_block(plan)

    # ── Step 2: Hybrid Retrieval ──────────────────────────────────────
    with st.expander("Step 2 — Hybrid Retrieval (FAISS + BM25)", expanded=True):
        st.caption("BM25 + dense RRF per query → deduplicated evidence pool")
        retrieval_info = result.get("retrieval_info") or {}
        if retrieval_info:
            st.markdown(
                f"**Requested:** `{retrieval_info.get('requested_mode', '—')}` · "
                f"**Actual:** `{retrieval_info.get('actual_mode', '—')}` · "
                f"**Vector store:** `{retrieval_info.get('vector_store', '—')}`"
            )
            if retrieval_info.get("warning"):
                st.warning(retrieval_info["warning"])
        audit_rows = result.get("retrieval_audit") or []
        hits = result.get("hits") or []
        st.metric("Queries executed", len({r.get("query_id") for r in audit_rows if r.get("query_id")}))
        st.metric("Audit rows (per query × chunk)", len(audit_rows))
        st.metric("Final unique hits", len(hits))

        if audit_rows:
            st.markdown("**Retrieval audit log**")
            st.dataframe(pd.DataFrame(audit_rows), width="stretch", height=280)
        if hits:
            st.markdown("**Final evidence pool (top hits)**")
            st.dataframe(_hits_dataframe(hits), width="stretch")

    # ── Step 3: Report Generation ─────────────────────────────────────
    with st.expander("Step 3 — Report Generation Agent", expanded=True):
        st.caption("Answers query-plan questions using case + XAI + retrieved evidence")
        st.markdown("**Generated report**")
        _json_block(result.get("generated_report"))

    # ── Step 4: Final narrative ───────────────────────────────────────
    with st.expander("Step 4 — Final narrative", expanded=True):
        st.caption("Unvalidated advisory output (no safety/citation gate)")
        if result.get("validators_skipped"):
            st.caption("Validation is disabled in this pipeline configuration.")
        _render_report(result.get("final_report") or {})


def _render_report(report: dict) -> None:
    """Human-readable final screening report."""
    if not report:
        st.warning("No final report available.")
        return
    cols = st.columns(3)
    cols[0].metric("Safety status", report.get("safety_status", "—"))
    cols[1].metric("Model", report.get("ollama_model", "—"))
    cols[2].metric("Citations", len(report.get("citations") or []))

    st.markdown("#### Screening summary")
    st.write(_display_text(report.get("screening_summary", "")))

    contributor = _display_text(report.get("contributor_analysis", ""))
    if contributor:
        st.markdown("#### Main contributors (for expert review)")
        st.write(contributor)

    st.markdown("#### Probability interpretation")
    st.write(_display_text(report.get("probability_interpretation", "")))

    st.markdown("#### XAI explanation")
    st.write(_display_text(report.get("xai_explanation", "")))

    st.markdown("#### Evidence-grounded interpretation")
    st.write(_display_text(report.get("evidence_grounded_interpretation", "")))

    actions = _display_text(report.get("actionable_recommendations", ""))
    if actions:
        st.markdown("#### Actionable recommendations (step-by-step)")
        st.success(actions)

    st.markdown("#### Priority summary for expert")
    rec = _display_text(report.get("non_clinical_recommendation", ""))
    st.info(rec if rec else "No recommendation generated.")

    protective = _display_text(report.get("protective_factors_note", ""))
    if protective:
        st.markdown("#### Protective factors")
        st.write(protective)

    st.markdown("#### Limitations")
    st.write(_display_text(report.get("limitations", "")))

    citations = report.get("citations") or []
    if citations:
        st.markdown("#### Citations")
        for c in citations:
            if isinstance(c, dict):
                st.markdown(f"- **{c.get('chunk_id', '')}**: {c.get('claim_text', '')}")

    with st.expander("Full JSON"):
        st.json(report)


def _load_final_reports() -> list[dict]:
    finals = results / "final_narratives.jsonl"
    if not finals.exists():
        return []
    reports = []
    for line in finals.read_text(encoding="utf-8").splitlines():
        if line.strip():
            reports.append(json.loads(line))
    return reports


with tab1:
    st.markdown("""
    **Pipeline (simplified)**

    | Step | Component | Output |
    |------|-----------|--------|
    | −1 | HMS Prediction | CatBoost probability + TreeSHAP XAI card |
    | 0 | Case Formatter | Structured case JSON |
    | 1 | Query Planning Agent (LLM) | Report questions + retrieval queries |
    | 2 | Hybrid Retrieval | FAISS + BM25 evidence pool |
    | 3 | Report Agent | Detailed advisory report answering planned questions |
    | 4 | Final narrative | Draft report (no validation gate) |

    Run a case on **Run pipeline**, then inspect every stage on **Pipeline trace**.
    """)

with tab2:
    if not _DEPS_OK:
        st.error(f"Missing dependencies: {_DEPS_ERROR}")
        st.code(f'py -3.11 -m pip install -e "{ROOT}[dev]"', language="powershell")

    if _DEPS_OK:
        from controlled_agentic_xai_rag.retrieval.index_sync import corpus_index_status

        corp = corpus_index_status()
        c1, c2, c3 = st.columns(3)
        c1.metric("Corpus chunks", corp.get("chunk_count", 0))
        c2.metric("FAISS index", "built" if corp.get("faiss_built") else "missing")
        c3.metric("Vector sync", "rebuild needed" if corp.get("indexes_need_rebuild") else "ok")
        if corp.get("indexes_need_rebuild"):
            st.info(
                "Text corpus includes all PDFs under `knowledge_db/pdfs/`, but the **FAISS index** "
                "is missing or out of date. Rebuild corpus after adding PDFs, then run the pipeline "
                "to embed all chunks via hybrid retrieval."
            )
        with st.expander("Corpus & vector index maintenance"):
            st.markdown(
                "1. Add PDFs to `knowledge_db/pdfs/` · "
                "2. **Rebuild corpus** · "
                "3. Run pipeline (hybrid retrieval builds the FAISS index on first run)."
            )
            bc1, bc2 = st.columns(2)
            if bc1.button("Rebuild corpus from PDFs"):
                import subprocess
                import sys as _sys

                with st.spinner("Running build_corpus.py…"):
                    r = subprocess.run(
                        [_sys.executable, str(ROOT / "scripts" / "build_corpus.py")],
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                    )
                if r.returncode == 0:
                    st.success(r.stdout.strip() or "Corpus rebuilt.")
                    st.rerun()
                else:
                    st.error(r.stderr or r.stdout or "build_corpus failed")
            if bc2.button("Anchor fingerprint (no re-chunk)"):
                import subprocess
                import sys as _sys

                r = subprocess.run(
                    [_sys.executable, str(ROOT / "scripts" / "sync_corpus.py"), "--anchor-only"],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                if r.returncode == 0:
                    st.success(r.stdout.strip() or "Fingerprint anchored.")
                    st.rerun()
                else:
                    st.error(r.stderr or r.stdout or "sync failed")

    use_live = True
    _PREDICTION_OK = True
    _PREDICTION_ERR = ""
    _n_test = 0
    _presets: dict = {}
    if _DEPS_OK:
        try:
            from controlled_agentic_xai_rag.prediction.hms_predictor import (
                get_preset_indices,
                get_test_set_size,
            )
            _n_test = get_test_set_size()
            _presets = get_preset_indices()
            st.caption(
                f"HMS held-out test rows: **{_n_test}** · Preset indices: "
                + ", ".join(f"`{k}`→{v}" for k, v in _presets.items())
            )
        except Exception as exc:
            _PREDICTION_OK = False
            _PREDICTION_ERR = str(exc)

    st.caption("Pipeline always runs **live HMS prediction** (CatBoost + TreeSHAP) before RAG.")

    hms_test_index = None
    case_id = "high_risk_case"

    if not _PREDICTION_OK:
        st.error(f"Live prediction unavailable: {_PREDICTION_ERR}")
    pick_mode = st.radio(
        "HMS row selection",
        ["Preset case (high / low / borderline)", "Custom test row index"],
        horizontal=True,
    )
    if pick_mode.startswith("Preset"):
        case_id = st.selectbox(
            "Preset (same split as Phase 1)",
            ["high_risk_case", "low_risk_case", "borderline_risk_case"],
        )
    else:
        hms_test_index = st.number_input(
            "HMS test row index",
            min_value=0,
            max_value=max(_n_test - 1, 0),
            value=0,
            step=1,
        )
        case_id = f"live_test_{int(hms_test_index)}"

    available_models = list_installed_models() if _DEPS_OK else ["llama3"]
    default_model = get_model() if _DEPS_OK else "llama3"
    default_idx = available_models.index(default_model) if default_model in available_models else 0
    ollama_model = st.selectbox(
        "Ollama model (query planning + report)",
        available_models,
        index=default_idx,
        help="Chat model for LLM query planning and report generation.",
    )

    embed_models = list_embedding_models() if _DEPS_OK else ["nomic-embed-text"]
    default_embed = get_embed_model() if _DEPS_OK else "nomic-embed-text"
    embed_idx = next(
        (i for i, m in enumerate(embed_models) if m == default_embed or m.split(":")[0] == default_embed.split(":")[0]),
        0,
    )
    embed_model = st.selectbox(
        "Embedding model (hybrid FAISS leg)",
        embed_models,
        index=embed_idx,
        help="Dedicated embedding model (e.g. nomic-embed-text). Chat models like llama3 cannot embed.",
    )

    embed_ok = False
    if _DEPS_OK:
        embed_ok = embeddings_available(embed_model.split(":")[0], force_check=True)
        ollama_status = "online" if ping() else "offline"
        embed_status = "available" if embed_ok else "unavailable"
        st.caption(
            f"Hybrid retrieval (FAISS + BM25): `{embed_model.split(':')[0]}` ({embed_status}) · "
            f"Ollama: {ollama_status}"
        )
        if not embed_ok:
            st.warning(
                "Embeddings unavailable — hybrid search will fall back to BM25 only until you run "
                "`ollama pull nomic-embed-text`."
            )
        elif embed_model.split(":")[0] != default_embed.split(":")[0]:
            st.caption("First hybrid run will rebuild the FAISS index for the selected embed model (may take several minutes).")

    st.divider()
    st.subheader("Inputs preview")
    preview_error = None
    preview_inputs = None
    if _DEPS_OK and _PREDICTION_OK:
        try:
            idx = int(hms_test_index) if hms_test_index is not None else None
            preview_inputs = _preview_live_inputs(case_id, idx)
        except Exception as exc:
            preview_error = str(exc)

    if preview_error:
        st.error(f"Could not load inputs: {preview_error}")
    elif preview_inputs:
        _render_pipeline_inputs(
            preview_inputs["card"],
            preview_inputs["profile"],
            source=preview_inputs["source"],
            prediction=preview_inputs.get("prediction"),
            run_options={
                "ollama_model": ollama_model,
                "embed_model": embed_model.split(":")[0],
            },
            expanded=True,
        )
    elif not _PREDICTION_OK:
        st.info("Fix live prediction errors above to preview inputs.")

    st.divider()

    if st.button("Run case", disabled=not _DEPS_OK or not _PREDICTION_OK):
        opts = PipelineOptions(
            ollama_model=ollama_model,
            live_prediction=True,
            hms_test_index=int(hms_test_index) if hms_test_index is not None else None,
            embed_model=embed_model.split(":")[0] if _DEPS_OK else None,
        )
        with st.spinner(f"Running with {ollama_model}..."):
            try:
                st.session_state.last_run_options = {
                    "ollama_model": ollama_model,
                    "embed_model": embed_model.split(":")[0],
                }
                result = run_case(case_id, opts)
                st.session_state.last_pipeline_result = result
                warning = (result.get("retrieval_info") or {}).get("warning")
                if warning:
                    st.warning(warning)
                n_q = len((result.get("query_plan") or {}).get("report_questions") or [])
                st.success(
                    f"Done ({result.get('ollama_model')}): "
                    f"{n_q} report questions · {result['final_report'].get('safety_status')} — see **Pipeline trace**"
                )
            except Exception as e:
                st.error(str(e))

with tab3:
    if st.session_state.last_pipeline_result:
        _render_pipeline_trace(st.session_state.last_pipeline_result)
    else:
        st.info("Run a case on the **Run pipeline** tab to see step-by-step outputs here.")

with tab4:
    if st.session_state.last_pipeline_result:
        _render_report(st.session_state.last_pipeline_result.get("final_report", {}))
    else:
        reports = _load_final_reports()
        if not reports:
            st.info("No results yet. Run the pipeline first.")
        else:
            case_ids = sorted({r.get("case_id", "") for r in reports if r.get("case_id")})
            selected = st.selectbox("Select case report (from disk)", case_ids)
            matching = [r for r in reports if r.get("case_id") == selected]
            if matching:
                _render_report(matching[-1])
                if len(matching) > 1:
                    st.caption(f"{len(matching)} runs on file — showing the most recent.")
