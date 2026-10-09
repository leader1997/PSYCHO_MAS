"""Streamlit UI for Local Agentic XAI-RAG.

Run from project root:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from local_agentic_xai_rag.ui.pages import (
    page_case_reports,
    page_evaluation,
    page_figures,
    page_overview,
    page_retrieval,
    page_run_pipeline,
    render_sidebar,
)

st.set_page_config(
    page_title="Local XAI-RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Overview": page_overview,
    "Run pipeline": page_run_pipeline,
    "Case reports": page_case_reports,
    "Retrieval explorer": page_retrieval,
    "Evaluation dashboard": page_evaluation,
    "Figures & paper": page_figures,
}

page_name = render_sidebar()
PAGES[page_name]()
