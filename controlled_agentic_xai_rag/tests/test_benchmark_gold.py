"""Benchmark gold chunk resolution tests."""

import pandas as pd

from controlled_agentic_xai_rag.evaluation.benchmark_gold import resolve_gold_chunks, load_corpus_chunks, load_corpus_metadata


def test_resolve_gold_not_empty_for_benchmark():
    bench = pd.read_csv("data/benchmarks/rag_benchmark_100_questions.csv").head(10)
    chunks = load_corpus_chunks()
    meta = load_corpus_metadata()
    counts = [len(resolve_gold_chunks(row, chunks, meta)) for _, row in bench.iterrows()]
    assert all(c > 0 for c in counts), counts
