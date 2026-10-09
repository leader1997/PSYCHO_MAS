"""Tests for retrieval metrics."""

from controlled_agentic_xai_rag.evaluation.metrics import mrr, precision_at_k


def test_precision_at_5():
    rel = {"a", "b"}
    ret = ["a", "x", "y", "b", "z"]
    assert precision_at_k(rel, ret, 5) == 0.4


def test_mrr():
    assert mrr({"b"}, ["a", "b", "c"]) == 0.5
