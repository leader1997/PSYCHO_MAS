"""Tests for corpus / vector index synchronization."""

from controlled_agentic_xai_rag.retrieval.index_sync import (
    compute_corpus_fingerprint,
    indexes_need_rebuild,
    record_corpus_built,
)


def test_fingerprint_stable(built_corpus):
    fp1 = compute_corpus_fingerprint()
    fp2 = compute_corpus_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64


def test_record_corpus_built(built_corpus):
    fp = record_corpus_built()
    assert fp
    assert indexes_need_rebuild()
