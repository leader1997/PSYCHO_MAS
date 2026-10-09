"""Tests for safety rules."""

from controlled_agentic_xai_rag.safety.clinical_boundary import clinical_boundary_flags, has_disclaimer


def test_diagnostic_overclaim():
    flags = clinical_boundary_flags("You have depression and anxiety.")
    assert flags["diagnostic_overclaiming"]


def test_disclaimer_present():
    text = "This is a model-based screening signal and not a clinical diagnosis."
    assert has_disclaimer(text)
