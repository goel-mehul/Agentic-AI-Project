"""
Automated tests for the Fact Checker Agent.
Run from project root: python -m pytest evals/test_fact_checker.py -v
"""

import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from agents.fact_checker import fact_checker_agent
from agents.state import ResearchState


def make_state(question: str, draft: str, chunks: list) -> ResearchState:
    return ResearchState(
        research_question=question,
        session_id=str(uuid.uuid4()),
        research_plan=[], search_strategy="",
        raw_papers=[], retrieved_chunks=chunks,
        evidence_quality={}, contradictions=[], gaps=[],
        draft_report=draft, report_sections={},
        final_report="", fact_check_notes=[],
        agent_logs=[], current_agent="",
        status="running", error=""
    )


MOCK_DRAFT = """## Executive Summary
RLHF and RAG are the most effective techniques for reducing hallucinations in LLMs.

## Key Findings
Reinforcement learning from human feedback reduces hallucinations by 40% [Smith et al., 2023].
Retrieval-augmented generation anchors outputs in verified documents [Jones et al., 2023].

## Gaps in the Literature
Limited research on multilingual hallucination reduction.

## References
1. Smith et al. (2023) - RLHF Reduces Hallucinations
2. Jones et al. (2023) - RAG for Factual Grounding"""

MOCK_CHUNKS = [
    {
        "content": "Title: RLHF Reduces Hallucinations\n\nAbstract: RLHF achieves 40% reduction in hallucination rate on TruthfulQA.",
        "metadata": {"title": "RLHF Reduces Hallucinations", "authors": "Smith et al.", "published": "2023", "source": "arxiv", "url": "https://arxiv.org/abs/1234"}
    },
    {
        "content": "Title: RAG for Factual Grounding\n\nAbstract: RAG reduces hallucinations by grounding responses in retrieved documents.",
        "metadata": {"title": "RAG for Factual Grounding", "authors": "Jones et al.", "published": "2023", "source": "arxiv", "url": "https://arxiv.org/abs/5678"}
    },
]


def test_fact_checker_produces_final_report():
    """Fact Checker must produce a non-empty final report."""
    state = make_state("How to reduce hallucinations?", MOCK_DRAFT, MOCK_CHUNKS)
    result = fact_checker_agent(state)
    assert len(result["final_report"]) > 100


def test_fact_checker_sets_status_complete():
    """Pipeline status must be 'complete' after Fact Checker runs."""
    state = make_state("How to reduce hallucinations?", MOCK_DRAFT, MOCK_CHUNKS)
    result = fact_checker_agent(state)
    assert result["status"] == "complete"


def test_fact_checker_appends_confidence_rating():
    """Final report must contain a confidence rating."""
    state = make_state("How to reduce hallucinations?", MOCK_DRAFT, MOCK_CHUNKS)
    result = fact_checker_agent(state)
    report_lower = result["final_report"].lower()
    assert "confidence" in report_lower


def test_fact_checker_handles_empty_draft():
    """Fact Checker must handle empty draft without crashing."""
    state = make_state("How to reduce hallucinations?", "", MOCK_CHUNKS)
    result = fact_checker_agent(state)
    assert result["status"] == "error"
    assert len(result["final_report"]) > 0


def test_fact_checker_sets_current_agent():
    """current_agent must be set to 'fact_checker'."""
    state = make_state("How to reduce hallucinations?", MOCK_DRAFT, MOCK_CHUNKS)
    result = fact_checker_agent(state)
    assert result["current_agent"] == "fact_checker"


def test_fact_checker_fact_check_notes_is_list():
    """fact_check_notes must always be a list."""
    state = make_state("How to reduce hallucinations?", MOCK_DRAFT, MOCK_CHUNKS)
    result = fact_checker_agent(state)
    assert isinstance(result["fact_check_notes"], list)


def test_fact_checker_final_report_contains_original_content():
    """Final report must preserve the core content of the draft."""
    state = make_state("How to reduce hallucinations?", MOCK_DRAFT, MOCK_CHUNKS)
    result = fact_checker_agent(state)
    assert "executive summary" in result["final_report"].lower()
    assert "references" in result["final_report"].lower()