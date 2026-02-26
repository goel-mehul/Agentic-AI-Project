"""
Automated tests for the Writer Agent.
Run from project root: python -m pytest evals/test_writer.py -v
"""

import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from agents.writer import writer_agent
from agents.state import ResearchState


def make_state(question: str, chunks: list, contradictions=None, gaps=None) -> ResearchState:
    return ResearchState(
        research_question=question,
        session_id=str(uuid.uuid4()),
        research_plan=[], search_strategy="Comprehensive search across multiple databases",
        raw_papers=[], retrieved_chunks=chunks,
        evidence_quality={"Test Paper (2023)": {"score": 0.85, "rationale": "Strong methodology"}},
        contradictions=contradictions or [],
        gaps=gaps or [],
        draft_report="", report_sections={},
        final_report="", fact_check_notes=[],
        agent_logs=[], current_agent="",
        status="running", error=""
    )


MOCK_CHUNKS = [
    {
        "content": "Title: RLHF Reduces Hallucinations\n\nAbstract: Reinforcement learning from human feedback significantly reduces hallucination rates in LLMs, achieving 40% improvement on TruthfulQA benchmark.",
        "metadata": {"title": "RLHF Reduces Hallucinations", "authors": "Smith et al.", "published": "2023", "source": "arxiv", "url": "https://arxiv.org/abs/1234"}
    },
    {
        "content": "Title: RAG for Factual Grounding\n\nAbstract: Retrieval-augmented generation grounds model outputs in retrieved documents, reducing hallucinations by anchoring responses to verified sources.",
        "metadata": {"title": "RAG for Factual Grounding", "authors": "Jones et al.", "published": "2023", "source": "arxiv", "url": "https://arxiv.org/abs/5678"}
    },
    {
        "content": "Title: Constitutional AI Methods\n\nAbstract: Constitutional AI training methods improve factual consistency by having models critique and revise their own outputs against a set of principles.",
        "metadata": {"title": "Constitutional AI Methods", "authors": "Lee et al.", "published": "2023", "source": "arxiv", "url": "https://arxiv.org/abs/9012"}
    },
]


def test_writer_produces_draft_report():
    """Writer must produce a non-empty draft report."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = writer_agent(state)
    assert len(result["draft_report"]) > 100


def test_writer_report_has_executive_summary():
    """Report must contain an Executive Summary section."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = writer_agent(state)
    assert "executive summary" in result["draft_report"].lower()


def test_writer_report_has_references():
    """Report must contain a References section."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = writer_agent(state)
    assert "references" in result["draft_report"].lower()


def test_writer_populates_report_sections():
    """Writer must parse the report into named sections."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = writer_agent(state)
    assert len(result["report_sections"]) >= 3


def test_writer_includes_contradictions_when_present():
    """When contradictions exist, report should address them."""
    contradictions = ["Smith et al. claim RLHF works; Lee et al. dispute this"]
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS,
                       contradictions=contradictions)
    result = writer_agent(state)
    report_lower = result["draft_report"].lower()
    assert "contradict" in report_lower or "debate" in report_lower or "disagree" in report_lower


def test_writer_sets_current_agent():
    """current_agent must be set to 'writer'."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = writer_agent(state)
    assert result["current_agent"] == "writer"


def test_writer_does_not_touch_final_report():
    """Writer owns draft_report, not final_report — that belongs to Fact Checker."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = writer_agent(state)
    assert result["final_report"] == ""