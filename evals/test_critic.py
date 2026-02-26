"""
Automated tests for the Critic Agent.
Run from project root: python -m pytest evals/test_critic.py -v
"""

import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from agents.critic import critic_agent
from agents.state import ResearchState


def make_state(question: str, chunks: list) -> ResearchState:
    return ResearchState(
        research_question=question,
        session_id=str(uuid.uuid4()),
        research_plan=[], search_strategy="",
        raw_papers=[], retrieved_chunks=chunks,
        evidence_quality={}, contradictions=[], gaps=[],
        draft_report="", report_sections={},
        final_report="", fact_check_notes=[],
        agent_logs=[], current_agent="",
        status="running", error=""
    )


# ── Reusable mock chunks ──────────────────────────────────────────────────────
# We use mock data here so these tests don't need real API calls to arXiv.
# This makes them fast and reliable.

MOCK_CHUNKS = [
    {
        "content": "Title: Reducing Hallucinations via RLHF\n\nAbstract: This paper proposes using reinforcement learning from human feedback to reduce factual hallucinations in large language models. We demonstrate a 40% reduction in hallucination rate on TruthfulQA.",
        "metadata": {"title": "Reducing Hallucinations via RLHF", "authors": "Smith et al.", "published": "2023", "source": "arxiv", "url": "https://arxiv.org/abs/1234"}
    },
    {
        "content": "Title: RAG Reduces Hallucinations\n\nAbstract: Retrieval-augmented generation significantly reduces hallucinations by grounding model outputs in retrieved documents. We show improvements across multiple factual benchmarks.",
        "metadata": {"title": "RAG Reduces Hallucinations", "authors": "Jones et al.", "published": "2023", "source": "arxiv", "url": "https://arxiv.org/abs/5678"}
    },
    {
        "content": "Title: Hallucinations are Irreducible\n\nAbstract: We argue that hallucinations in LLMs are a fundamental property of the architecture and cannot be fully eliminated through post-training techniques alone.",
        "metadata": {"title": "Hallucinations are Irreducible", "authors": "Lee et al.", "published": "2022", "source": "semantic_scholar", "url": "https://example.com"}
    },
]


def test_critic_returns_evidence_quality():
    """Critic must score at least one paper."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = critic_agent(state)
    assert len(result["evidence_quality"]) >= 1


def test_critic_returns_gaps():
    """Critic must identify at least one literature gap."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = critic_agent(state)
    assert len(result["gaps"]) >= 1


def test_critic_handles_empty_chunks_gracefully():
    """Critic must not crash when no papers are retrieved."""
    state = make_state("Some question", [])
    result = critic_agent(state)
    # Should still return valid state, just with empty/default values
    assert result["status"] == "running"
    assert len(result["agent_logs"]) >= 1


def test_critic_sets_current_agent():
    """current_agent must be set to 'critic'."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = critic_agent(state)
    assert result["current_agent"] == "critic"


def test_critic_does_not_touch_report_fields():
    """Critic must not write to fields owned by the Writer."""
    state = make_state("How to reduce LLM hallucinations?", MOCK_CHUNKS)
    result = critic_agent(state)
    assert result["draft_report"] == ""
    assert result["final_report"] == ""