"""
Automated tests for the Search Agent.
Run from project root: python -m pytest evals/test_search.py -v
"""

import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from agents.search import _search_arxiv, _search_semantic_scholar, search_agent
from agents.state import ResearchState


def make_state(question: str, plan: list[str]) -> ResearchState:
    return ResearchState(
        research_question=question,
        session_id=str(uuid.uuid4()),
        research_plan=plan, search_strategy="",
        raw_papers=[], retrieved_chunks=[],
        evidence_quality={}, contradictions=[], gaps=[],
        draft_report="", report_sections={},
        final_report="", fact_check_notes=[],
        agent_logs=[], current_agent="",
        status="running", error=""
    )


def test_arxiv_returns_papers():
    """arXiv search should return at least 1 paper for a known topic."""
    papers = _search_arxiv("transformer attention mechanism", max_results=3)
    assert len(papers) >= 1
    assert "title" in papers[0]
    assert "abstract" in papers[0]


def test_arxiv_paper_has_required_fields():
    """Each paper must have all fields the rest of the pipeline depends on."""
    papers = _search_arxiv("large language models", max_results=2)
    required = ["paper_id", "title", "authors", "abstract", "published", "source", "url"]
    for paper in papers:
        for field in required:
            assert field in paper, f"Missing field: {field}"


def test_search_agent_populates_raw_papers():
    """Search agent must populate raw_papers from the queries."""
    state = make_state(
        "How does retrieval augmented generation reduce hallucinations?",
        ["retrieval augmented generation", "LLM hallucination reduction"]
    )
    result = search_agent(state)
    assert len(result["raw_papers"]) > 0


def test_search_agent_populates_retrieved_chunks():
    """Search agent must populate retrieved_chunks via ChromaDB."""
    state = make_state(
        "What is reinforcement learning from human feedback?",
        ["reinforcement learning from human feedback RLHF"]
    )
    result = search_agent(state)
    assert len(result["retrieved_chunks"]) > 0


def test_retrieved_chunks_have_metadata():
    """Each chunk must have content and metadata for downstream agents."""
    state = make_state(
        "How do transformer models work?",
        ["transformer self-attention mechanism"]
    )
    result = search_agent(state)
    for chunk in result["retrieved_chunks"]:
        assert "content" in chunk
        assert "metadata" in chunk
        assert "title" in chunk["metadata"]


def test_search_deduplicates_papers():
    """Running the same query twice should not produce duplicate papers."""
    state = make_state(
        "neural network training",
        ["neural network training", "neural network training"]  # duplicate queries
    )
    result = search_agent(state)
    titles = [p["title"].lower() for p in result["raw_papers"]]
    assert len(titles) == len(set(titles)), "Duplicate papers found!"