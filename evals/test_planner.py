"""
Automated tests for the Planner Agent.
Run from project root: pytest evals/test_planner.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import uuid
import pytest
from agents.planner import planner_agent
from agents.state import ResearchState


def make_state(question: str) -> ResearchState:
    """Helper to build a minimal state for testing."""
    return ResearchState(
        research_question=question,
        session_id=str(uuid.uuid4()),
        research_plan=[], search_strategy="",
        raw_papers=[], retrieved_chunks=[],
        evidence_quality={}, contradictions=[], gaps=[],
        draft_report="", report_sections={},
        final_report="", fact_check_notes=[],
        agent_logs=[], current_agent="",
        status="running", error=""
    )


def test_planner_returns_search_queries():
    """Planner must return at least 3 search queries."""
    state = make_state("What techniques reduce hallucinations in LLMs?")
    result = planner_agent(state)
    assert len(result["research_plan"]) >= 3


def test_planner_returns_strategy():
    """Planner must return a non-empty search strategy."""
    state = make_state("How does retrieval augmented generation work?")
    result = planner_agent(state)
    assert len(result["search_strategy"]) > 20


def test_planner_writes_logs():
    """Planner must write at least one log entry."""
    state = make_state("What is reinforcement learning from human feedback?")
    result = planner_agent(state)
    assert len(result["agent_logs"]) >= 1


def test_planner_sets_current_agent():
    """current_agent field must be set to 'planner'."""
    state = make_state("How do transformer attention mechanisms work?")
    result = planner_agent(state)
    assert result["current_agent"] == "planner"


def test_planner_does_not_touch_other_fields():
    """Planner should not modify fields it doesn't own."""
    state = make_state("What is federated learning?")
    result = planner_agent(state)
    # These fields belong to later agents — planner must not touch them
    assert result["raw_papers"] == []
    assert result["final_report"] == ""
    assert result["draft_report"] == ""