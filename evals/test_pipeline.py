"""
Tests for the LangGraph pipeline structure.
Run from project root: python -m pytest evals/test_pipeline.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from agents.pipeline import research_graph, create_initial_state


def test_graph_compiles():
    """The LangGraph graph must compile without errors."""
    from agents.pipeline import build_research_graph
    graph = build_research_graph()
    assert graph is not None


def test_create_initial_state_has_all_fields():
    """Initial state must have all required ResearchState fields."""
    state = create_initial_state("test question")
    required_fields = [
        "research_question", "session_id", "research_plan",
        "search_strategy", "raw_papers", "retrieved_chunks",
        "evidence_quality", "contradictions", "gaps",
        "draft_report", "report_sections", "final_report",
        "fact_check_notes", "agent_logs", "current_agent",
        "status", "error"
    ]
    for field in required_fields:
        assert field in state, f"Missing field: {field}"


def test_initial_state_question_is_set():
    """research_question must be set correctly in initial state."""
    question = "How do transformers work?"
    state = create_initial_state(question)
    assert state["research_question"] == question


def test_initial_state_status_is_running():
    """Initial status must be 'running', not 'complete'."""
    state = create_initial_state("test")
    assert state["status"] == "running"


def test_initial_state_lists_are_empty():
    """All list fields must start empty."""
    state = create_initial_state("test")
    assert state["research_plan"] == []
    assert state["raw_papers"] == []
    assert state["agent_logs"] == []
    assert state["fact_check_notes"] == []


def test_each_session_gets_unique_id():
    """Every research session must have a unique session_id."""
    state1 = create_initial_state("question 1")
    state2 = create_initial_state("question 2")
    assert state1["session_id"] != state2["session_id"]


def test_research_graph_is_compiled():
    """The module-level research_graph must be a compiled runnable."""
    # If it compiled, it has an invoke method
    assert hasattr(research_graph, "invoke")
    assert hasattr(research_graph, "stream")