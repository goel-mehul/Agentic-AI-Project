"""
Tests for the FastAPI backend.
Run from project root: python -m pytest evals/test_api.py -v

Uses FastAPI's TestClient — no need to run the server manually.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint_returns_200():
    """Root endpoint must return 200 and service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "agents" in data


def test_root_lists_all_five_agents():
    """Root endpoint must list all 5 agents."""
    response = client.get("/")
    agents = response.json()["agents"]
    assert "planner"      in agents
    assert "search"       in agents
    assert "critic"       in agents
    assert "writer"       in agents
    assert "fact_checker" in agents


def test_start_research_returns_session_id():
    """POST /research must return a session_id."""
    response = client.post(
        "/research",
        json={"question": "How does transformer attention work?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "running"


def test_start_research_empty_question_returns_400():
    """POST /research with empty question must return 400."""
    response = client.post("/research", json={"question": ""})
    assert response.status_code == 400


def test_start_research_whitespace_question_returns_400():
    """POST /research with whitespace question must return 400."""
    response = client.post("/research", json={"question": "   "})
    assert response.status_code == 400


def test_get_result_unknown_session_returns_404():
    """GET /research/{id} with unknown session_id must return 404."""
    response = client.get("/research/nonexistent-session-id")
    assert response.status_code == 404


def test_get_result_running_session_returns_running():
    """GET /research/{id} for a running session must return 'running'."""
    # Start a session
    post_response = client.post(
        "/research",
        json={"question": "What is federated learning?"}
    )
    session_id = post_response.json()["session_id"]

    # Immediately poll — should still be running
    get_response = client.get(f"/research/{session_id}")
    assert get_response.status_code == 200
    assert get_response.json()["status"] == "running"