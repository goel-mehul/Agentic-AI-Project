"""
state.py — Shared State for the Multi-Agent Pipeline
=====================================================

This is the most important file in the project.

Every agent in our pipeline reads from and writes to this single
ResearchState object. Think of it as a shared whiteboard that gets
passed from agent to agent — each one reads what previous agents
wrote, does its job, and adds its own outputs.

LangGraph manages passing this state between agents automatically.
We just define the shape of it here using Python's TypedDict.
"""

from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    """
    The shared state object that flows through all 5 agents.

    Why TypedDict?
    - Gives us type hints so our editor can autocomplete field names
    - Makes the structure explicit and readable
    - LangGraph requires this pattern for its StateGraph

    The Annotated[list, operator.add] pattern on agent_logs is special:
    it tells LangGraph to APPEND to this list rather than replace it.
    Every other field gets replaced when an agent writes to it.
    """

    # ── Input ────────────────────────────────────────────────────────────
    research_question: str      # The user's original question
    session_id: str             # Unique ID for this research run

    # ── Planner Agent outputs ────────────────────────────────────────────
    research_plan: list[str]    # List of search queries to run
    search_strategy: str        # High-level description of approach

    # ── Search Agent outputs ─────────────────────────────────────────────
    raw_papers: list[dict]      # All papers fetched from arXiv / Semantic Scholar
    retrieved_chunks: list[dict] # Top-k most relevant paper sections (from ChromaDB)
    citation_counts: dict       # Maps paper_id -> citation count (from Semantic Scholar)

    # ── Critic Agent outputs ─────────────────────────────────────────────
    evidence_quality: dict      # Quality scores per paper
    contradictions: list[str]   # Conflicting findings between papers
    gaps: list[str]             # Topics not covered by the evidence

    # ── Writer Agent outputs ─────────────────────────────────────────────
    draft_report: str           # Initial markdown report
    report_sections: dict       # Report broken into named sections

    # ── Fact Checker Agent outputs ───────────────────────────────────────
    final_report: str           # Verified, finalized report
    fact_check_notes: list[str] # Corrections made by the fact checker

    # ── Pipeline metadata ────────────────────────────────────────────────
    # Annotated + operator.add = "append to this list, don't replace it"
    # This means every agent's logs accumulate across the whole pipeline
    agent_logs: Annotated[list[str], operator.add]
    current_agent: str          # Which agent is currently running
    status: str                 # "running" | "complete" | "error"
    error: str                  # Error message if something went wrong