"""
pipeline.py — LangGraph Pipeline
==================================

WHAT IT DOES:
    Wires all 5 agents together into a single LangGraph StateGraph.
    This is the conductor that orchestrates the whole pipeline.

WHY IT EXISTS:
    Instead of manually calling agents one by one, LangGraph manages:
    - State passing between agents automatically
    - Streaming updates after each agent completes
    - A clean single entry point: research_graph.invoke(state)
    - Future support for conditional loops and parallel execution

HOW IT FITS:
    This is what the FastAPI backend will call.
    Everything upstream (agents) feeds into this.
    Everything downstream (API, frontend) consumes from this.

WHAT YOU'RE LEARNING:
    - How LangGraph StateGraph works
    - The difference between invoke() and stream()
    - How to build a production-ready agent orchestration layer
"""

import uuid
from langgraph.graph import StateGraph, END
from .state import ResearchState
from .planner import planner_agent
from .search import search_agent
from .critic import critic_agent
from .writer import writer_agent
from .fact_checker import fact_checker_agent


def build_research_graph() -> StateGraph:
    """
    Constructs and compiles the multi-agent LangGraph pipeline.

    The graph looks like this:
        planner → search → critic → writer → fact_checker → END

    Each node is an agent function with signature:
        def agent(state: ResearchState) -> ResearchState

    LangGraph automatically:
    - Passes state from one node to the next
    - Handles the Annotated[list, operator.add] reducer for agent_logs
    - Supports .stream() for real-time updates
    """

    # Initialize the graph with our shared state type
    # LangGraph uses this to know what fields exist and how to merge them
    graph = StateGraph(ResearchState)

    # ── Add nodes (each node = one agent) ────────────────────────────────
    graph.add_node("planner",      planner_agent)
    graph.add_node("search",       search_agent)
    graph.add_node("critic",       critic_agent)
    graph.add_node("writer",       writer_agent)
    graph.add_node("fact_checker", fact_checker_agent)

    # ── Define edges (the flow between agents) ────────────────────────────
    # set_entry_point tells LangGraph where to start
    graph.set_entry_point("planner")

    # add_edge defines the linear flow
    graph.add_edge("planner",      "search")
    graph.add_edge("search",       "critic")
    graph.add_edge("critic",       "writer")
    graph.add_edge("writer",       "fact_checker")
    graph.add_edge("fact_checker", END)

    # compile() validates the graph and returns a runnable object
    return graph.compile()


def create_initial_state(research_question: str) -> ResearchState:
    """
    Creates a clean initial state for a new research session.

    This is the only place where state is created from scratch.
    All fields start empty — agents populate them as they run.
    """
    return ResearchState(
        research_question=research_question,
        session_id=str(uuid.uuid4()),
        research_plan=[],
        search_strategy="",
        raw_papers=[],
        retrieved_chunks=[],
        citation_counts={},
        evidence_quality={},
        contradictions=[],
        gaps=[],
        draft_report="",
        report_sections={},
        final_report="",
        fact_check_notes=[],
        agent_logs=[],
        current_agent="",
        status="running",
        error=""
    )


# ── Compile once at import time ───────────────────────────────────────────────
# We compile the graph once when this module is imported, then reuse it
# for every research request. Compiling is expensive; running is cheap.
research_graph = build_research_graph()