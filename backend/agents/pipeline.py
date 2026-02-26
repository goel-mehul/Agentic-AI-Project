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

def should_search_again(state: ResearchState) -> str:
    """
    Routing function: decides whether to loop back to Search or continue to Writer.

    Loops back if ALL of these are true:
    - We've done fewer than 2 extra search passes (prevents infinite loops)
    - The Critic found at least 2 gaps (worth searching again)
    - We have fewer than 35 papers (don't over-collect)

    Otherwise: proceed to Writer.
    """
    iteration   = state.get("search_iteration", 0)
    gaps        = state.get("gaps", [])
    raw_papers  = state.get("raw_papers", [])
    gap_queries = state.get("gap_queries", [])

    # Safety limits
    if iteration >= 3:           # Already done 2 extra passes (1 original + 2 loops)
        return "writer"
    if len(gaps) < 2:            # Not enough gaps to justify another search
        return "writer"
    if len(raw_papers) >= 35:    # Already have plenty of papers
        return "writer"
    if not gap_queries:          # No gap queries generated
        return "writer"

    return "search"              # Loop back for another retrieval pass


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

    # Conditional edge: Critic → Search (loop) or Critic → Writer (continue)
    graph.add_conditional_edges(
        "critic",
        should_search_again,
        {
            "search": "search",   # Loop back for more retrieval
            "writer": "writer"    # Proceed to writing
        }
    )

    # Linear flow after Writer
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
        search_iteration=0,
        gap_queries=[],
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

