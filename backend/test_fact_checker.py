"""
Manual test for the Fact Checker Agent.
Run from project root: python backend/test_fact_checker.py

This runs the FULL pipeline end to end — all 5 agents.
This is the most important test script in the project.
"""

import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agents.planner import planner_agent
from agents.search import search_agent
from agents.critic import critic_agent
from agents.writer import writer_agent
from agents.fact_checker import fact_checker_agent
from agents.state import ResearchState


def make_state(question: str) -> ResearchState:
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


if __name__ == "__main__":
    question = "What are the most effective techniques for reducing hallucinations in large language models?"

    print("🚀 Running full 5-agent pipeline...\n")
    state = make_state(question)

    print("1/5 Planner...")
    state = planner_agent(state)
    print(f"    ✓ {len(state['research_plan'])} search queries")

    print("2/5 Search...")
    state = search_agent(state)
    print(f"    ✓ {len(state['raw_papers'])} papers, {len(state['retrieved_chunks'])} chunks")

    print("3/5 Critic...")
    state = critic_agent(state)
    print(f"    ✓ {len(state['gaps'])} gaps, {len(state['contradictions'])} contradictions")

    print("4/5 Writer...")
    state = writer_agent(state)
    print(f"    ✓ {len(state['draft_report'].split())} word draft")

    print("5/5 Fact Checker...")
    state = fact_checker_agent(state)
    print(f"    ✓ {len(state['fact_check_notes'])} corrections")

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(state["final_report"])
    print("=" * 60)

    print(f"\nPipeline status: {state['status']}")
    print(f"Total papers found: {len(state['raw_papers'])}")
    print(f"Corrections made: {state['fact_check_notes']}")
    print("\n✅ Full pipeline test complete!")