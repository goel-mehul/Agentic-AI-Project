"""
Quick test script for the Planner Agent.
Run from the backend/ directory: python test_planner.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.planner import planner_agent
from agents.state import ResearchState
import uuid

def test_planner():
    # Build a minimal initial state — only what the planner needs
    state = ResearchState(
        research_question="What are the most effective techniques for reducing hallucinations in large language models?",
        session_id=str(uuid.uuid4()),
        research_plan=[],
        search_strategy="",
        raw_papers=[],
        retrieved_chunks=[],
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

    print("Running Planner Agent...\n")
    result = planner_agent(state)

    print("=== RESEARCH PLAN ===")
    for i, query in enumerate(result["research_plan"], 1):
        print(f"  {i}. {query}")

    print(f"\n=== STRATEGY ===")
    print(f"  {result['search_strategy']}")

    print(f"\n=== LOGS ===")
    for log in result["agent_logs"]:
        print(f"  {log}")

    print("\n✅ Planner Agent test passed!")

if __name__ == "__main__":
    test_planner()