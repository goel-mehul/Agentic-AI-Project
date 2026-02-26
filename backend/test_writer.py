"""
Manual test for the Writer Agent.
Run from project root: python backend/test_writer.py
"""

import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agents.planner import planner_agent
from agents.search import search_agent
from agents.critic import critic_agent
from agents.writer import writer_agent
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

    print("Step 1: Planner...")
    state = make_state(question)
    state = planner_agent(state)
    print(f"  ✓ {len(state['research_plan'])} queries generated")

    print("Step 2: Search...")
    state = search_agent(state)
    print(f"  ✓ {len(state['raw_papers'])} papers, {len(state['retrieved_chunks'])} chunks")

    print("Step 3: Critic...")
    state = critic_agent(state)
    print(f"  ✓ {len(state['contradictions'])} contradictions, {len(state['gaps'])} gaps")

    print("Step 4: Writer...\n")
    state = writer_agent(state)

    print("=" * 60)
    print(state["draft_report"])
    print("=" * 60)

    print(f"\n✅ Writer complete!")
    print(f"   Sections: {list(state['report_sections'].keys())}")
    print(f"   Word count: {len(state['draft_report'].split())}")