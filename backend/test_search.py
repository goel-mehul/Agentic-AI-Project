"""
Manual test for the Search Agent.
Run from backend/: python test_search.py
"""

import sys, os, uuid
sys.path.insert(0, os.path.dirname(__file__))

from agents.planner import planner_agent
from agents.search import search_agent
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

    print("Step 1: Running Planner...\n")
    state = make_state(question)
    state = planner_agent(state)

    print("Search queries generated:")
    for q in state["research_plan"]:
        print(f"  • {q}")

    print("\nStep 2: Running Search Agent...\n")
    state = search_agent(state)

    print(f"Papers collected: {len(state['raw_papers'])}")
    print(f"Relevant chunks retrieved: {len(state['retrieved_chunks'])}")

    print("\nTop 3 most relevant papers:")
    for chunk in state["retrieved_chunks"][:3]:
        meta = chunk["metadata"]
        print(f"\n  📄 {meta['title']}")
        print(f"     {meta['authors']} ({meta['published']}) [{meta['source']}]")

    print("\n✅ Search Agent test complete!")