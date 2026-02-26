"""
Manual test for the Critic Agent.
Run from project root: python backend/test_critic.py
"""

import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agents.planner import planner_agent
from agents.search import search_agent
from agents.critic import critic_agent
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

    print("Step 1: Planner...\n")
    state = make_state(question)
    state = planner_agent(state)
    print(f"  Generated {len(state['research_plan'])} queries")

    print("\nStep 2: Search...\n")
    state = search_agent(state)
    print(f"  Found {len(state['raw_papers'])} papers")
    print(f"  Retrieved {len(state['retrieved_chunks'])} relevant chunks")

    print("\nStep 3: Critic...\n")
    state = critic_agent(state)
    
    print("=== EVIDENCE QUALITY ===")
    evidence = state["evidence_quality"] or {}
    for title, score_data in list(evidence.items())[:4]:
        if score_data is None:
            print(f"  ? — {title}")
        elif isinstance(score_data, dict):
            print(f"  {score_data.get('score', '?')} — {title}")
            print(f"         {score_data.get('rationale', '')}")
        else:
            print(f"  {score_data} — {title}")

    print("\n=== CONTRADICTIONS ===")
    if state["contradictions"]:
        for c in state["contradictions"]:
            print(f"  ⚡ {c}")
    else:
        print("  None found")

    print("\n=== GAPS ===")
    for g in state["gaps"]:
        print(f"  🕳️  {g}")

    print("\n=== CRITIC LOGS ===")
    for log in state["agent_logs"]:
        print(f"  {log}")

    print("\n✅ Critic Agent test complete!")