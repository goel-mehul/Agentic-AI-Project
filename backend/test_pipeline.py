"""
Test the full LangGraph pipeline using research_graph.stream()

This is different from test_fact_checker.py — instead of calling
agents manually, we let LangGraph orchestrate everything.

Run from project root: python backend/test_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agents.pipeline import research_graph, create_initial_state

if __name__ == "__main__":
    question = "What are the most effective techniques for reducing hallucinations in large language models?"

    print(f"🔬 Research question: {question}\n")
    print("=" * 60)

    # create_initial_state builds a clean starting state
    initial_state = create_initial_state(question)

    # ── research_graph.stream() ───────────────────────────────────────────
    # This is the key difference from manual chaining.
    # .stream() yields a snapshot after EACH agent completes.
    # Each snapshot is a dict like: {"planner": {...updated state...}}
    # This is what lets us show real-time updates in the frontend.

    final_state = None

    for step in research_graph.stream(initial_state):
        # step is a dict with one key: the agent that just ran
        for agent_name, state_update in step.items():
            print(f"\n✅ Agent complete: {agent_name.upper()}")

            # Print that agent's logs
            for log in state_update.get("agent_logs", []):
                print(f"   {log}")

            final_state = state_update

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    if final_state:
        print(final_state.get("final_report", "No report generated"))
        print(f"\nStatus: {final_state.get('status')}")
        print(f"Papers found: {len(final_state.get('raw_papers', []))}")

    print("\n✅ LangGraph pipeline test complete!")