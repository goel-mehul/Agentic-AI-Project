"""
planner.py — The Planner Agent
===============================

WHAT IT DOES:
    Takes the user's research question and breaks it into a concrete
    research plan: a set of focused search queries and a strategy.

WHY IT EXISTS:
    A broad question like "how do LLMs hallucinate?" is too vague to
    search for directly. The Planner produces specific, targeted queries
    that the Search Agent can actually use to find relevant papers.

HOW IT FITS:
    First node in the LangGraph pipeline.
    Reads:  research_question (set by the user)
    Writes: research_plan, search_strategy, agent_logs

WHAT YOU'RE LEARNING:
    - How to call the Anthropic API
    - How to use structured JSON outputs from an LLM
    - How an agent reads from and writes to shared state
"""

import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from .state import ResearchState

# Load API key from backend/.env
load_dotenv()

# One shared client — reused across all calls
client = Anthropic()


# ── System Prompt ─────────────────────────────────────────────────────────────
# This is the "personality" and instructions for this specific agent.
# Notice it's very focused: plan only, no searching, no writing.

PLANNER_SYSTEM_PROMPT = """You are a research planning expert working with academic databases.

Your job: take a research question and create a precise search plan.

Output a JSON object with exactly these fields:
- "sub_questions": list of 3-5 focused sub-questions that together fully answer the main question
- "search_queries": list of 4-6 specific search queries optimized for arXiv and Semantic Scholar
- "strategy": 2-3 sentences describing the overall research approach

Rules:
- Be specific and technical, not vague
- Search queries should use academic terminology
- Each query should target a different aspect of the question
- Output ONLY valid JSON, no explanation, no markdown code fences"""


# ── Agent Function ────────────────────────────────────────────────────────────
# Every agent in our pipeline follows the same pattern:
#   def agent_name(state: ResearchState) -> ResearchState
# LangGraph calls this function and passes the current state.
# We return the updated state.

def planner_agent(state: ResearchState) -> ResearchState:
    """
    Planner Agent — first node in the LangGraph pipeline.

    Args:
        state: The shared ResearchState (only research_question is set at this point)

    Returns:
        Updated state with research_plan and search_strategy populated
    """
    question = state["research_question"]

    # Log what we're doing (these stream to the frontend in real time)
    state["current_agent"] = "planner"
    state["agent_logs"] = [f"🧠 Planner: Analyzing question — '{question}'"]

    # ── Call the Anthropic API ────────────────────────────────────────────
    # We ask Claude to act as a research planner and return structured JSON.
    # Using claude-haiku for speed and low cost on this lightweight task.

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=PLANNER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Research question: {question}\n\nCreate the research plan."
            }
        ]
    )

    # ── Parse the JSON response ───────────────────────────────────────────
    raw = response.content[0].text.strip()

    # Defensive: strip markdown code fences if Claude added them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    plan_data = json.loads(raw)

    # ── Write outputs back to state ───────────────────────────────────────
    state["research_plan"]   = plan_data.get("search_queries", [])
    state["search_strategy"] = plan_data.get("strategy", "")

    n = len(state["research_plan"])
    state["agent_logs"] = [
        f"✅ Planner: Generated {n} search queries",
        f"📋 Strategy: {state['search_strategy']}"
    ]

    return state