"""
critic.py — The Critic Agent
=============================

WHAT IT DOES:
    Reads all retrieved paper sections and critically evaluates them:
    - Scores each paper's quality and relevance (0.0 to 1.0)
    - Identifies contradictions between papers
    - Notes gaps in the literature
    - Flags which papers are most trustworthy

WHY IT EXISTS:
    Not all evidence is equal. A 2018 paper may be outdated. A paper
    with only 3 citations may be fringe. Two papers may directly
    disagree. The Writer needs to know this BEFORE writing — otherwise
    it produces a confidently wrong report.

HOW IT FITS:
    Third node in the pipeline.
    Reads:  retrieved_chunks (from Search), research_question
    Writes: evidence_quality, contradictions, gaps, agent_logs

WHAT YOU'RE LEARNING:
    - How to use LLMs for evaluation, not just generation
    - Why multi-agent systems need internal quality checks
    - How to prompt for structured critical analysis
"""

import json
from anthropic import Anthropic
from dotenv import load_dotenv
from .state import ResearchState

load_dotenv()
client = Anthropic()


# ── System Prompt ─────────────────────────────────────────────────────────────
# Notice this prompt is completely different from the Planner's.
# This agent is a skeptic — its job is to find problems, not solutions.

CRITIC_SYSTEM_PROMPT = """You are a rigorous academic peer reviewer with high standards.

Your job: critically evaluate a set of research papers retrieved for a given question.

Output a JSON object with exactly these fields:
- "quality_scores": dict mapping shortened paper titles to scores 0.0-1.0 with one-line rationale
  Example: {"Attention Is All You Need (2017)": {"score": 0.95, "rationale": "Seminal, highly cited, directly relevant"}}
- "contradictions": list of strings describing conflicting findings between papers (empty list if none)
- "gaps": list of strings describing important aspects of the question NOT covered by the evidence
- "high_quality_papers": list of 3-5 paper titles that are most reliable and relevant
- "summary": 2-3 sentences on overall evidence quality

Be intellectually honest. Finding weaknesses makes the final report MORE credible, not less.
Output ONLY valid JSON, no explanation, no markdown."""


def critic_agent(state: ResearchState) -> ResearchState:
    """
    Critic Agent — third node in the LangGraph pipeline.

    Args:
        state: ResearchState with retrieved_chunks populated by Search agent

    Returns:
        Updated state with evidence_quality, contradictions, and gaps populated
    """
    chunks   = state["retrieved_chunks"]
    question = state["research_question"]

    state["current_agent"] = "critic"
    state["agent_logs"]    = [
        f"🔬 Critic: Evaluating {len(chunks)} evidence pieces..."
    ]

    # Edge case: no papers found — note it and move on gracefully
    if not chunks:
        state["evidence_quality"] = {}
        state["contradictions"]   = ["Insufficient papers found to identify contradictions."]
        state["gaps"]             = ["No papers retrieved — search returned no results."]
        state["agent_logs"]       = ["⚠️ Critic: No evidence to evaluate."]
        return state

    # ── Format evidence for the critic ───────────────────────────────────
    # We truncate each abstract to 800 chars to stay within token limits.
    # The critic doesn't need the full text — just enough to judge quality.

    evidence_text = "\n\n---\n\n".join([
        f"Paper: {c['metadata'].get('title', 'Unknown')}\n"
        f"Authors: {c['metadata'].get('authors', 'Unknown')}\n"
        f"Published: {c['metadata'].get('published', 'Unknown')}\n"
        f"Source: {c['metadata'].get('source', 'Unknown')}\n\n"
        f"{c['content'][:800]}"
        for c in chunks[:8]  # Cap at 8 to control token usage
    ])

    # ── Call Claude ───────────────────────────────────────────────────────
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Fast + cheap for evaluation tasks
        max_tokens=1500,
        system=CRITIC_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research question: {question}\n\n"
                    f"Retrieved evidence:\n\n{evidence_text}\n\n"
                    f"Please critically evaluate this evidence."
                )
            }
        ]
    )

    # ── Parse response ────────────────────────────────────────────────────