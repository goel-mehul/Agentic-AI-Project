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

from anthropic import Anthropic
from dotenv import load_dotenv
from .state import ResearchState

load_dotenv()
client = Anthropic()


# ── System Prompt ─────────────────────────────────────────────────────────────
# Notice this prompt is completely different from the Planner's.
# This agent is a skeptic — its job is to find problems, not solutions.

CRITIC_SYSTEM_PROMPT = """You are a rigorous academic peer reviewer with high standards.

Your job: critically evaluate a set of research papers retrieved for a given question by calling the submit_critique tool with these fields:

- quality_scores: mapping of paper titles to scores 0.0-1.0 with brief rationale. Papers with 100+ citations should receive a quality boost of up to 0.1.
- contradictions: list of strings describing conflicting findings between papers (empty list if none)
- gaps: list of strings describing important aspects of the question NOT covered by the evidence. Be specific — name the missing technique categories, domains, or comparisons.
- high_quality_papers: list of 3-5 paper titles that are most reliable and relevant
- summary: 2-3 sentences on overall evidence quality — which papers are most relevant, what is covered well, and what critical gaps remain.

Be intellectually honest. Finding weaknesses makes the final report MORE credible, not less.
If your summary identifies limitations, those limitations must appear in the gaps field."""

CRITIC_TOOL = {
    "name": "submit_critique",
    "description": "Submit a structured critical evaluation of the retrieved research evidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "quality_scores": {
                "type": "object",
                "description": "Dict mapping paper titles to objects with score (0.0-1.0) and rationale. Papers with 100+ citations should receive a quality boost of up to 0.1.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "score":     {"type": "number"},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"]
                }
            },
            "contradictions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conflicting findings between papers. Empty list if none."
            },
            "gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Important aspects of the question NOT covered by the evidence. Be thorough — even well-covered topics have gaps. Consider: missing technique categories, lack of empirical comparisons, underrepresented domains, missing failure mode analysis, or absence of scalability discussion. Only return empty list if evidence is truly comprehensive."
            },
            "high_quality_papers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 paper titles that are most reliable and relevant."
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentences on overall evidence quality. This field is required and must not be empty. Describe which papers are most relevant, what the collection covers well, and what critical gaps remain."
            },
        },
        "required": ["quality_scores", "contradictions", "gaps", "high_quality_papers", "summary"]
    }
}

GAP_QUERY_TOOL = {
    "name": "generate_gap_queries",
    "description": "Convert evidence gaps into precise academic search queries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 short, specific search queries (4-7 words each) suitable for arXiv and Semantic Scholar. Each query should directly target one of the identified gaps using academic terminology."
            }
        },
        "required": ["queries"]
    }
}

GAP_QUERY_SYSTEM_PROMPT = """You are an expert academic search specialist.

Your job: convert descriptions of evidence gaps into precise, short search queries
suitable for arXiv and Semantic Scholar. Call the generate_gap_queries tool.

Rules:
- Each query must be 4-7 words
- Use academic/technical terminology
- Each query should target a different gap
- Queries should be specific enough to find relevant papers"""


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
    # Get citation counts from state

    citation_counts = state.get("citation_counts", {})

    evidence_text = "\n\n---\n\n".join([
        f"Paper: {c['metadata'].get('title', 'Unknown')}\n"
        f"Authors: {c['metadata'].get('authors', 'Unknown')}\n"
        f"Published: {c['metadata'].get('published', 'Unknown')}\n"
        f"Source: {c['metadata'].get('source', 'Unknown')}\n\n"
        f"Citations: {citation_counts.get(c['metadata'].get('paper_id', ''), 'Unknown')}\n\n"
        f"{c['content'][:800]}"
        for c in chunks[:8]  # Cap at 8 to control token usage
    ])

    # ── Call Claude ───────────────────────────────────────────────────────
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        temperature=0.2,
        system=CRITIC_SYSTEM_PROMPT,
        tools=[CRITIC_TOOL],
        tool_choice={"type": "any"},
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

    tool_block = next(b for b in response.content if b.type == "tool_use")
    critique   = tool_block.input

    state["evidence_quality"] = critique.get("quality_scores", {})
    state["contradictions"]   = critique.get("contradictions", [])
    state["gaps"]             = critique.get("gaps", [])

    # NEW: dedicated Haiku call to generate gap queries instead of string slicing
    gap_queries = []
    if state["gaps"]:
        gaps_text = "\n".join(f"- {g}" for g in state["gaps"][:3])

        gap_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            temperature=0.2,
            system=GAP_QUERY_SYSTEM_PROMPT,
            tools=[GAP_QUERY_TOOL],
            tool_choice={"type": "any"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Research question: {question}\n\n"
                        f"Evidence gaps identified:\n{gaps_text}\n\n"
                        f"Generate precise search queries to fill these gaps."
                    )
                }
            ]
        )

        gap_tool_block = next(b for b in gap_response.content if b.type == "tool_use")
        gap_queries    = gap_tool_block.input.get("queries", [])

    state["gap_queries"] = gap_queries

    n_papers        = len(state["evidence_quality"])
    n_contradictions = len(state["contradictions"])
    n_gaps           = len(state["gaps"])
    summary          = critique.get("summary", "")
    if not summary:
        n_papers = len(critique.get("quality_scores", {}))
        summary = f"Evaluated {n_papers} papers. Found {n_gaps} gap(s) in evidence coverage."

    logs = [
        f"✅ Critic: Evaluation complete",
        f"⚡ Found {n_contradictions} contradiction(s), {n_gaps} gap(s)",
    ]

    for c in state["contradictions"][:2]:
        logs.append(f"⚡ {c}")

    for g in state["gaps"][:3]:
        logs.append(f"🕳️  {g}")

    logs.append(f"🎯 Generated {len(gap_queries)} gap-filling search queries")
    logs.append(f"📊 Summary: {summary}")

    state["agent_logs"] = logs

    return state