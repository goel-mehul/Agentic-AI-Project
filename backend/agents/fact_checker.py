"""
fact_checker.py — The Fact Checker Agent
==========================================

WHAT IT DOES:
    Audits the Writer's draft report against the original source evidence.
    Flags overstated or unsupported claims, corrects them, and adds a
    confidence rating to the final report.

WHY IT EXISTS:
    Writers (human and AI) tend to smooth over uncertainty. A claim like
    "RLHF eliminates hallucinations" sounds cleaner than "RLHF reduces
    hallucinations by ~40% in some benchmarks." The Fact Checker enforces
    the more honest version.

HOW IT FITS:
    Fifth and final node in the pipeline.
    Reads:  draft_report (from Writer), retrieved_chunks (from Search),
            research_question
    Writes: final_report, fact_check_notes, status, agent_logs

WHAT YOU'RE LEARNING:
    - The "agent checking agent" pattern
    - How to use LLMs for verification, not just generation
    - Why confidence ratings matter in AI-generated content
    - How to finalize and annotate a pipeline's output
"""

import json
from anthropic import Anthropic
from dotenv import load_dotenv
from .state import ResearchState

load_dotenv()
client = Anthropic()


FACT_CHECKER_SYSTEM_PROMPT = """You are a meticulous fact-checker reviewing an AI-generated research report.

Your job: verify that every major claim in the report is actually supported by the provided source evidence.

Output a JSON object with exactly these fields:
- "corrected_report": the full corrected markdown report (fix issues inline, preserve structure)
- "corrections_made": list of strings describing each change you made (empty list if none needed)
- "overall_confidence": exactly one of "High", "Medium", or "Low"
- "confidence_rationale": 1-2 sentences explaining the confidence rating

Confidence guide:
- High: Most claims are well-supported by multiple recent, relevant papers
- Medium: Claims are generally supported but some gaps or dated sources exist
- Low: Limited evidence, significant gaps, or heavy reliance on weak sources

Rules:
- Only correct claims that are genuinely unsupported or overstated
- Preserve the report's structure, tone, and markdown formatting
- If a claim is supported, leave it alone — don't over-correct
- Output ONLY valid JSON, no explanation, no markdown fences"""


def fact_checker_agent(state: ResearchState) -> ResearchState:
    """
    Fact Checker Agent — fifth and final node in the LangGraph pipeline.

    Args:
        state: ResearchState with draft_report and retrieved_chunks populated

    Returns:
        Updated state with final_report, fact_check_notes, and status="complete"
    """
    draft    = state["draft_report"]
    chunks   = state["retrieved_chunks"]
    question = state["research_question"]

    state["current_agent"] = "fact_checker"
    state["agent_logs"]    = ["🔎 Fact Checker: Auditing report against source evidence..."]

    # Edge case: no draft to check
    if not draft:
        state["final_report"]     = "Error: No draft report was produced by the Writer."
        state["fact_check_notes"] = ["Draft report was empty — nothing to verify."]
        state["status"]           = "error"
        return state

    # ── Format source evidence for verification ───────────────────────────
    # We give the Fact Checker the same evidence the Writer had,
    # so it can check claims against the actual source material.

    source_text = "\n\n---\n\n".join([
        f"Source: {c['metadata'].get('title', 'Unknown')} "
        f"({c['metadata'].get('published', '?')})\n"
        f"Authors: {c['metadata'].get('authors', 'Unknown')}\n\n"
        f"{c['content'][:600]}"
        for c in chunks[:6]  # Top 6 most relevant sources
    ])

    # ── Call Claude ───────────────────────────────────────────────────────
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku sufficient for verification
        max_tokens=4000,
        system=FACT_CHECKER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research question: {question}\n\n"
                    f"# Draft Report to Fact-Check\n\n{draft}\n\n"
                    f"# Source Evidence\n\n{source_text}\n\n"
                    f"Please fact-check the report."
                )
            }
        ]
    )

    # ── Parse response ────────────────────────────────────────────────────
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    corrected_report = result.get("corrected_report", draft)
    corrections      = result.get("corrections_made", [])
    confidence       = result.get("overall_confidence", "Medium")
    rationale        = result.get("confidence_rationale", "")

    # ── Append fact-check summary to the bottom of the report ────────────
    fact_check_footer = (
        f"\n\n---\n\n"
        f"## ✅ Fact-Check Report\n\n"
        f"**Overall Confidence:** {confidence}\n\n"
        f"**Rationale:** {rationale}\n\n"
    )

    if corrections:
        fact_check_footer += (
            "**Corrections Made:**\n" +
            "\n".join(f"- {c}" for c in corrections)
        )
    else:
        fact_check_footer += (
            "**Corrections Made:** None — all claims verified against sources."
        )

    # ── Write final outputs ───────────────────────────────────────────────
    state["final_report"]     = corrected_report + fact_check_footer
    state["fact_check_notes"] = corrections
    state["status"]           = "complete"  # Pipeline is done!

    n = len(corrections)
    state["agent_logs"] = [
        f"✅ Fact Checker: Audit complete — {n} correction(s) made",
        f"📊 Overall confidence: {confidence}",
        f"💡 {rationale}",
        "🎉 Pipeline complete! Final report ready."
    ]

    return state