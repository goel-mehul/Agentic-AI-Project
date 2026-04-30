"""
writer.py — The Writer Agent
=============================

WHAT IT DOES:
    Synthesizes all retrieved evidence into a structured research report.
    Critically: it uses the Critic's assessment to write a NUANCED report
    that caveats weak evidence and explicitly notes contradictions/gaps.

WHY IT EXISTS:
    A report that treats all evidence equally is misleading. The Writer
    knows what the Critic found and writes accordingly — flagging contested
    claims, noting gaps, and being honest about evidence quality.

HOW IT FITS:
    Fourth node in the pipeline.
    Reads:  retrieved_chunks, evidence_quality, contradictions, gaps,
            search_strategy, research_question (all from previous agents)
    Writes: draft_report, report_sections, agent_logs

WHAT YOU'RE LEARNING:
    - How to pass rich context to an LLM for synthesis
    - Why model selection matters (Sonnet vs Haiku)
    - How agents build on each other's outputs
    - Structured report generation with markdown
"""

import json
from anthropic import Anthropic
from dotenv import load_dotenv
from .state import ResearchState

load_dotenv()
client = Anthropic()


WRITER_SYSTEM_PROMPT = """You are an expert academic science writer producing research synthesis reports.

Your reports are used by graduate students and researchers who need accurate, well-sourced summaries.

Write in Markdown format with these exact sections:
1. ## Executive Summary (3-4 sentences capturing the key answer)
2. ## Key Findings (organized thematically, with inline citations like [Author et al., Year])
3. ## Contradictions & Debates (only if contradictions exist — be specific)
4. ## Gaps in the Literature (what this evidence does NOT cover)
5. ## Methodology Notes (brief notes on evidence quality and limitations)
6. ## References (numbered list of all papers cited)

Critical rules:
- Be specific and technical, not vague
- Every major claim needs an inline citation
- If evidence quality is low for a claim, caveat it explicitly
- If two papers disagree, present BOTH views — never pick one without justification
- Aim for 600-900 words in the main body
- Never fabricate citations — only cite papers actually in the evidence

GROUNDING RULES:
- Base every specific claim on the provided excerpts
- If a specific detail (metric, percentage, method name) is not explicitly
  stated in the excerpts, do not include it
- You may use your knowledge to contextualize and connect findings, but all
  specific factual claims must trace back to a provided excerpt
- If evidence on a topic is limited or absent, say so explicitly rather than
  filling gaps with general knowledge

WHEN EVIDENCE IS INSUFFICIENT:
- Do not fabricate claims to fill gaps
- Explicitly note in the Gaps section when important aspects of the question
  are not addressed by the retrieved evidence
- For any claim you are uncertain about, add a qualifier such as
  'according to the available abstracts' or 'based on the provided evidence'"""


def writer_agent(state: ResearchState) -> ResearchState:
    """
    Writer Agent — fourth node in the LangGraph pipeline.

    Args:
        state: ResearchState with chunks, critique, and question populated

    Returns:
        Updated state with draft_report and report_sections populated
    """
    question         = state["research_question"]
    chunks           = state["retrieved_chunks"]
    evidence_quality = state["evidence_quality"]
    contradictions   = state["contradictions"]
    gaps             = state["gaps"]
    strategy         = state.get("search_strategy", "")

    state["current_agent"] = "writer"
    state["agent_logs"]    = ["✍️ Writer: Synthesizing evidence into report..."]

    # ── Build rich context ────────────────────────────────────────────────
    # We give the Writer everything it needs in a structured prompt.
    # This is the key to getting a nuanced, well-grounded report.

    # Sort chunks by faithfulness score — lost in the middle fix
    # Best chunk first, second best last, weakest in the middle
    chunks_sorted = sorted(
        chunks,
        key=lambda x: x.get('faithfulness_score', 0),
        reverse=True
    )
    if len(chunks_sorted) > 1:
        best        = chunks_sorted[0]
        remaining   = chunks_sorted[1:]
        second_best = remaining.pop(0)
        chunks_sorted = [best] + remaining + [second_best]

    # Format evidence with strong separators and faithfulness scores
    evidence_sections = []
    for i, chunk in enumerate(chunks_sorted[:8]):
        title     = chunk['metadata'].get('title', 'Unknown')
        published = chunk['metadata'].get('published', 'Unknown')
        authors   = chunk['metadata'].get('authors', 'Unknown')
        source    = chunk['metadata'].get('source', 'Unknown')
        url       = chunk['metadata'].get('url', '')
        score     = chunk.get('faithfulness_score', 'N/A')

        section = (
            f"=== EXCERPT {i+1}: {title} ({published}) ===\n"
            f"Authors: {authors} | Source: {source} | "
            f"Faithfulness score: {score}\n"
            f"URL: {url}\n\n"
            f"{chunk['content'][:1000]}"
        )
        evidence_sections.append(section)

    evidence_text = "\n\n".join(evidence_sections)

    # Format critic outputs so Writer knows what to caveat
    quality_text = json.dumps(evidence_quality, indent=2) if evidence_quality else "Not available"

    contradictions_text = (
        "\n".join(f"- {c}" for c in contradictions)
        if contradictions else "None identified by the Critic."
    )

    gaps_text = (
        "\n".join(f"- {g}" for g in gaps)
        if gaps else "None identified by the Critic."
    )

    # ── Call Claude Sonnet ────────────────────────────────────────────────
    # We use Sonnet here (not Haiku) because writing quality matters more
    # than speed for the final report. This is a deliberate model choice.

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        temperature=0.7,
        system=WRITER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"# Research Question\n{question}\n\n"
                    f"# Search Strategy Used\n{strategy}\n\n"
                    f"# Evidence Quality (from Critic Agent)\n{quality_text}\n\n"
                    f"# Contradictions Found\n{contradictions_text}\n\n"
                    f"# Literature Gaps\n{gaps_text}\n\n"
                    f"# Retrieved Evidence\n\n{evidence_text}\n\n"
                    f"Please write the research synthesis report now."
                )
            }
        ]
    )

    draft = response.content[0].text.strip()
    state["draft_report"] = draft

    # ── Parse report into sections ────────────────────────────────────────
    # Store each section separately so the Fact Checker and frontend
    # can access individual sections without parsing the full markdown.

    sections = {}
    current_section = "preamble"
    current_lines = []

    for line in draft.split("\n"):
        if line.startswith("## "):
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    state["report_sections"] = sections

    word_count = len(draft.split())
    state["agent_logs"] = [
        f"✅ Writer: Report complete ({word_count} words, {len(sections)} sections)",
        "📝 Passing to Fact Checker for verification..."
    ]

    return state