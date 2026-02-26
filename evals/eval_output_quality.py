"""
evals/eval_output_quality.py
============================
Quantitative evaluation of research pipeline output quality.

Metrics:
  1. report_completeness  — All required sections present?
  2. citation_presence    — Claims backed by citations?
  3. question_coverage    — Report addresses the original question?
  4. evidence_grounding   — Content grounded in retrieved papers?
  5. critic_quality       — Did the Critic do meaningful work?

Usage:
  python evals/eval_output_quality.py --question "How does RLHF work?"
  python evals/eval_output_quality.py --save-results results/rlhf.json
"""

import re
import json
import argparse
import sys
import os
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ── Data class ──────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    name: str
    score: float        # 0.0 – 1.0
    grade: str          # A/B/C/D/F
    passed: bool
    threshold: float
    details: str


def _grade(score: float) -> str:
    if score >= 0.9: return "A"
    if score >= 0.8: return "B"
    if score >= 0.7: return "C"
    if score >= 0.6: return "D"
    return "F"


# ── Metric 1: Report Completeness ────────────────────────────────────────────

def eval_completeness(report: str) -> MetricResult:
    """All 4 required sections must be present."""
    required = ["executive summary", "key findings", "references", "fact-check"]
    found    = [s for s in required if s in report.lower()]
    score    = len(found) / len(required)
    missing  = [s for s in required if s not in report.lower()]

    return MetricResult(
        name="report_completeness",
        score=score, grade=_grade(score),
        passed=score >= 0.75, threshold=0.75,
        details=f"{len(found)}/{len(required)} sections present."
                + (f" Missing: {missing}" if missing else "")
    )


# ── Metric 2: Citation Presence ──────────────────────────────────────────────

def eval_citations(report: str, papers: list) -> MetricResult:
    """Count citation patterns and paper title mentions."""
    patterns = [r'\(\d{4}\)', r'\[\d{4}\]', r'et al\.', r'\[\w+ et al']
    citation_count = sum(len(re.findall(p, report, re.IGNORECASE)) for p in patterns)

    title_mentions = sum(
        1 for paper in papers[:5]
        if (t := paper.get("title", "")) and len(t) > 10
        and " ".join(t.split()[:4]).lower() in report.lower()
    )

    score = min(1.0, (citation_count + title_mentions * 2) / 6)

    return MetricResult(
        name="citation_presence",
        score=score, grade=_grade(score),
        passed=score >= 0.5, threshold=0.5,
        details=f"{citation_count} citation patterns, {title_mentions} title mentions found."
    )


# ── Metric 3: Question Coverage ──────────────────────────────────────────────

def eval_coverage(report: str, question: str, plan: list) -> MetricResult:
    """Key terms from the question should appear in the report."""
    stop = {"the","a","an","is","are","what","how","why","which","in","of","for",
            "to","and","or","with","do","does","did","can","have","has"}
    q_words = {w.lower().strip("?.,") for w in question.split()
               if w.lower() not in stop and len(w) > 3}

    report_lower = report.lower()
    covered = {w for w in q_words if w in report_lower}
    keyword_score = len(covered) / max(len(q_words), 1)

    plan_score = 0.5
    if plan:
        plan_hits = sum(
            1 for q in plan
            if len({w.lower() for w in q.split() if w.lower() not in stop}
                   & set(report_lower.split())) /
               max(len({w for w in q.split() if len(w) > 3}), 1) > 0.4
        )
        plan_score = plan_hits / len(plan)

    score = 0.6 * keyword_score + 0.4 * plan_score

    return MetricResult(
        name="question_coverage",
        score=score, grade=_grade(score),
        passed=score >= 0.6, threshold=0.6,
        details=f"{len(covered)}/{len(q_words)} question keywords found in report."
    )


# ── Metric 4: Evidence Grounding ─────────────────────────────────────────────

def eval_grounding(report: str, chunks: list) -> MetricResult:
    """Check how many retrieved sources are actually reflected in the report."""
    if not chunks:
        return MetricResult("evidence_grounding", 0.0, "F", False, 0.4,
                            "No retrieved chunks available.")

    grounded = 0
    report_lower = report.lower()

    for chunk in chunks[:8]:
        content = chunk.get("content", "")
        title   = chunk.get("metadata", {}).get("title", "")
        words   = content.lower().split()
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        matches  = sum(1 for t in trigrams[:20] if t in report_lower)

        if matches > 2 or (title and title[:30].lower() in report_lower):
            grounded += 1

    score = min(1.0, grounded / min(4, len(chunks)))

    return MetricResult(
        name="evidence_grounding",
        score=score, grade=_grade(score),
        passed=score >= 0.4, threshold=0.4,
        details=f"Report grounded in {grounded}/{min(8, len(chunks))} retrieved sources."
    )


# ── Metric 5: Critic Quality ─────────────────────────────────────────────────

def eval_critic(contradictions: list, gaps: list, quality: dict) -> MetricResult:
    """Did the Critic do substantive work — not just rubber-stamp?"""
    score, notes = 0.0, []

    if len(gaps) >= 2:
        score += 0.4; notes.append(f"{len(gaps)} gaps identified ✓")
    elif len(gaps) == 1:
        score += 0.2; notes.append("Only 1 gap found")
    else:
        notes.append("No gaps — suspicious")

    if len(quality) >= 3:
        score += 0.3; notes.append(f"Quality scores for {len(quality)} papers ✓")
    elif quality:
        score += 0.15; notes.append(f"Quality scores for only {len(quality)} papers")

    if contradictions:
        score += 0.3; notes.append(f"{len(contradictions)} contradiction(s) found ✓")
    else:
        score += 0.1; notes.append("No contradictions (may be fine for some topics)")

    score = min(1.0, score)

    return MetricResult(
        name="critic_quality",
        score=score, grade=_grade(score),
        passed=score >= 0.5, threshold=0.5,
        details=" | ".join(notes)
    )


# ── Full Evaluation Suite ─────────────────────────────────────────────────────

def evaluate(state: dict) -> dict:
    """Run all 5 metrics on a completed pipeline state. Returns full report."""
    results = [
        eval_completeness(state.get("final_report", "")),
        eval_citations(state.get("final_report", ""), state.get("raw_papers", [])),
        eval_coverage(
            state.get("final_report", ""),
            state.get("research_question", ""),
            state.get("research_plan", [])
        ),
        eval_grounding(state.get("final_report", ""), state.get("retrieved_chunks", [])),
        eval_critic(
            state.get("contradictions", []),
            state.get("gaps", []),
            state.get("evidence_quality", {})
        ),
    ]

    overall = sum(r.score for r in results) / len(results)
    passed  = sum(1 for r in results if r.passed)

    return {
        "question":      state.get("research_question", ""),
        "timestamp":     datetime.now().isoformat(),
        "overall_score": round(overall, 3),
        "overall_grade": _grade(overall),
        "passed":        passed,
        "total":         len(results),
        "papers_found":  len(state.get("raw_papers", [])),
        "iterations":    state.get("search_iteration", 1),
        "metrics":       [asdict(r) for r in results],
    }


def print_report(report: dict):
    """Pretty-print the evaluation report to stdout."""
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Question:  {report['question'][:80]}")
    print(f"Papers:    {report['papers_found']}")
    print(f"Iterations:{report['iterations']} search passes")
    print(f"\nOverall:   {report['overall_grade']}  ({report['overall_score']:.1%})")
    print(f"Passed:    {report['passed']}/{report['total']} metrics\n")

    for m in report["metrics"]:
        status = "✅" if m["passed"] else "❌"
        print(f"  {status} {m['name']:<25} {m['grade']}  ({m['score']:.1%})")
        print(f"     {m['details']}")
    print(f"{'='*60}\n")


# ── CLI Runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate research pipeline output quality")
    parser.add_argument("--question", required=True, help="Research question to evaluate")
    parser.add_argument("--save", help="Path to save JSON results (optional)")
    args = parser.parse_args()

    print(f"\n🚀 Running pipeline for: {args.question}")
    print("This will take 2-3 minutes...\n")

    from agents.pipeline import research_graph, create_initial_state
    state = create_initial_state(args.question)

    # Run the pipeline
    final_state = dict(state)
    for step in research_graph.stream(state):
        for agent, node_state in step.items():
            final_state.update(node_state)
            for log in node_state.get("agent_logs", []):
                print(f"  [{agent}] {log}")

    # Score the output
    report = evaluate(final_state)
    print_report(report)

    # Optionally save results
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Results saved to {args.save}")