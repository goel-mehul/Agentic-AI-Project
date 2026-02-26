"""
evals/benchmark_vs_gpt.py
==========================
Benchmarks our multi-agent pipeline against GPT-4o one-shot generation.

Same question → both systems → same 5 eval metrics → comparison table.

Usage:
  cd backend
  python ../evals/benchmark_vs_gpt.py --question "How does RLHF work?"
  python ../evals/benchmark_vs_gpt.py --question "How does RLHF work?" --save ../evals/results/rlhf.json
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))


GPT_SYSTEM_PROMPT = """You are an expert academic research writer.
Write a comprehensive research report answering the given question.

Structure your response with these exact sections:
## Executive Summary
## Key Findings
## Contradictions & Debates
## Gaps in the Literature
## References

Include inline citations like [Author et al., Year] where possible.
Be specific and technical. Aim for 600-900 words."""


def get_gpt_report(question: str) -> str:
    """Generate a research report using GPT-4o one-shot."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("  Calling GPT-4o...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Research question: {question}"}
        ],
        max_tokens=2000,
        temperature=0.3
    )
    return response.choices[0].message.content


def run_our_pipeline(question: str) -> dict:
    """Run our multi-agent pipeline and return the final state."""
    from agents.pipeline import research_graph, create_initial_state

    print("  Running multi-agent pipeline (this takes 2-3 mins)...")
    state = create_initial_state(question)
    final_state = dict(state)

    for step in research_graph.stream(state):
        for agent, node_state in step.items():
            final_state.update(node_state)
            for log in node_state.get("agent_logs", []):
                print(f"    [{agent}] {log}")

    return final_state


def score_report(state: dict) -> dict:
    """Score a pipeline state using the 5 eval metrics."""
    import re
    from dataclasses import asdict

    report   = state.get("final_report", "")
    papers   = state.get("raw_papers", [])
    question = state.get("research_question", "")
    plan     = state.get("research_plan", [])
    chunks   = state.get("retrieved_chunks", [])
    contras  = state.get("contradictions", [])
    gaps     = state.get("gaps", [])
    quality  = state.get("evidence_quality", {})

    def grade(s):
        if s >= 0.9: return "A"
        if s >= 0.8: return "B"
        if s >= 0.7: return "C"
        if s >= 0.6: return "D"
        return "F"

    # 1. Completeness
    required = ["executive summary", "key findings", "references", "fact-check"]
    # GPT won't have fact-check section so adjust for it
    if not papers:  # GPT mock state
        required = ["executive summary", "key findings", "references", "contradictions"]
    found = [s for s in required if s in report.lower()]
    completeness = len(found) / len(required)

    # 2. Citations
    patterns = [r'\(\d{4}\)', r'\[\d{4}\]', r'et al\.', r'\[\w+ et al']
    citation_count = sum(len(re.findall(p, report, re.IGNORECASE)) for p in patterns)
    title_mentions = sum(
        1 for p in papers[:5]
        if (t := p.get("title", "")) and len(t) > 10
        and " ".join(t.split()[:4]).lower() in report.lower()
    )
    citations = min(1.0, (citation_count + title_mentions * 2) / 6)

    # 3. Coverage
    stop = {"the","a","an","is","are","what","how","why","which","in","of",
            "for","to","and","or","with","do","does","did","can","have","has"}
    q_words = {w.lower().strip("?.,") for w in question.split()
               if w.lower() not in stop and len(w) > 3}
    covered = {w for w in q_words if w in report.lower()}
    coverage = len(covered) / max(len(q_words), 1)

    # 4. Grounding
    if not chunks:
        grounding = 0.3  # GPT has no retrieved chunks — penalize fairly
    else:
        grounded = 0
        for chunk in chunks[:8]:
            content = chunk.get("content", "")
            title   = chunk.get("metadata", {}).get("title", "")
            words   = content.lower().split()
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            matches  = sum(1 for t in trigrams[:20] if t in report.lower())
            if matches > 2 or (title and title[:30].lower() in report.lower()):
                grounded += 1
        grounding = min(1.0, grounded / min(4, len(chunks)))

    # 5. Critic quality
    if not contras and not gaps and not quality:
        critic = 0.2  # GPT has no critic — penalize fairly
    else:
        s = 0.0
        if len(gaps) >= 2:       s += 0.4
        elif len(gaps) == 1:     s += 0.2
        if len(quality) >= 3:    s += 0.3
        elif quality:            s += 0.15
        if contras:              s += 0.3
        else:                    s += 0.1
        critic = min(1.0, s)

    metrics = [
        {"name": "report_completeness", "score": completeness, "grade": grade(completeness)},
        {"name": "citation_presence",   "score": citations,    "grade": grade(citations)},
        {"name": "question_coverage",   "score": coverage,     "grade": grade(coverage)},
        {"name": "evidence_grounding",  "score": grounding,    "grade": grade(grounding)},
        {"name": "critic_quality",      "score": critic,       "grade": grade(critic)},
    ]

    overall = sum(m["score"] for m in metrics) / len(metrics)

    return {
        "overall_score": round(overall, 3),
        "overall_grade": grade(overall),
        "papers_found":  len(papers),
        "iterations":    state.get("search_iteration", 1),
        "metrics":       metrics,
    }


def print_comparison(question: str, our: dict, gpt: dict):
    print(f"\n{'='*65}")
    print(f"BENCHMARK RESULTS")
    print(f"Question: {question[:70]}")
    print(f"{'='*65}")
    print(f"{'METRIC':<28} {'OURS':>12} {'GPT-4o':>12} {'WINNER':>10}")
    print(f"{'-'*65}")

    for our_m, gpt_m in zip(our["metrics"], gpt["metrics"]):
        winner = "← Ours" if our_m["score"] > gpt_m["score"] else \
                 "← GPT"  if gpt_m["score"] > our_m["score"] else "  Tie"
        print(f"{our_m['name']:<28} "
              f"{our_m['grade']} ({our_m['score']:.0%})   "
              f"{gpt_m['grade']} ({gpt_m['score']:.0%})   "
              f"{winner}")

    print(f"{'-'*65}")
    print(f"{'OVERALL':<28} "
          f"{our['overall_grade']} ({our['overall_score']:.0%})   "
          f"{gpt['overall_grade']} ({gpt['overall_score']:.0%})")
    print(f"\nOur papers retrieved: {our['papers_found']}")
    print(f"Our search passes:    {our['iterations']}")
    print(f"GPT-4o papers:        0 (answers from memory)")
    print(f"{'='*65}\n")


def benchmark(question: str, save_path: str = None):
    print(f"\n🚀 Benchmarking: {question}\n")

    # Run both systems
    print("Step 1/2: Our pipeline")
    our_state  = run_our_pipeline(question)
    our_scores = score_report(our_state)

    print("\nStep 2/2: GPT-4o")
    gpt_text = get_gpt_report(question)
    gpt_state = {
        "research_question": question,
        "final_report":      gpt_text,
        "raw_papers": [], "retrieved_chunks": [],
        "research_plan": [], "contradictions": [],
        "gaps": [], "evidence_quality": {},
        "search_iteration": 1,
    }
    gpt_scores = score_report(gpt_state)

    print_comparison(question, our_scores, gpt_scores)

    results = {
        "question":          question,
        "timestamp":         datetime.now().isoformat(),
        "our_system":        our_scores,
        "gpt4o":             gpt_scores,
        "gpt4o_report_text": gpt_text,
        "our_report_text":   our_state.get("final_report", ""),
        "summary": {
            "winner":     "our_system" if our_scores["overall_score"] > gpt_scores["overall_score"] else "gpt4o",
            "our_score":  our_scores["overall_score"],
            "gpt_score":  gpt_scores["overall_score"],
            "our_papers": our_scores["papers_found"],
            "our_iters":  our_scores["iterations"],
        }
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--save", help="Path to save JSON results")
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), "..", "backend"))
    benchmark(args.question, args.save)