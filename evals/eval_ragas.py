"""
eval_ragas.py — RAGAs evaluation for the research pipeline
===========================================================

RAGAs is the industry standard framework for evaluating RAG pipelines.
Measures: faithfulness, answer relevancy, context precision, context recall

Run from project root:
python evals/eval_ragas.py --question "How does RLHF work?"

Requirements:
pip install ragas langchain-anthropic langchain-huggingface sentence-transformers
"""

import sys
import os
import argparse
import uuid
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.planner import planner_agent
from agents.search import search_agent
from agents.critic import critic_agent
from agents.writer import writer_agent
from agents.fact_checker import fact_checker_agent
from agents.state import ResearchState

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def make_state(question: str) -> ResearchState:
    return ResearchState(
        research_question=question,
        session_id=str(uuid.uuid4()),
        research_plan=[], search_strategy="",
        raw_papers=[], retrieved_chunks=[],
        citation_counts={},
        evidence_quality={}, contradictions=[], gaps=[],
        search_iteration=0, gap_queries=[],
        draft_report="", report_sections={},
        final_report="", fact_check_notes=[],
        faithfulness_scores={},
        agent_logs=[], current_agent="",
        status="running", error=""
    )


def run_pipeline(question: str) -> dict:
    print(f"Running pipeline for: {question}\n")
    state = make_state(question)

    print("1/5 Planner...")
    state = planner_agent(state)

    print("2/5 Search...")
    state = search_agent(state)

    print("3/5 Critic...")
    state = critic_agent(state)

    print("4/5 Writer...")
    state = writer_agent(state)

    print("5/5 Fact Checker...")
    state = fact_checker_agent(state)

    return state


def run_ragas_eval(question: str, save_path: str = None):
    state = run_pipeline(question)

    # Prepare RAGAs inputs
    contexts = [
        chunk['content']
        for chunk in state['retrieved_chunks']
    ]

    data = {
        "question":     [question],
        "answer":       [state['final_report']],
        "contexts":     [contexts],
        "ground_truth": [question]
    }

    dataset = Dataset.from_dict(data)

    print("\nConfiguring RAGAs with Claude Haiku as judge...\n")

    # Use Claude Haiku as the judge LLM
    claude_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=4096
        )
    )

    # Use HuggingFace embeddings locally — no OpenAI needed
    hf_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    )

    # Initialize metrics with Claude as judge
    metrics = [
        Faithfulness(llm=claude_llm),
        AnswerRelevancy(llm=claude_llm, embeddings=hf_embeddings),
        ContextPrecision(llm=claude_llm),
        ContextRecall(llm=claude_llm),
    ]

    print("Running RAGAs evaluation...\n")

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=claude_llm,
        embeddings=hf_embeddings
    )

    print("=" * 60)
    print("RAGAs EVALUATION RESULTS")
    print("=" * 60)

    # Print results safely regardless of format
    try:
        df = result.to_pandas()
        for col in ['faithfulness', 'answer_relevancy',
                    'context_precision', 'context_recall']:
            if col in df.columns:
                val = df[col].iloc[0]
                print(f"{col:25s}: {val:.3f}")
    except Exception:
        print(result)

    print("=" * 60)

    # Build output dict safely
    ragas_scores = {}
    try:
        df = result.to_pandas()
        ragas_scores = df.to_dict(orient='records')[0]
    except Exception:
        ragas_scores = str(result)

    output = {
        "question": question,
        "ragas_scores": ragas_scores,
        "pipeline_faithfulness_avg": (
            sum(state['faithfulness_scores'].values()) /
            len(state['faithfulness_scores'])
            if state['faithfulness_scores'] else 0
        ),
        "papers_found":      len(state['raw_papers']),
        "search_iterations": state['search_iteration'],
        "corrections_made":  len(state['fact_check_notes'])
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {save_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', required=True)
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    run_ragas_eval(args.question, args.save)