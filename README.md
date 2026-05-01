# Multi-Agent Research Scientist

An autonomous AI research pipeline that takes a research question and produces a verified, cited research report — without any human involvement between input and output.

**Live demo:** [agentic-ai-project-mehul-goel.vercel.app](https://agentic-ai-project-mehul-goel.vercel.app)

Built to demonstrate genuine understanding of agentic AI system design: multi-agent orchestration, iterative retrieval loops, RAG pipelines, real-time streaming, and full-stack deployment.

---

## What It Does

Submit a research question. Five specialized AI agents handle everything else.

```
"What are the most effective techniques for reducing hallucinations in LLMs?"
```

And it will:

1. **Plan** a targeted search strategy with 4-6 academic search queries
2. **Search** arXiv and Semantic Scholar for relevant papers (no API keys needed)
3. **Evaluate** the evidence — scoring quality (weighted by citation count), finding contradictions, identifying gaps
4. **Loop back** to Search with gap-targeted queries if critical gaps are found (up to 3 passes)
5. **Write** a structured research report with inline citations
6. **Fact-check** every claim against the source evidence
7. **Deliver** a verified, confidence-rated report with a PDF download option

All steps happen automatically, in 2-4 minutes depending on how many retrieval passes are needed.

---

## Architecture

The system is a **LangGraph StateGraph** — five specialized AI agents connected by directed edges, including a **conditional loop** that routes back to Search when the Critic identifies critical gaps in the evidence.

```
User Question
      │
      ▼
┌─────────────┐
│   Planner   │  Turns your question into 4-6 optimized academic search queries
└──────┬──────┘
       │
       ▼
┌─────────────┐  ◄─────────────────────────────────┐
│   Search    │  Fetches papers from arXiv +        │
│             │  Semantic Scholar (with citation    │
│             │  counts). Stores in ChromaDB →      │
│             │  retrieves top-8 by semantic        │
│             │  similarity. On loop passes, uses   │
│             │  gap-targeted queries instead.      │
└──────┬──────┘                                     │
       │                                            │ Loop back if
       ▼                                            │ gaps ≥ 2 and
┌─────────────┐                                     │ iterations < 3
│   Critic    │  Scores evidence quality (boosting  │
│             │  highly-cited papers), flags        │
│             │  contradictions, identifies gaps,   │
│             │  generates gap-filling queries ─────┘
└──────┬──────┘
       │  (proceed when gaps < 2, or max passes reached)
       ▼
┌─────────────┐
│   Writer    │  Synthesizes structured markdown report with citations (Sonnet)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Fact Checker   │  Verifies every claim · adds confidence rating · corrects overstatements
└─────────────────┘
       │
       ▼
  Final Report
```

Each agent has one job, one system prompt, and writes to its own fields in a shared `ResearchState` object. No agent knows what the others do internally.

### Key Design Decisions

**Why the iterative retrieval loop?**
The Critic identifies specific gaps in the evidence after each search pass. Instead of just flagging them, the pipeline routes back to the Search agent with new queries targeting those exact gaps. LangGraph's conditional edges make this a clean architectural decision — `should_search_again()` checks iteration count, gap count, and paper count to decide whether to loop or proceed. This is what makes the system genuinely agentic rather than a linear script.

**Why citation-weighted scoring?**
Semantic Scholar returns citation counts for free. A paper cited 400 times has been validated by the community over years — that's a meaningful quality signal. The Critic uses citation counts as a secondary quality score alongside content relevance, giving established literature appropriate weight without ignoring newer work.

**Why 5 separate agents instead of 1?**
Each agent has a single, well-defined job. This makes each one independently testable, replaceable, and improvable. The Critic can be made more rigorous without touching the Writer. The Search agent can be swapped for a different data source without affecting anything else.

**Why LangGraph?**
LangGraph manages state passing between agents, supports streaming (so the frontend gets real-time updates), and makes conditional routing between agents a first-class feature. The iterative retrieval loop is implemented as a single conditional edge — 10 lines of code for a significant architectural capability.

**Why ChromaDB?**
Instead of feeding all 20-40 retrieved papers to the Writer (expensive, noisy), ChromaDB converts each abstract into a vector embedding and retrieves only the most semantically relevant chunks. This is the RAG (Retrieval-Augmented Generation) pattern used in production AI systems.

**Why two different Claude models?**
The Planner, Critic, and Fact Checker use Claude Haiku (fast, cheap) because their tasks are structured and mechanical. The Writer uses Claude Sonnet (higher quality) because synthesis quality directly affects the output the user sees. This is a deliberate cost/quality tradeoff — roughly 10x cost difference between the models.

**Why does the Fact Checker exist?**
Writers (human and AI) tend to smooth over uncertainty. Having a separate agent verify the Writer's claims against source evidence catches overstatements and adds a confidence rating. This "agent checking agent" pattern is fundamental to building reliable agentic systems.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph |
| LLM | Anthropic Claude (Haiku + Sonnet) |
| Paper Retrieval | arXiv API, Semantic Scholar API |
| Vector Store | ChromaDB |
| Backend API | FastAPI + WebSockets |
| Frontend | React + Vite |
| Testing | pytest (44 tests) |
| Backend Hosting | Railway |
| Frontend Hosting | Vercel |

---

## Project Structure

```
research-agent/
├── backend/
│   ├── agents/
│   │   ├── state.py          # Shared ResearchState TypedDict
│   │   ├── planner.py        # Agent 1 — query decomposition
│   │   ├── search.py         # Agent 2 — arXiv + Semantic Scholar + ChromaDB RAG
│   │   │                     #           handles first pass and gap-filling passes
│   │   ├── critic.py         # Agent 3 — evidence quality + citation weighting
│   │   │                     #           generates gap_queries for retrieval loop
│   │   ├── writer.py         # Agent 4 — report synthesis (Sonnet)
│   │   ├── fact_checker.py   # Agent 5 — claim verification + confidence rating
│   │   └── pipeline.py       # LangGraph StateGraph with conditional retrieval loop
│   └── main.py               # FastAPI + WebSocket streaming server
├── frontend/
│   └── src/
│       ├── App.jsx           # React app — shows search passes, gap-filling badge
│       ├── agents.js         # Agent metadata (single source of truth)
│       └── components.css    # All styles
├── evals/
│   ├── test_planner.py
│   ├── test_search.py
│   ├── test_critic.py
│   ├── test_writer.py
│   ├── test_fact_checker.py
│   ├── test_pipeline.py
│   ├── test_api.py
│   ├── eval_output_quality.py   # 5-metric quantitative evaluation script
│   ├── benchmark_vs_gpt.py      # GPT-4o comparison benchmark
│   └── results/                 # Saved evaluation JSON outputs
└── docs/
    ├── setup.md
    └── project-info/
        ├── PROJECT_DEEP_DIVE.md
        └── INTERVIEW_PREP.md
```

---

## Running Locally

**Prerequisites:** Python 3.11+, Node.js 18+, Anthropic API key

```bash
# Clone and set up Python environment
git clone https://github.com/goel-mehul/Agentic-AI-Project.git
cd research-agent
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Add your API key
cp backend/.env.example backend/.env
# Edit backend/.env and add ANTHROPIC_API_KEY=...

# Terminal 1 — backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/research` | Start a research session → returns `session_id` |
| `GET` | `/research/{id}` | Poll for completed results |
| `WS` | `/ws/{id}` | Stream real-time agent updates |

Interactive docs at `http://localhost:8000/docs`

---

## Tests

```bash
python -m pytest evals/ -v
```

44 tests across every layer. Each agent has unit tests with mock data (fast, no API calls). Search agent tests hit real arXiv. API tests use FastAPI's TestClient.

---

## Evaluation Framework

The project includes a quantitative evaluation suite that scores any completed report on 5 metrics:

| Metric | What It Checks | Threshold |
|--------|----------------|-----------|
| Report Completeness | All required sections present | ≥ 75% |
| Citation Presence | Claims backed by references | ≥ 50% |
| Question Coverage | Report addresses the question | ≥ 60% |
| Evidence Grounding | Content grounded in retrieved papers | ≥ 40% |
| Critic Quality | Depth of gap and contradiction analysis | ≥ 50% |

Run on any question:
```bash
cd backend
python ../evals/eval_output_quality.py \
  --question "How does RLHF work?" \
  --save ../evals/results/rlhf.json
```

### GPT-4o Benchmark

The same 5 metrics can be run against a GPT-4o one-shot response for comparison:

```bash
python ../evals/benchmark_vs_gpt.py \
  --question "How does RLHF work?" \
  --save ../evals/results/benchmark_rlhf.json
```

Our pipeline scores higher on citation presence and evidence grounding (retrieves real papers with verifiable citations). GPT-4o scores comparably on question coverage (broader parametric knowledge). The benchmark is honest about what each approach is good at.

---

## Cost

Using Claude Haiku for most agents and Sonnet only for the Writer:

| Usage | Estimated Cost |
|-------|----------------|
| Per research run (1 search pass) | ~$0.03 – $0.06 |
| Per research run (3 search passes) | ~$0.08 – $0.15 |
| 100 research runs | ~$5 – $15 |
| Full development | ~$10 – $20 total |

---

## What I Learned Building This

Built incrementally over multiple development phases, each committed separately, to demonstrate genuine development process rather than a single code dump (~90 commits total).

Key concepts practiced:
- **Agentic AI design** — decomposing complex tasks into specialized agents with single responsibilities
- **Conditional routing** — LangGraph conditional edges for iterative retrieval loops
- **Citation-weighted evidence** — using external quality signals (citation counts) in LLM prompts
- **RAG pipelines** — vector embeddings, semantic retrieval, context management
- **Async Python** — FastAPI, WebSockets, running blocking LangGraph in thread pools
- **Multi-agent verification** — having agents check each other's work
- **Evaluation-driven development** — writing quantitative metrics and benchmarks before calling something "working"
- **Production deployment** — Railway + Vercel, environment management, handling API rate limits

---

## V4 Improvements

A significant update to the pipeline was made in v4, addressing core architectural 
weaknesses identified during code review and adding new capabilities.

---

### 1. Structured Output via Claude's `tool_use` API

**Problem:** All three structured-output agents (Planner, Critic, Fact Checker) were 
prompting Claude to "output only valid JSON" and then manually parsing the response — 
stripping markdown fences, calling bare `json.loads()`, and hoping the output was 
well-formed. This approach was fragile: any truncation or formatting variation caused 
an unhandled crash.

**Fix:** Replaced manual JSON parsing with Claude's `tool_use` API across all three 
agents. Each agent now defines an explicit output schema as a tool definition. Claude 
is forced to call the tool with validated, structured arguments — the response arrives 
as a Python dict with no parsing required.

**Result:** Eliminates all bare `json.loads()` calls, all markdown fence stripping, 
and all crash risk from malformed output. Also fixed a `ddict` typo in the Critic's 
original system prompt that was sending a malformed instruction to the model.

---

### 2. Improved Gap Query Generation

**Problem:** When the Critic identified evidence gaps, it generated gap-filling search 
queries by taking the first 8 words of each gap description string. This produced 
useless queries like *"The evidence does not sufficiently cover the"* — essentially 
stop words with no search value.

**Fix:** Replaced the string-slicing approach with a dedicated Claude Haiku call using 
the `tool_use` API. The Critic now passes its gap descriptions to a second Haiku call 
with a `generate_gap_queries` tool schema, which returns 2-3 precise, academic 
4-7 word search queries directly targeting each gap.

**Result:** Gap-filling queries are now meaningful academic search terms that actually 
find relevant papers on subsequent loop passes.

---

### 3. Cosine Similarity Faithfulness Scoring

**Problem:** ChromaDB was computing cosine similarity between retrieved chunks and the 
research question during retrieval but the scores were being discarded. There was no 
signal indicating whether the retrieved papers were actually relevant to the question.

**Fix:** 
- Switched ChromaDB collection from default L2 distance to cosine similarity space 
  (`hnsw:space: cosine`)
- Captured `results["distances"]` from ChromaDB and converted to cosine similarity 
  scores (0-1) per chunk using `similarity = round(1 - dist, 3)`
- Stored scores in `ResearchState` as `faithfulness_scores`
- Computed average faithfulness score per search pass
- Streamed scores to the frontend in real time via the agent activity feed
- Surfaced the final average faithfulness score in the sidebar stats

**Result:** Every search pass now emits a faithfulness signal. Scores above 0.3 
indicate acceptable retrieval quality. Scores below 0.3 trigger a warning in the 
logs. The final score is visible in the sidebar alongside papers found, agents, 
and searches.

**Why cosine similarity:** Cosine similarity measures the angle between two embedding 
vectors regardless of magnitude, making it the correct metric for semantic similarity 
between variable-length texts. L2 distance (the ChromaDB default) is sensitive to 
vector magnitude and produces scores that are harder to interpret.

---

### 4. Gaps and Contradictions Streamed to UI

**Problem:** The Critic was identifying detailed gaps and contradictions in the 
evidence but only logging the counts ("Found 3 contradictions, 7 gaps") to the 
frontend. The actual content was stored in state but never surfaced in the UI.

**Fix:** Updated the Critic's `agent_logs` to include the full text of the top 2 
contradictions and top 3 gaps on every evaluation pass, streamed in real time to 
the Agent Activity feed.

**Result:** During the demo, viewers can watch the Critic identify specific academic 
gaps and contradictions as they stream in — making the system's reasoning process 
visible and interpretable rather than a black box.

---

### 5. Faithfulness Score in Frontend Sidebar

**Problem:** The faithfulness scores were computed and logged but never surfaced as 
a persistent UI element after the pipeline completed.

**Fix:** 
- Added `faithfulness_scores` to the WebSocket `complete` message payload in 
  `main.py`
- Added `faithfulnessAvg` state to `App.jsx`
- Computed the average across all chunk scores on pipeline completion
- Added a "Faithfulness" stat to the sidebar alongside Papers, Agents, Searches, 
  and Report

**Result:** The final faithfulness score is now persistently visible in the sidebar 
after every completed research run.

---

### 6. Paper Limit Increased from 35 to 50

**Problem:** The loop condition `len(raw_papers) < 35` was too aggressive. Pass 1 
typically retrieves 20-25 papers. Pass 2 adds another 15, bringing the total to 
35-40 — which immediately blocked further loop passes even when significant gaps 
remained. The gap-filling logic was being short-circuited by the paper count 
condition before it could do meaningful work.

**Fix:** Increased the paper limit from 35 to 50 in `should_search_again()`.

**Result:** The iterative retrieval loop now runs more effectively when gaps exist, 
allowing the gap-filling queries to find additional relevant papers across all 3 
permitted passes.

---

### 7. Critic Prompt Restored and Improved

**Problem:** During v4 development, multiple prompt iterations caused the Critic to 
inconsistently populate the `gaps` field — sometimes returning 0 gaps despite a 
summary clearly describing missing evidence. The summary field also returned empty 
intermittently.

**Fix:** 
- Restored the original v1 explicit field-by-field instruction style in the system 
  prompt, adapted for tool_use
- Added explicit instruction: *"If your summary identifies limitations, those 
  limitations must appear in the gaps field"*
- Added auto-generated summary fallback when the model returns an empty summary field

**Result:** The Critic now consistently populates gaps when evidence is incomplete, 
and the summary field always contains meaningful content.

---

### Summary of Files Changed in V4

| File | Change |
|------|--------|
| `backend/agents/planner.py` | Replaced JSON parsing with tool_use · removed unused sub_questions field |
| `backend/agents/critic.py` | Replaced JSON parsing with tool_use · gap queries via dedicated Haiku call · restored explicit prompt · gaps/contradictions streamed to UI |
| `backend/agents/fact_checker.py` | Replaced JSON parsing with tool_use · max_tokens increased to 6000 |
| `backend/agents/search.py` | ChromaDB cosine similarity space · faithfulness scores captured and stored |
| `backend/agents/state.py` | Added faithfulness_scores field |
| `backend/agents/pipeline.py` | Added faithfulness_scores to initial state · paper limit 35 → 50 |
| `backend/main.py` | faithfulness_scores added to WebSocket complete message |
| `frontend/src/App.jsx` | Faithfulness score state · sidebar stat · WebSocket handler |
| `backend/test_*.py` | All test files updated with missing state fields |

## V5 Improvements

A second round of targeted improvements focusing on prompt engineering 
quality, LLM sampling parameters, and standardized evaluation.

---

### 1. Explicit Temperature Settings Per Agent

**Problem:** All agents were using the Anthropic API default temperature 
of 1.0. Evaluation agents (Planner, Critic, Fact Checker) benefit from 
lower temperatures for consistency and precision. The Writer benefits 
from slightly higher temperature for synthesis quality.

**Fix:** Set explicit temperature per agent based on task type:

| Agent | Temperature | Rationale |
|-------|-------------|-----------|
| Planner | 0.3 | Consistent, focused query generation |
| Critic | 0.2 | Rigorous, deterministic evidence evaluation |
| Fact Checker | 0.1 | Precise, conservative claim verification |
| Writer | 0.7 | Creative synthesis with controlled variation |

**Result:** More consistent structured outputs from evaluation agents 
and better synthesis quality from the Writer.

---

### 2. Chunk Ordering — Lost in the Middle Fix

**Problem:** Retrieved chunks were passed to the Writer in ChromaDB's 
default cosine similarity order (best first, worst last). Research shows 
LLMs attend more strongly to content at the beginning and end of long 
prompts — the "lost in the middle" phenomenon — meaning the second-best 
chunk was being underweighted.

**Fix:** Implemented optimal chunk ordering before passing to the Writer:
- Best chunk (highest faithfulness score) → position 1
- Second best chunk → last position  
- Remaining chunks → middle positions

```python
chunks_sorted = sorted(chunks, key=lambda x: x.get('faithfulness_score', 0), reverse=True)
if len(chunks_sorted) > 1:
    best        = chunks_sorted[0]
    remaining   = chunks_sorted[1:]
    second_best = remaining.pop(0)
    chunks_sorted = [best] + remaining + [second_best]
```

**Result:** The Writer now sees the two highest-quality chunks in the 
most attended positions, improving synthesis quality for the most 
relevant evidence.

---

### 3. Stronger Chunk Separators with Faithfulness Scores

**Problem:** Chunks were joined with weak `---` separators, risking 
the model treating all chunks as one continuous block. Faithfulness 
scores were computed but never visible to the Writer.

**Fix:** Replaced weak separators with explicit labeled separators 
including per-chunk metadata and faithfulness scores:

=== EXCERPT 1: THaMES (2024-09-17) ===
Authors: Mengfei Liang | Source: arxiv | Faithfulness score: 0.616

=== EXCERPT 2: How do language models learn facts? (2025-03-27) ===
Authors: Nicolas Zucchet | Source: arxiv | Faithfulness score: 0.537

**Result:** The Writer can now distinguish chunk boundaries cleanly 
and reference individual faithfulness scores when calibrating claim 
confidence. This is visible in Methodology Notes sections where the 
Writer now explicitly cites per-chunk faithfulness scores and flags 
lower-scoring sources for appropriate caution.

---

### 4. Grounding Rules in Writer Prompt

**Problem:** The Writer had no explicit instruction to stay within 
retrieved evidence. It would blend parametric memory with retrieved 
content, causing the Fact Checker to make 4-9 corrections per run 
and return Medium confidence ratings.

**Fix:** Added explicit grounding instructions to the Writer system prompt:

GROUNDING RULES:
- Base every specific claim on the provided excerpts
- If a specific detail is not explicitly stated in the excerpts,
do not include it
- You may contextualize and connect findings, but all specific
factual claims must trace back to a provided excerpt

WHEN EVIDENCE IS INSUFFICIENT:
- Do not fabricate claims to fill gaps
- Explicitly note when important aspects are not covered
- Add qualifiers: "according to the available abstracts" or
"based on the provided evidence"

**Result:** Fact Checker corrections dropped from 4-9 per run to 0 
corrections. Overall confidence improved. The Writer now produces 
appropriately hedged claims that the Fact Checker can fully verify.

---

### 5. Paper Limit Increased from 35 to 50

**Problem:** The loop condition `len(raw_papers) < 35` was too 
aggressive. Pass 1 retrieves ~20-25 papers, Pass 2 adds ~15 more — 
hitting 35-40 papers and blocking further passes even when significant 
gaps remained. The gap-filling logic was being short-circuited.

**Fix:** Increased paper limit from 35 to 50 in `should_search_again()`.

**Result:** The iterative retrieval loop now runs more effectively across all 3 permitted passes, allowing gap-filling queries to find additional relevant papers. Total papers collected increased from ~35 to ~50 per run.

---

### 6. RAGAs Evaluation Script

**Addition:** Added `evals/eval_ragas.py` — an evaluation script 
using RAGAs, the industry standard framework for evaluating RAG 
pipelines, configured to use Claude Haiku as the judge LLM.

Metrics evaluated:
- **Faithfulness** — are report claims grounded in retrieved chunks?
- **Answer Relevancy** — does the report address the research question?
- **Context Precision** — are retrieved chunks relevant to the question?
- **Context Recall** — did retrieval find everything needed?

```bash
python evals/eval_ragas.py \
  --question "How does RLHF work?" \
  --save evals/results/ragas_v5.json
```

Note: RAGAs v1.0 dropped support for non-OpenAI LLMs in their 
InstructorLLM interface. The script uses the latest compatible 
approach — if RAGAs updates their Anthropic support, this will 
work without modification.

---

### V5 Evaluation Results

Running the custom 5-metric evaluation suite after v5 improvements:

| Metric | Score | Details |
|--------|-------|---------|
| Report Completeness | A (100%) | All 4 required sections present |
| Citation Presence | A (100%) | 60 citation patterns found |
| Question Coverage | A (100%) | 8/8 keywords covered |
| Evidence Grounding | A (100%) | Grounded in 8/8 sources |
| Critic Quality | A (100%) | 9 gaps, 8 quality scores, 2 contradictions |
| **Overall** | **A (100%)** | **5/5 metrics passed** |

Fact Checker: **0 corrections** across multiple runs (down from 4-9 in v1)

---

### Summary of Files Changed in V5

| File | Change |
|------|--------|
| `backend/agents/planner.py` | temperature=0.3 |
| `backend/agents/critic.py` | temperature=0.2 on both Haiku calls |
| `backend/agents/fact_checker.py` | temperature=0.1 |
| `backend/agents/writer.py` | temperature=0.7 · chunk ordering · stronger separators · grounding rules · not-found behavior |
| `backend/agents/pipeline.py` | paper limit 35 → 50 |
| `evals/eval_ragas.py` | New RAGAs evaluation script |
| `evals/results/custom_eval_v5.json` | V5 evaluation results |

*Built by Mehul Goel · NYU · 2026*
