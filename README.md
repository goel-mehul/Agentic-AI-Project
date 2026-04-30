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

*Built by Mehul Goel · NYU · 2026*
