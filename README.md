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

*Built by Mehul Goel · NYU · 2026*
