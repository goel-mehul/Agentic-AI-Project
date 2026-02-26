# Multi-Agent Research Scientist

An autonomous AI research pipeline that takes a research question and produces a verified, cited research report — without any human involvement between input and output.

**Live demo:** [agentic-ai-project-mehul-goel.vercel.app](https://agentic-ai-project-mehul-goel.vercel.app)

Built to demonstrate genuine understanding of agentic AI system design: multi-agent orchestration, RAG pipelines, real-time streaming, and full-stack deployment.

---

## What It Does

Submit a research question. Five specialized AI agents handle everything else.

```
"What are the most effective techniques for reducing hallucinations in LLMs?"
```

And it will:

1. **Plan** a targeted search strategy with 4-6 academic search queries
2. **Search** arXiv and Semantic Scholar for relevant papers (no API keys needed)
3. **Evaluate** the evidence — scoring quality, finding contradictions, identifying gaps
4. **Write** a structured research report with inline citations
5. **Fact-check** every claim against the source evidence
6. **Deliver** a verified, confidence-rated report with a PDF download option

All five steps happen automatically, in sequence, in about 2-3 minutes.

---

## Architecture

The system is a **LangGraph StateGraph** — five specialized AI agents connected by directed edges, each reading from and writing to a shared `ResearchState` object.

```
User Question
      │
      ▼
┌─────────────┐
│   Planner   │  Turns your question into 4-6 optimized academic search queries
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Search    │  Fetches papers from arXiv + Semantic Scholar
│             │  Stores in ChromaDB → retrieves top-8 by semantic similarity
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Critic    │  Scores evidence quality, flags contradictions, notes gaps
└──────┬──────┘
       │
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

**Why 5 separate agents instead of 1?**
Each agent has a single, well-defined job. This makes each one independently testable, replaceable, and improvable. The Critic can be made more rigorous without touching the Writer. The Search agent can be swapped for a different data source without affecting anything else.

**Why LangGraph?**
LangGraph manages state passing between agents, supports streaming (so the frontend gets real-time updates), and makes it trivial to add conditional loops later (e.g., looping back to Search if the Critic finds too many gaps).

**Why ChromaDB?**
Instead of feeding all 20+ retrieved papers to the Writer (expensive, noisy), ChromaDB converts each abstract into a vector embedding and retrieves only the most semantically relevant chunks. This is the RAG (Retrieval-Augmented Generation) pattern used in production AI systems.

**Why two different Claude models?**
The Planner, Search, Critic, and Fact Checker use Claude Haiku (fast, cheap) because their tasks are structured and mechanical. The Writer uses Claude Sonnet (higher quality) because synthesis quality directly affects the output the user sees. This is a deliberate cost/quality tradeoff.

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
| Testing | pytest |
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
│   │   ├── critic.py         # Agent 3 — evidence quality evaluation
│   │   ├── writer.py         # Agent 4 — report synthesis (Sonnet)
│   │   ├── fact_checker.py   # Agent 5 — claim verification + confidence rating
│   │   └── pipeline.py       # LangGraph StateGraph definition
│   └── main.py               # FastAPI + WebSocket streaming server
├── frontend/
│   └── src/
│       ├── App.jsx           # React app — landing, workspace, live agent feed
│       ├── agents.js         # Agent metadata (single source of truth)
│       └── components.css    # All styles
├── evals/                    # pytest suites for every layer
│   ├── test_planner.py
│   ├── test_search.py
│   ├── test_critic.py
│   ├── test_writer.py
│   ├── test_fact_checker.py
│   ├── test_pipeline.py
│   └── test_api.py
└── docs/
    ├── setup.md
    └── project-info/
        ├── PROJECT_DEEP_DIVE.md   # Full technical documentation
        └── INTERVIEW_PREP.md      # Project explained for interviews
```

---

## Running Locally

**Prerequisites:** Python 3.11+, Node.js 18+, Anthropic API key

```bash
# Clone and set up Python environment
git clone https://github.com/YOUR_USERNAME/research-agent.git
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

### Example

```bash
# Start a research session
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "How does retrieval augmented generation reduce hallucinations?"}'

# Response:
# {"session_id": "abc-123", "status": "running", "message": "..."}

# Connect via WebSocket to receive live updates as each agent completes
```

---

## Tests

```bash
python -m pytest evals/ -v
```

Each agent has unit tests with mock data (fast, no API calls). Search agent tests hit real arXiv. API tests use FastAPI's TestClient. ~40 tests total.

---

## Evaluation Framework

The project includes a quantitative evaluation suite measuring:

| Metric | What It Checks | Threshold |
|--------|----------------|-----------|
| Report Completeness | All required sections present | ≥ 75% |
| Citation Presence | Claims backed by references | ≥ 50% |
| Question Coverage | Report addresses the question | ≥ 60% |
| Evidence Grounding | Content grounded in retrieved papers | ≥ 40% |
| Critic Quality | Depth of gap and contradiction analysis | ≥ 50% |

---

## Cost

Using Claude Haiku for 4 agents and Sonnet for the Writer:

| Usage | Estimated Cost |
|-------|----------------|
| Per research run | ~$0.03 – $0.10 |
| 100 research runs | ~$3 – $10 |
| Full development | ~$5 – $15 total |

---

## Planned Improvements

- **Iterative refinement loop** — conditional edge after Critic that loops back to Search when critical gaps are identified
- **Full PDF parsing** — retrieve full paper text, not just abstracts
- **Parallel search** — run arXiv queries concurrently with `asyncio.gather`
- **Persistent storage** — replace in-memory session store with Redis
- **Export formats** — DOCX export in addition to PDF

---

## What I Learned Building This

This project was built incrementally over 10 steps, each committed separately, to demonstrate genuine development process rather than a single code dump.

Key concepts practiced:
- **Agentic AI design** — decomposing complex tasks into specialized agents with single responsibilities
- **LangGraph orchestration** — StateGraph, node functions, reducers, streaming
- **RAG pipelines** — vector embeddings, semantic retrieval, context management
- **Async Python** — FastAPI, WebSockets, running blocking LangGraph in thread pools
- **Multi-agent verification** — having agents check each other's work
- **Evaluation-driven development** — writing quantitative metrics before calling something "working"

---

*Built by Mehul Goel · NYU · 2026*
