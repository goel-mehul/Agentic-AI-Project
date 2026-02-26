# Multi-Agent Research Scientist

An autonomous AI research pipeline that takes a research question and produces a verified, cited research report — without any human involvement between input and output.

Built as a portfolio project demonstrating **agentic AI system design**, multi-agent orchestration, RAG pipelines, and real-time streaming interfaces.

---

## What It Does

Ask it a research question like:

> *"What are the most effective techniques for reducing hallucinations in large language models?"*

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
│   Planner   │  Breaks question into focused search queries
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Search    │  Fetches papers → stores in ChromaDB → retrieves top-k chunks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Critic    │  Scores evidence quality, flags contradictions, notes gaps
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Writer    │  Synthesizes structured markdown report with citations
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Fact Checker   │  Verifies claims vs. sources, adds confidence rating
└─────────────────┘
       │
       ▼
  Final Report
```

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
| Vector Store | ChromaDB (local) |
| Backend API | FastAPI + WebSockets |
| Frontend | React + Vite |
| Testing | pytest |

---

## Project Structure

```
research-agent/
├── backend/
│   ├── agents/
│   │   ├── state.py          # Shared ResearchState TypedDict
│   │   ├── planner.py        # Agent 1: Decomposes research question
│   │   ├── search.py         # Agent 2: arXiv + Semantic Scholar + ChromaDB
│   │   ├── critic.py         # Agent 3: Evidence quality evaluation
│   │   ├── writer.py         # Agent 4: Report synthesis (uses Sonnet)
│   │   ├── fact_checker.py   # Agent 5: Claim verification
│   │   └── pipeline.py       # LangGraph StateGraph definition
│   ├── main.py               # FastAPI + WebSocket streaming server
│   ├── test_planner.py       # Manual test scripts
│   ├── test_search.py
│   ├── test_critic.py
│   ├── test_writer.py
│   └── test_fact_checker.py
├── frontend/
│   └── src/
│       ├── App.jsx           # Main React component
│       ├── agents.js         # Agent metadata
│       ├── components.css    # All styles
│       └── main.jsx          # Entry point
├── evals/
│   ├── test_planner.py       # Automated pytest suites
│   ├── test_search.py
│   ├── test_critic.py
│   ├── test_writer.py
│   ├── test_fact_checker.py
│   ├── test_pipeline.py
│   └── test_api.py
└── docs/
    └── setup.md
```

---

## Setup & Running

### Prerequisites
- Python 3.11+
- Node.js 18+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### 1. Clone and set up Python environment

```bash
git clone https://github.com/YOUR_USERNAME/research-agent.git
cd research-agent

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt
```

### 2. Add your API key

```bash
cp backend/.env.example backend/.env
# Open backend/.env and add your Anthropic API key
```

### 3. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### 5. Run the test suite

```bash
# From project root
python -m pytest evals/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/research` | Start a research session, returns `session_id` |
| GET | `/research/{id}` | Poll for results |
| WS | `/ws/{id}` | Stream real-time agent updates |

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
