# Research Scientist — Complete Project Documentation

Everything about how this project works, why decisions were made, what was learned, and what problems came up along the way.

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [How It Works — Full Technical Walkthrough](#how-it-works)
3. [Every File Explained](#every-file-explained)
4. [Why Each Decision Was Made](#why-each-decision-was-made)
5. [What I Learned at Each Step](#what-i-learned-at-each-step)
6. [Problems I Faced and How I Fixed Them](#problems-i-faced)
7. [Key Concepts Explained Simply](#key-concepts-explained-simply)

---

## What This Project Is

A 5-agent AI pipeline that autonomously conducts academic research. You give it a question. It searches real academic databases, evaluates the quality of the evidence, writes a structured report, fact-checks that report, and delivers a verified, cited research summary — without any human involvement between input and output.

**The core insight:** Complex tasks are better handled by multiple specialized agents than one general agent. Each agent has one job, does it well, and passes its output to the next. This is how production AI systems at companies like Anthropic, OpenAI, and Google are actually built.

**Why I built it:** To demonstrate genuine understanding of agentic AI system design — not just "I called an API," but "I understand how to decompose tasks, orchestrate agents, manage shared state, build evaluation pipelines, and deploy full-stack AI applications."

---

## How It Works — Full Technical Walkthrough

### The Pipeline

When a user submits a research question, here is exactly what happens:

**Step 1: The Planner Agent**

The Planner receives the raw research question and calls Claude Haiku with a carefully crafted system prompt that instructs it to act as a "research planning expert." It returns structured JSON containing:
- 4-6 specific search queries optimized for academic databases
- A search strategy description explaining the approach

Why does this exist separately? Because "What reduces LLM hallucinations?" is a bad search query. "Retrieval augmented generation factual grounding LLM" is a good one. The Planner's entire job is to turn human questions into machine-optimized queries.

**Step 2: The Search Agent**

Takes the Planner's queries and does three things:

1. Queries arXiv (free, no API key, 2M+ papers) using the `arxiv` Python library
2. Queries Semantic Scholar (free, no API key, 200M+ papers) via their REST API
3. Deduplicates results by title to avoid counting the same paper twice
4. Stores all paper abstracts in ChromaDB as vector embeddings
5. Queries ChromaDB with the original research question to find the most semantically relevant chunks

The ChromaDB step is the RAG (Retrieval-Augmented Generation) pattern. Instead of sending all 20+ papers to the Writer (expensive and noisy), ChromaDB mathematically finds which 8 are most relevant to the question. It does this by converting both the papers and the question into vectors — lists of numbers that capture meaning — and finding the closest matches.

**Step 3: The Critic Agent**

Reads all 8 retrieved paper sections and sends them to Claude Haiku with a "rigorous peer reviewer" system prompt. Returns structured JSON with:
- Quality scores (0.0-1.0) for each paper with one-line rationale
- Contradictions between papers (e.g., "Paper A claims X; Paper B claims the opposite")
- Gaps in the literature (important aspects of the question not covered)

This is where the system becomes more sophisticated than a naive "search and summarize" tool. The Writer will use these scores and flags to write a nuanced, honest report rather than blindly summarizing everything equally.

**Step 4: The Writer Agent**

This is the only agent that uses Claude Sonnet (more expensive, higher quality) because the output quality here directly determines what the user sees. It receives:
- The original research question
- All 8 paper sections with full metadata
- The Critic's quality scores
- The list of contradictions
- The list of literature gaps

And produces a structured markdown report with:
- Executive Summary
- Key Findings (organized thematically, with inline citations)
- Contradictions & Debates
- Gaps in the Literature
- Methodology Notes
- References

**Step 5: The Fact Checker Agent**

Reads the Writer's draft alongside the original source evidence and looks for claims that go beyond what the evidence actually supports. Returns:
- A corrected version of the report (overstatements softened or removed)
- A list of every correction made
- An overall confidence rating: High, Medium, or Low
- A rationale for the confidence rating

This appended fact-check footer at the bottom of every report is what makes the system trustworthy rather than confidently wrong.

### The State Object

All five agents share a single `ResearchState` TypedDict. Every agent reads from it and writes to it. The state flows through the LangGraph pipeline:

```python
class ResearchState(TypedDict):
    research_question: str        # Input — never changes
    session_id: str               # Unique ID for this run
    research_plan: list[str]      # Planner writes this
    search_strategy: str          # Planner writes this
    raw_papers: list[dict]        # Search writes this
    retrieved_chunks: list[dict]  # Search writes this (ChromaDB results)
    evidence_quality: dict        # Critic writes this
    contradictions: list[str]     # Critic writes this
    gaps: list[str]               # Critic writes this
    draft_report: str             # Writer writes this
    report_sections: dict         # Writer writes this
    final_report: str             # Fact Checker writes this
    fact_check_notes: list[str]   # Fact Checker writes this
    agent_logs: Annotated[list[str], operator.add]  # All agents append
    current_agent: str            # Updated by each agent
    status: str                   # "running" → "complete"
    error: str                    # Set if something fails
```

The `Annotated[list, operator.add]` on `agent_logs` is a LangGraph reducer — it tells LangGraph to append to the list rather than replace it. Every other field just gets overwritten by the agent that owns it.

### The API Layer

FastAPI exposes two types of endpoints:

**REST:** `POST /research` starts the pipeline and immediately returns a `session_id`. It doesn't wait for the pipeline to finish — it kicks it off as a background task and returns right away. This is important because the pipeline takes 2-3 minutes and you can't hold an HTTP connection open that long.

**WebSocket:** `WS /ws/{session_id}` is what the frontend connects to for real-time updates. The pipeline runs in a thread (because LangGraph is synchronous/blocking), and uses an `asyncio.Queue` to pass results back to the async WebSocket handler as each agent completes. This is the correct pattern for running blocking code inside an async FastAPI application.

### The Frontend

React single-page application with three states:
- **Idle:** Landing page with question input
- **Running:** Workspace with live agent activity feed (WebSocket updates)
- **Complete:** Final report rendered in markdown with PDF download

The WebSocket receives three types of messages:
- `{"type": "log", "agent": "planner", "message": "..."}` — individual log lines
- `{"type": "agent_complete", "agent": "search"}` — agent finished
- `{"type": "complete", "report": "...", "papers_found": 26}` — pipeline done

---

## Every File Explained

### `backend/agents/state.py`
The heart of the architecture. Defines `ResearchState` — the shared data structure that all agents read from and write to. TypedDict gives us type safety. The `Annotated[list, operator.add]` reducer for `agent_logs` is the only LangGraph-specific pattern here.

### `backend/agents/planner.py`
Calls Claude Haiku with a "research planning expert" system prompt. Parses JSON response. Writes `research_plan` and `search_strategy` to state. Strips markdown code fences from Claude's response (a common LLM output quirk).

### `backend/agents/search.py`
Three distinct sections: `_search_arxiv()`, `_search_semantic_scholar()`, and `_store_and_retrieve()`. The ChromaDB client is initialized at module level (once) and a new collection is created per session. Deduplication happens by lowercasing and comparing titles.

### `backend/agents/critic.py`
Formats paper content with 800-char truncation to control token usage. The system prompt is written to be a skeptic — its job is to find problems, not validate. Handles the edge case of no papers gracefully.

### `backend/agents/writer.py`
The only agent using Sonnet. Formats the Critic's outputs (quality scores, contradictions, gaps) into the prompt alongside the evidence. After generating the report, parses it into sections by splitting on `## ` headers. This section parsing lets the Fact Checker and frontend access individual sections.

### `backend/agents/fact_checker.py`
Only touches `final_report`, `fact_check_notes`, and `status`. Appends a formatted fact-check footer to the report. Sets `status = "complete"` — this is the only place in the entire codebase where status becomes "complete," signaling the pipeline is done.

### `backend/agents/pipeline.py`
The conductor. Creates a LangGraph `StateGraph`, adds 5 nodes, defines 4 edges (plus entry point and END), and compiles it once at import time. Exposes `research_graph` (the compiled runnable) and `create_initial_state()` (factory function for clean state).

### `backend/main.py`
FastAPI app with CORS middleware, in-memory session store (dict), active WebSocket connections (dict), and a thread pool executor. The `_run_pipeline` async function uses `asyncio.Queue` to bridge between the synchronous LangGraph thread and the async WebSocket handler.

### `frontend/src/App.jsx`
All React state lives here (lifted to the top-level App component). `Landing` and `Workspace` are pure display components receiving props. WebSocket stored in `useRef` (not useState) to avoid re-renders on connect/disconnect. Cleanup in `useEffect` return function.

### `frontend/src/agents.js`
Single source of truth for agent metadata (id, icon, label, description). Both the landing page preview and the workspace sidebar import from here. Change it once, updates everywhere.

### `evals/`
Full pytest suite for every agent and the API. Agents tested with mock data (no real API calls) for speed and reliability. Search agent tests do make real arXiv calls because the core value is live retrieval. API tests use FastAPI's `TestClient`.

---

## Why Each Decision Was Made

**Why LangGraph instead of just calling agents in a loop?**
LangGraph gives you streaming (`.stream()` yields after each agent), proper state management with reducers, and makes conditional loops trivial to add later. A manual loop would work but you'd have to rebuild all of that yourself.

**Why Claude Haiku for 4 agents and Sonnet for 1?**
Haiku is ~20x cheaper than Sonnet and fast enough for structured tasks (parse JSON, fetch papers, score evidence). Sonnet's higher quality only makes a visible difference in synthesis tasks where nuance matters. This is a real production cost optimization.

**Why ChromaDB instead of just sending all papers to the Writer?**
Context windows cost money and have limits. 20 papers × 500 tokens each = 10,000 tokens just for evidence. ChromaDB lets you find the 8 most relevant ones, saving ~60% of tokens with better results (less noise).

**Why separate Critic and Fact Checker?**
The Critic evaluates evidence quality before writing. The Fact Checker verifies the written report after writing. They have different jobs at different pipeline stages. Combining them would mean the Writer receives less structured critique and the verification step would be harder to isolate and test.

**Why TypedDict instead of a Pydantic model for state?**
LangGraph expects TypedDict for its StateGraph. Pydantic would add validation overhead on every state update. TypedDict gives us type hints without runtime cost.

**Why FastAPI over Flask?**
FastAPI has native async support (critical for WebSockets), auto-generated API docs at `/docs`, and built-in Pydantic validation for request/response models. Flask's WebSocket support requires extensions and is harder to make async-native.

**Why WebSocket instead of Server-Sent Events or polling?**
Polling (repeated GET requests) is wasteful and adds latency. Server-Sent Events are simpler than WebSocket but only one-directional. WebSocket is bidirectional and the standard for real-time UI updates in production apps.

**Why in-memory session store instead of a database?**
For a portfolio project this is fine. The tradeoff is sessions are lost on server restart. Production would use Redis for persistence across restarts and horizontal scaling. The code is structured so swapping the `sessions` dict for Redis calls would be a small change.

**Why Vercel for frontend and Railway for backend?**
Vercel is purpose-built for frontend frameworks — it detects Vite automatically, handles CDN distribution, and deploys in under a minute. Railway is the best free option for Python web services — it handles Nixpacks buildpacks, environment variables, and auto-deploys from GitHub.

---

## What I Learned at Each Step

**Step 1-2 (Setup):**
Why `.gitignore` matters — API keys committed to GitHub get scraped by bots within seconds. Virtual environments isolate project dependencies so "works on my machine" problems disappear. TypedDict is the right tool for defining shared state in LangGraph.

**Step 3 (Planner):**
System prompts are agent personalities — swapping the prompt changes the agent's entire job. Asking LLMs to return JSON is the standard pattern for structured data extraction. Every agent function follows the same signature: `(state) -> state`.

**Step 4 (Search):**
Many valuable academic data sources are completely free and open (arXiv, Semantic Scholar). Vector embeddings convert text into numbers where similar meaning = similar numbers. The RAG pattern (retrieve then generate) is the backbone of almost every production AI system. Deduplication matters — two sources often index the same paper.

**Step 5 (Critic):**
LLMs are excellent evaluators, not just generators. The "LLM-as-judge" pattern is widely used in production AI for quality control. Mock data in tests keeps test suites fast and independent from external services. Each agent should only write to its own state fields.

**Step 6 (Writer):**
Model selection is a real engineering decision with cost/quality tradeoffs. Rich, structured prompts produce better structured outputs. Parsing generated content into structured sections enables downstream processing.

**Step 7 (Fact Checker):**
The "agent checking agent" pattern is fundamental to reliable systems. Single-agent outputs are fragile — verification is how you build trust. Confidence ratings are honest engineering — telling users when to be skeptical is more valuable than projecting false certainty.

**Step 8 (LangGraph):**
`invoke()` blocks until done; `stream()` yields after each node — for real-time UIs you always want `stream()`. Compiling a graph once at import time and reusing it is a performance pattern. The graph structure IS the architecture — it's immediately understandable visually.

**Step 9 (FastAPI):**
CORS is why browsers block cross-origin requests — middleware tells them it's allowed. `run_in_executor` is the correct way to run blocking (synchronous) code inside async FastAPI without freezing the event loop. Session IDs are how you connect stateless HTTP with stateful background processes. `asyncio.Queue` bridges synchronous threads and async handlers cleanly.

**Step 10 (React):**
WebSocket stored in `useRef` not `useState` because you don't want React to re-render when the socket connects. `useEffect` cleanup functions prevent memory leaks when components unmount. Component state should be lifted to the lowest common ancestor that needs it. `@media print` CSS makes browser print-to-PDF produce clean output.

**Deployment:**
Railway needs `requirements.txt` in the root of the deployed directory. Railway's "Root Directory" setting is how you deploy a subdirectory of a monorepo. Vercel auto-detects Vite and handles everything. `wss://` (secure WebSocket) is required for HTTPS sites — `ws://` will be blocked.

---

## Problems I Faced

**`ModuleNotFoundError: No module named 'anthropic'` when running pytest**
Root cause: Mac had both Anaconda Python and venv. `pytest` command used Anaconda's Python which didn't have the packages installed. Fix: always use `python -m pytest` which forces the currently active Python interpreter.

**`cannot import name 'search_agent' from 'agents.search'`**
Root cause: Function name in the file didn't match what was being imported. Fix: check that `def search_agent(...)` exists in the file and the file is saved. VS Code sometimes shows unsaved changes with a dot on the tab.

**`TypeError: 'NoneType' object is not subscriptable` on evidence_quality**
Root cause: Critic agent returned `None` for `evidence_quality` when Claude responded in an unexpected format (only 2 papers found due to arXiv rate limiting). Fix: defensive coding — `evidence = state["evidence_quality"] or {}` before calling `.items()`.

**arXiv HTTP 429 errors**
Root cause: Too many test runs in quick succession hit arXiv's rate limits. Fix: wait 2-3 minutes between runs, or use mock data in tests instead of real API calls.

**UI not showing landing page on load**
Root cause: Vite was caching an old state where `status` wasn't `"idle"`. Fix: hard refresh (`Cmd + Shift + R`) or open in incognito window.

**Railway deployment: "Error creating build plan with Railpack"**
Root cause: Railway was trying to build from the repo root, not the `backend/` subdirectory. Fix: set Root Directory to `backend` in Railway service Settings → Networking.

**Vercel hitting `//research` double slash URL**
Root cause: The `API` constant had a trailing slash. Fix: remove trailing slash from the URL string.

**WebSocket not connecting on deployed version**
Root cause: `ws://` (insecure WebSocket) was blocked by the browser on an HTTPS page. Fix: use `wss://` (secure WebSocket) for the deployed URL.

**UI updates only appearing at the end (not real-time)**
Root cause: The `_run_pipeline` function was collecting all LangGraph steps in a list before processing them. Fix: use `asyncio.Queue` — the LangGraph thread puts each step on the queue as it completes, and the async handler picks them up and sends WebSocket updates immediately.

---

## Key Concepts Explained Simply

**Agentic AI**
Instead of one AI doing everything, you break the task into specialized roles. Each agent has one job, a specific prompt that defines its personality, and clear inputs/outputs. More reliable, more testable, easier to improve one piece without breaking others.

**LangGraph**
A library for building AI agent pipelines as graphs. You define nodes (agents) and edges (flow between them). LangGraph handles passing state between nodes, supports streaming, and makes conditional branching (loops, retries) easy to add.

**RAG (Retrieval-Augmented Generation)**
Instead of asking an LLM to answer from memory (which leads to hallucinations), you first retrieve relevant documents from a database, then give those documents to the LLM as context. The LLM generates based on real retrieved evidence rather than making things up.

**Vector Embeddings**
A way to represent text as numbers. Similar texts get similar numbers. This lets you do semantic search — find documents that mean the same thing as your query, even if they use different words. ChromaDB stores these embeddings and does the similarity search.

**TypedDict**
A Python type hint tool that defines the shape of a dictionary. Gives you autocomplete and type checking without runtime overhead. LangGraph uses TypedDict to define the state schema for a pipeline.

**WebSocket**
A persistent two-way connection between browser and server. Unlike HTTP (request → response → close), WebSocket stays open so the server can push updates to the client at any time. Essential for real-time UI updates in long-running tasks.

**asyncio.Queue**
A thread-safe queue for passing data between synchronous threads and async code. The LangGraph pipeline runs in a thread (blocking), and uses the queue to send results to the async FastAPI handler without blocking the event loop.

**CORS (Cross-Origin Resource Sharing)**
A browser security policy that blocks requests from one domain (vercel.app) to another (railway.app) by default. The FastAPI `CORSMiddleware` adds headers that tell the browser the cross-origin request is allowed.
