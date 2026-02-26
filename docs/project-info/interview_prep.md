# Interview Prep — Research Scientist Project

Read this before any interview where you mention this project. Know it well enough to speak naturally, not recite it.

---

## The 30-Second Pitch

> "I built a 5-agent AI research pipeline. You give it a question, and it autonomously searches academic databases, evaluates the evidence quality, writes a structured report, and fact-checks its own claims — without any human in the loop. I built it to demonstrate real understanding of agentic AI system design: how to decompose complex tasks, orchestrate agents with LangGraph, implement RAG pipelines, and deploy full-stack AI applications."

Use this when someone asks "tell me about a project you've worked on."

---

## The 2-Minute Deep Dive

> "The architecture is a LangGraph StateGraph — five specialized agents connected by directed edges, all sharing a single state object.
>
> The Planner takes your question and turns it into 4-6 optimized search queries. The Search agent hits arXiv and Semantic Scholar — both free, no API keys — fetches papers, stores them in ChromaDB, and uses semantic similarity to retrieve only the most relevant chunks. That's the RAG pattern.
>
> The Critic reads those chunks and does something most systems skip: it scores each paper's quality, flags contradictions between papers, and identifies gaps in the evidence. The Writer receives all of that context — not just the papers, but the critique — and uses Claude Sonnet to synthesize a structured report with real inline citations. Then the Fact Checker reads the draft against the sources and corrects any overstatements.
>
> On the backend it's FastAPI with WebSocket streaming — the pipeline runs in a thread pool since LangGraph is synchronous, and I use asyncio.Queue to bridge back to the async WebSocket handler for real-time updates. Frontend is React on Vercel, backend on Railway."

Use this when an interviewer says "walk me through how it works."

---

## Questions You Will Definitely Get Asked

**"Why five agents instead of one?"**
> "Each agent has a single, well-defined responsibility. The Critic can be made more rigorous without touching the Writer. The Search agent can be swapped for a different data source without affecting anything downstream. It's also testable — I have a full pytest suite where each agent is tested in isolation with mock data. One god-agent doing everything is harder to debug, harder to improve, and harder to trust."

**"Why LangGraph specifically?"**
> "A few reasons. It gives you streaming out of the box — `.stream()` yields after each node completes, which is what powers the real-time UI. It handles the state reducer pattern — I use `Annotated[list, operator.add]` for the agent logs so they append rather than overwrite. And it makes conditional loops trivial to add. My next improvement is a loop where if the Critic finds too many gaps, the pipeline routes back to Search for another retrieval pass. That's one line in LangGraph."

**"What's RAG and why did you use it?"**
> "Retrieval-Augmented Generation — instead of asking an LLM to answer from memory, you first retrieve relevant documents and give them as context. I use it because the Search agent might find 25 papers but only 8 are actually relevant. Feeding all 25 to the Writer wastes tokens and adds noise. ChromaDB converts each abstract into a vector embedding — a list of numbers capturing semantic meaning — and finds the closest matches to the research question mathematically. Better results, lower cost."

**"Why two different Claude models?"**
> "Cost optimization. Haiku is about 20x cheaper than Sonnet and fast enough for structured tasks — parsing JSON, scoring evidence, verifying claims. Sonnet's higher reasoning quality only makes a visible difference in synthesis tasks where nuance matters. Using Sonnet for everything would cost roughly $0.50 per run instead of $0.05. At scale that's a 10x cost difference for marginal gain."

**"How does the real-time streaming work?"**
> "LangGraph's `.stream()` is synchronous and blocking, but FastAPI is async. You can't call blocking code directly in async FastAPI without freezing the event loop and blocking all other requests. So the pipeline runs in a thread pool via `run_in_executor`. I use an `asyncio.Queue` to pass results from the synchronous thread back to the async WebSocket handler — the thread puts each completed step on the queue, and the async side picks them up and sends WebSocket messages immediately. That's why you see each agent update in real time."

**"What would you do differently / what are the limitations?"**
> "A few things. The session store is in-memory, so if the Railway server restarts, active sessions are lost — production would use Redis. The iterative refinement loop I mentioned would improve report quality for complex questions. I'm also only using paper abstracts, not full text — full PDF parsing would give the Writer much richer evidence to work from. And the fact-checker is limited by the same abstracts the Writer used — it can only verify claims against what was retrieved, not ground truth."

**"How did you test it?"**
> "Full pytest suite across all layers. Each agent has unit tests with mock data — fast, no API calls, tests the logic in isolation. The Search agent tests do make real arXiv calls because the core value is live retrieval. The API layer has tests using FastAPI's TestClient. The pipeline has structural tests that validate the graph compiles and initial state has all required fields. Around 40 tests total, all passing."

**"What problems did you run into?"**
> "A few interesting ones. arXiv rate-limiting caused the Critic to receive almost no evidence during consecutive test runs — I learned to use mock data in unit tests rather than hitting real APIs. The asyncio/threading boundary was the trickiest engineering problem — you can't just call synchronous blocking code from async FastAPI. The deployment had a WebSocket issue where `ws://` gets blocked by browsers on HTTPS pages — has to be `wss://`. And there was a subtle state bug where the Critic returned None for evidence_quality when Claude responded in an unexpected format, which I fixed with defensive coding."

---

## Numbers to Know

| Metric | Value |
|--------|-------|
| Agents | 5 |
| Tests | ~40 |
| Git commits | 70+ |
| Papers retrieved per run | 20-30 |
| Papers after ChromaDB retrieval | 8 (top-k) |
| Average pipeline runtime | 2-3 minutes |
| Cost per research run | ~$0.03-0.10 |
| Models used | Claude Haiku (4 agents), Claude Sonnet (Writer) |
| Frontend hosting | Vercel |
| Backend hosting | Railway |

---

## Key Terms to Use Naturally

These are words that signal you understand the domain. Use them in answers but only when they fit naturally — forced jargon is worse than no jargon.

- **Agentic AI** / **multi-agent system**
- **LangGraph StateGraph**
- **RAG (Retrieval-Augmented Generation)**
- **Vector embeddings** / **semantic similarity**
- **Agent orchestration**
- **State reducer** (specifically `Annotated[list, operator.add]`)
- **Streaming** (`.stream()` vs `.invoke()`)
- **Thread pool executor** / **asyncio**
- **LLM-as-judge** (the Critic and Fact Checker pattern)
- **Conditional edges** (the planned loop improvement)
- **Cost-quality tradeoff** (Haiku vs Sonnet)

---

## What This Project Demonstrates

Be ready to explicitly connect the project to what the role needs. Map it like this:

**"We need someone who understands AI systems"**
→ I built a 5-agent pipeline with LangGraph, understand StateGraph, reducers, streaming, and have practical experience with RAG and vector stores.

**"We need someone who can ship full-stack"**
→ Python backend (FastAPI, async, WebSockets), React frontend (hooks, real-time state, WebSocket client), deployed on Railway + Vercel.

**"We need someone who writes production-quality code"**
→ 70+ commits showing incremental development, full pytest suite (~40 tests), separation of concerns between agents/pipeline/API/UI, defensive error handling.

**"We need someone who can learn fast"**
→ Built this from scratch, hit real problems (rate limiting, async/sync boundary, WebSocket security), diagnosed and fixed each one.

**"We need someone who thinks about cost and scale"**
→ Deliberate model selection (Haiku vs Sonnet), RAG to reduce token usage, in-memory sessions with a clear path to Redis for scale.

---

## The One Thing to Convey

If you only land one point in the whole conversation, make it this:

**You understand WHY the system is designed the way it is — not just how to code it.**

Anyone can call an API and get a response. The interesting part is knowing why you use 5 agents instead of 1, why RAG is better than stuffing everything in context, why the Critic exists before the Writer and the Fact Checker after, why Haiku for some steps and Sonnet for others, why WebSocket instead of polling.

That's the difference between someone who followed a tutorial and someone who can design systems from scratch.
