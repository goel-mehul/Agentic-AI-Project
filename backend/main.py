"""
main.py — FastAPI Backend
==========================

WHAT IT DOES:
    Exposes the multi-agent pipeline as a web service with:
    - POST /research      — Start a research session
    - GET  /research/{id} — Poll for results
    - WS   /ws/{id}       — Stream real-time agent updates

WHY IT EXISTS:
    The React frontend can't import Python directly. FastAPI creates
    an HTTP/WebSocket interface that any frontend (or API client) can use.

HOW IT FITS:
    Sits between the pipeline (backend) and the UI (frontend).
    The pipeline is synchronous Python; FastAPI makes it async-compatible
    and exposes it over the network.

WHAT YOU'RE LEARNING:
    - FastAPI basics (routes, request/response models)
    - WebSockets for real-time communication
    - Running blocking code in async context (asyncio + threads)
    - In-memory session management
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.pipeline import research_graph, create_initial_state

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Agent Research Scientist",
    description="5-agent AI pipeline for academic research synthesis",
    version="1.0.0"
)

# CORS allows the React frontend (localhost:5173) to call this API
# Without this, browsers block cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://agentic-ai-project-mehul-goel.vercel.app", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for running the synchronous LangGraph pipeline
# in a way that doesn't block FastAPI's async event loop
executor = ThreadPoolExecutor(max_workers=4)

# In-memory session store: session_id -> state dict
# In production you'd use Redis; this is fine for a portfolio project
sessions: dict[str, dict] = {}

# Active WebSocket connections: session_id -> WebSocket
active_connections: dict[str, WebSocket] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What techniques reduce hallucinations in LLMs?"
            }
        }


class ResearchResponse(BaseModel):
    session_id: str
    status: str
    message: str


# ── WebSocket helper ──────────────────────────────────────────────────────────

async def send_update(session_id: str, payload: dict):
    """Send a JSON update to the WebSocket client for this session."""
    ws = active_connections.get(session_id)
    if ws:
        try:
            await ws.send_json(payload)
        except Exception:
            # Client disconnected — that's OK, just stop sending
            active_connections.pop(session_id, None)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check — confirms the API is running."""
    return {
        "service": "Multi-Agent Research Scientist",
        "status": "running",
        "agents": ["planner", "search", "critic", "writer", "fact_checker"],
        "docs": "/docs"
    }


@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest):
    """
    Start a new research session.

    Creates initial state, starts the pipeline in the background,
    and immediately returns a session_id. The frontend uses this
    session_id to connect via WebSocket for live updates.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Research question cannot be empty")

    # Build initial state
    state = create_initial_state(request.question)
    session_id = state["session_id"]

    # Store placeholder so WebSocket can find it immediately
    sessions[session_id] = {"status": "running", "state": None}

    # Run pipeline in background — don't await it here
    asyncio.create_task(_run_pipeline(session_id, state))

    return ResearchResponse(
        session_id=session_id,
        status="running",
        message="Pipeline started. Connect via WebSocket for live updates."
    )


@app.get("/research/{session_id}")
async def get_result(session_id: str):
    """
    Get the result of a completed research session.
    Use this to poll for results if not using WebSocket.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if session["status"] == "running":
        return {"status": "running", "message": "Research in progress..."}

    state = session.get("state", {})
    return {
        "status":            state.get("status", "unknown"),
        "session_id":        session_id,
        "research_question": state.get("research_question", ""),
        "final_report":      state.get("final_report", ""),
        "papers_found":      len(state.get("raw_papers", [])),
        "contradictions":    state.get("contradictions", []),
        "gaps":              state.get("gaps", []),
        "fact_check_notes":  state.get("fact_check_notes", [])
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time pipeline updates.

    The frontend connects here immediately after POST /research.
    It receives JSON messages like:
      {"type": "log",            "agent": "planner", "message": "..."}
      {"type": "agent_complete", "agent": "search"}
      {"type": "complete",       "report": "...", "papers_found": 23}
      {"type": "error",          "message": "..."}
    """
    await websocket.accept()
    active_connections[session_id] = websocket

    try:
        ping_counter = 0
        while session_id in sessions and sessions[session_id]["status"] == "running":
            await asyncio.sleep(1)
             ping_counter += 1
            if ping_counter % 30 == 0:  # ping every 30 seconds
            # Send keepalive ping every 30 seconds to prevent timeout
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

        # Send final state if complete
        if session_id in sessions and sessions[session_id]["status"] == "complete":
            state = sessions[session_id]["state"]
            await websocket.send_json({
                "type": "complete",
                "report": state.get("final_report", ""),
                "papers_found": len(state.get("raw_papers", [])),
                "contradictions": state.get("contradictions", []),
                "gaps": state.get("gaps", [])
            })

    except WebSocketDisconnect:
        pass
    finally:
        active_connections.pop(session_id, None)


# ── Background pipeline runner ────────────────────────────────────────────────

async def _run_pipeline(session_id: str, initial_state: dict):
    """
    Runs the LangGraph pipeline and streams updates via WebSocket
    as each agent completes — not just at the end.
    """
    loop = asyncio.get_event_loop()
    final_state = dict(initial_state)

    try:
        # We use a queue to pass results from the thread back to async
        queue = asyncio.Queue()

        def run_graph():
            """Runs in a thread — puts each step into the queue."""
            try:
                for step in research_graph.stream(initial_state):
                    # Put the step on the queue so the async side can send it
                    loop.call_soon_threadsafe(queue.put_nowait, ("step", step))
                # Signal that we're done
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

        # Start the graph in a background thread
        import threading
        thread = threading.Thread(target=run_graph, daemon=True)
        thread.start()

        # Process queue items as they arrive — this is the real-time part
        while True:
            msg_type, payload = await queue.get()

            if msg_type == "error":
                raise Exception(payload)

            if msg_type == "done":
                break

            if msg_type == "step":
                # Each step = one agent just finished
                for agent_name, node_state in payload.items():
                    final_state.update(node_state)

                    # Send each log line immediately
                    for log in node_state.get("agent_logs", []):
                        await send_update(session_id, {
                            "type":    "log",
                            "agent":   agent_name,
                            "message": log
                        })
                        await asyncio.sleep(0.05)

                    # Tell frontend this agent is done
                    await send_update(session_id, {
                        "type":  "agent_complete",
                        "agent": agent_name
                    })

                    # Small pause so the UI can render the update visually
                    await asyncio.sleep(0.3)

        # Store final state
        sessions[session_id] = {"status": "complete", "state": final_state}

        # Send the completed report
        await send_update(session_id, {
            "type":           "complete",
            "report":         final_state.get("final_report", ""),
            "papers_found":   len(final_state.get("raw_papers", [])),
            "contradictions": final_state.get("contradictions", []),
            "gaps":           final_state.get("gaps", [])
        })

    except Exception as e:
        sessions[session_id] = {
            "status": "error",
            "state": {**initial_state, "error": str(e), "status": "error"}
        }
        await send_update(session_id, {
            "type":    "error",
            "message": f"Pipeline error: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)