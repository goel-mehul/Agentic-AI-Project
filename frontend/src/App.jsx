import { useState, useEffect, useRef } from "react"
import ReactMarkdown from "react-markdown"
import { AGENTS, AGENT_IDS } from "./agents"

const API = "http://localhost:8000"

// ── API helpers ──────────────────────────────────────────────────────────────

async function startResearch(question) {
  const res = await fetch(`${API}/research`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ question }),
  })
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

// ── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [question, setQuestion]               = useState("")
  const [status, setStatus]                   = useState("idle")
  const [logs, setLogs]                       = useState([])
  const [activeAgent, setActiveAgent]         = useState(null)
  const [completedAgents, setCompletedAgents] = useState(new Set())
  const [report, setReport]                   = useState(null)
  const [papersFound, setPapersFound]         = useState(0)
  const [error, setError]                     = useState("")
  const wsRef    = useRef(null)
  const logEndRef = useRef(null)

  // Auto-scroll logs to bottom
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  // Cleanup WebSocket on unmount
  useEffect(() => () => wsRef.current?.close(), [])

  async function handleSubmit(e) {
    e.preventDefault()
    if (!question.trim() || status === "running") return

    setStatus("running")
    setLogs([])
    setReport(null)
    setActiveAgent(null)
    setCompletedAgents(new Set())
    setError("")
    setPapersFound(0)

    try {
      const { session_id } = await startResearch(question)
      connectWebSocket(session_id)
    } catch (err) {
      setError(err.message)
      setStatus("error")
    }
  }

  function connectWebSocket(sessionId) {
    const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`)
    wsRef.current = ws

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data)

      if (msg.type === "log") {
        setLogs(prev => [...prev, { agent: msg.agent, text: msg.message }])
        setActiveAgent(msg.agent)
      }
      if (msg.type === "agent_complete") {
        setCompletedAgents(prev => new Set([...prev, msg.agent]))
      }
      if (msg.type === "complete") {
        setReport(msg.report)
        setPapersFound(msg.papers_found || 0)
        setStatus("complete")
        setActiveAgent(null)
        setCompletedAgents(new Set(AGENT_IDS))
      }
      if (msg.type === "error") {
        setError(msg.message)
        setStatus("error")
      }
    }

    ws.onerror = () => {
      setError("WebSocket connection failed. Is the backend running on port 8000?")
      setStatus("error")
    }
  }

  function reset() {
    wsRef.current?.close()
    setStatus("idle")
    setQuestion("")
    setLogs([])
    setReport(null)
    setActiveAgent(null)
    setCompletedAgents(new Set())
    setError("")
  }

  function getAgentState(agentId) {
    if (completedAgents.has(agentId)) return "done"
    if (activeAgent === agentId)      return "active"
    return "idle"
  }

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <span className="logo-text">Research Scientist</span>
            <span className="logo-badge">5-Agent Pipeline</span>
          </div>
          {status !== "idle" && (
            <button className="btn-ghost" onClick={reset}>
              ← New Research
            </button>
          )}
        </div>
      </header>

      <main className="main">

        {/* ── Landing ── */}
        {status === "idle" && (
          <Landing
            question={question}
            setQuestion={setQuestion}
            onSubmit={handleSubmit}
          />
        )}

        {/* ── Running / Complete ── */}
        {status !== "idle" && (
          <Workspace
            question={question}
            status={status}
            logs={logs}
            logEndRef={logEndRef}
            activeAgent={activeAgent}
            getAgentState={getAgentState}
            report={report}
            papersFound={papersFound}
            error={error}
          />
        )}
      </main>
    </div>
  )
}

// ── Landing Component ─────────────────────────────────────────────────────────

function Landing({ question, setQuestion, onSubmit }) {
  return (
    <div className="landing">
      <div className="landing-headline">
        <h1>Five AI agents.<br />One research report.</h1>
        <p className="subhead">
          Ask a research question. A pipeline of specialized agents searches
          arXiv, evaluates evidence, and synthesizes a verified report —
          automatically.
        </p>
      </div>

      <form className="query-form" onSubmit={onSubmit}>
        <div className="query-box">
          <textarea
            className="query-input"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="e.g. What are the most effective techniques for reducing hallucinations in large language models?"
            rows={3}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                onSubmit(e)
              }
            }}
          />
          <button
            className="btn-primary"
            type="submit"
            disabled={!question.trim()}
          >
            Start Research →
          </button>
        </div>
      </form>

      <div className="agent-grid">
        {AGENTS.map((a, i) => (
          <div className="agent-card" key={a.id}>
            <span className="agent-num">0{i + 1}</span>
            <span className="agent-icon">{a.icon}</span>
            <span className="agent-name">{a.label}</span>
            <span className="agent-desc">{a.desc}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Workspace Component ───────────────────────────────────────────────────────

function Workspace({
  question, status, logs, logEndRef,
  activeAgent, getAgentState,
  report, papersFound, error
}) {
  return (
    <div className="workspace">

      {/* Left: Pipeline status */}
      <aside className="sidebar">
        <p className="sidebar-label">Pipeline</p>
        <div className="question-chip">{question}</div>

        <div className="pipeline">
          {AGENTS.map((a, i) => {
            const state = getAgentState(a.id)
            return (
              <div className={`pipeline-step step-${state}`} key={a.id}>
                {i > 0 && <div className="step-line" />}
                <div className="step-dot">
                  {state === "done"   && <span className="dot-check">✓</span>}
                  {state === "active" && <span className="dot-pulse" />}
                  {state === "idle"   && <span className="dot-idle" />}
                </div>
                <div className="step-text">
                  <div className="step-name">{a.icon} {a.label}</div>
                  <div className="step-desc">{a.desc}</div>
                </div>
              </div>
            )
          })}
        </div>

        {status === "complete" && (
          <div className="stats">
            <div className="stat">
              <span className="stat-n">{papersFound}</span>
              <span className="stat-l">Papers</span>
            </div>
            <div className="stat">
              <span className="stat-n">5</span>
              <span className="stat-l">Agents</span>
            </div>
          </div>
        )}
      </aside>

      {/* Right: Logs + Report */}
      <div className="content">

        {/* Live logs */}
        {logs.length > 0 && (
          <div className="log-panel">
            <div className="panel-header">
              Agent Activity
              {status === "running" && (
                <span className="live-dot">● LIVE</span>
              )}
            </div>
            <div className="log-feed">
              {logs.map((log, i) => (
                <div className="log-row" key={i}>
                  <span className="log-agent">{log.agent}</span>
                  <span className="log-text">{log.text}</span>
                </div>
              ))}
              {status === "running" && (
                <div className="log-row">
                  <span className="log-agent">{activeAgent}</span>
                  <span className="thinking">
                    <span /><span /><span />
                  </span>
                </div>
              )}
              <div ref={logEndRef} />
            </div>
          </div>
        )}

        {/* Error */}
        {status === "error" && (
          <div className="error-box">
            <div className="error-title">Pipeline Error</div>
            <div className="error-msg">{error}</div>
            <div className="error-hint">
              Make sure the backend is running:{" "}
              <code>uvicorn main:app --reload</code>
            </div>
          </div>
        )}

        {/* Final report */}
        {status === "complete" && report && (
          <div className="report-panel">
            <div className="panel-header">Research Report</div>
            <div className="report-body">
              <ReactMarkdown>{report}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}