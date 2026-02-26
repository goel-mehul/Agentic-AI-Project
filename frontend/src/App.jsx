import { useState, useEffect, useRef } from "react"
import ReactMarkdown from "react-markdown"
import { AGENTS, AGENT_IDS } from "./agents"

const API = "https://agentic-ai-project-production.up.railway.app"

async function startResearch(question) {
  const res = await fetch(`${API}/research`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ question }),
  })
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export default function App() {
  const [question, setQuestion]               = useState("")
  const [status, setStatus]                   = useState("idle")
  const [logs, setLogs]                       = useState([])
  const [agentLogs, setAgentLogs]             = useState({})   // per-agent logs
  const [activeAgent, setActiveAgent]         = useState(null)
  const [selectedAgent, setSelectedAgent]     = useState(null) // clicked agent
  const [completedAgents, setCompletedAgents] = useState(new Set())
  const [report, setReport]                   = useState(null)
  const [papersFound, setPapersFound]         = useState(0)
  const [error, setError]                     = useState("")
  const [activeTab, setActiveTab]             = useState("logs") // "logs" | "report"
  const wsRef     = useRef(null)
  const logEndRef = useRef(null)

  useEffect(() => {
    if (activeTab === "logs") {
      logEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [logs, activeTab])

  useEffect(() => () => wsRef.current?.close(), [])

  // When report arrives, auto-switch to report tab
  useEffect(() => {
    if (report) setActiveTab("report")
  }, [report])

  async function handleSubmit(e) {
    e.preventDefault()
    if (!question.trim() || status === "running") return

    setStatus("running")
    setLogs([])
    setAgentLogs({})
    setReport(null)
    setActiveAgent(null)
    setSelectedAgent(null)
    setCompletedAgents(new Set())
    setError("")
    setPapersFound(0)
    setActiveTab("logs")

    try {
      const { session_id } = await startResearch(question)
      connectWebSocket(session_id)
    } catch (err) {
      setError(err.message)
      setStatus("error")
    }
  }

  function connectWebSocket(sessionId) {
    const ws = new WebSocket(`wss://agentic-ai-project-production.up.railway.app/ws/${sessionId}`)
    wsRef.current = ws

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data)

      if (data.type === 'ping') return;

      if (msg.type === "log") {
        const entry = { agent: msg.agent, text: msg.message, ts: Date.now() }
        setLogs(prev => [...prev, entry])
        // Also store per-agent
        setAgentLogs(prev => ({
          ...prev,
          [msg.agent]: [...(prev[msg.agent] || []), entry]
        }))
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
    setAgentLogs({})
    setReport(null)
    setActiveAgent(null)
    setSelectedAgent(null)
    setCompletedAgents(new Set())
    setError("")
  }

  function getAgentState(agentId) {
    if (completedAgents.has(agentId)) return "done"
    if (activeAgent === agentId)      return "active"
    return "idle"
  }

  // What to show in the right panel when an agent is selected
  const selectedAgentData = selectedAgent
    ? AGENTS.find(a => a.id === selectedAgent)
    : null
  const selectedAgentLogs = selectedAgent
    ? (agentLogs[selectedAgent] || [])
    : []

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-hex">⬡</div>
            <div>
              <div className="logo-title">Research Scientist</div>
              <div className="logo-sub">Multi-Agent AI Pipeline</div>
            </div>
          </div>
          {status !== "idle" && (
            <button className="btn-new" onClick={reset}>
              + New Research
            </button>
          )}
        </div>
      </header>

      <main className="main">
        {status === "idle" ? (
          <Landing question={question} setQuestion={setQuestion} onSubmit={handleSubmit} />
        ) : (
          <Workspace
            question={question}
            status={status}
            logs={logs}
            logEndRef={logEndRef}
            activeAgent={activeAgent}
            selectedAgent={selectedAgent}
            setSelectedAgent={setSelectedAgent}
            selectedAgentData={selectedAgentData}
            selectedAgentLogs={selectedAgentLogs}
            getAgentState={getAgentState}
            report={report}
            papersFound={papersFound}
            error={error}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
          />
        )}
      </main>
    </div>
  )
}

// ── Landing ───────────────────────────────────────────────────────────────────

function Landing({ question, setQuestion, onSubmit }) {
  return (
    <div className="landing">
      <div className="landing-hero">
        <div className="hero-tag">Agentic AI Research Tool</div>
        <h1 className="hero-title">
          Ask a question.<br />
          <span className="hero-accent">Five agents find the answer.</span>
        </h1>
        <p className="hero-sub">
          A pipeline of specialized AI agents searches academic databases,
          evaluates evidence quality, and synthesizes a verified research
          report — all automatically.
        </p>
      </div>

      <form className="query-form" onSubmit={onSubmit}>
        <label className="query-label">Your research question</label>
        <div className="query-box">
          <textarea
            className="query-input"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="e.g. What are the most effective techniques for reducing hallucinations in large language models?"
            rows={4}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                onSubmit(e)
              }
            }}
          />
          <div className="query-footer">
            <span className="query-hint">Press Enter to submit · Shift+Enter for new line</span>
            <button className="btn-submit" type="submit" disabled={!question.trim()}>
              Start Research →
            </button>
          </div>
        </div>
      </form>

      <div className="pipeline-preview">
        <div className="preview-label">How it works</div>
        <div className="preview-steps">
          {AGENTS.map((a, i) => (
            <div className="preview-step" key={a.id}>
              <div className="preview-step-num">{String(i+1).padStart(2,"0")}</div>
              <div className="preview-step-icon">{a.icon}</div>
              <div className="preview-step-info">
                <div className="preview-step-name">{a.label}</div>
                <div className="preview-step-desc">{a.desc}</div>
              </div>
              {i < AGENTS.length - 1 && <div className="preview-arrow">→</div>}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Workspace ─────────────────────────────────────────────────────────────────

function Workspace({
  question, status, logs, logEndRef,
  activeAgent, selectedAgent, setSelectedAgent,
  selectedAgentData, selectedAgentLogs,
  getAgentState, report, papersFound,
  error, activeTab, setActiveTab
}) {
  return (
    <div className="workspace">

      {/* ── Left sidebar: Agent pipeline ── */}
      <aside className="sidebar">
        <div className="sidebar-section">
          <div className="sidebar-section-label">Research Question</div>
          <div className="question-display">{question}</div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-label">
            Agent Pipeline
            {status === "running" && <span className="running-badge">Running</span>}
            {status === "complete" && <span className="done-badge">Complete</span>}
          </div>

          <div className="agent-list">
            {AGENTS.map((agent, i) => {
              const state    = getAgentState(agent.id)
              const isSelected = selectedAgent === agent.id
              const isClickable = state === "done" || state === "active"

              return (
                <div
                  key={agent.id}
                  className={`agent-row agent-${state} ${isSelected ? "agent-selected" : ""} ${isClickable ? "agent-clickable" : ""}`}
                  onClick={() => isClickable && setSelectedAgent(
                    isSelected ? null : agent.id
                  )}
                >
                  {/* Connector line */}
                  {i > 0 && <div className={`connector connector-${state}`} />}

                  {/* Status dot */}
                  <div className={`agent-dot dot-${state}`}>
                    {state === "done"   && <span className="dot-icon">✓</span>}
                    {state === "active" && <span className="dot-spinner" />}
                    {state === "idle"   && <span className="dot-empty" />}
                  </div>

                  {/* Agent info */}
                  <div className="agent-info">
                    <div className="agent-row-name">
                      <span className="agent-row-icon">{agent.icon}</span>
                      {agent.label}
                    </div>
                    <div className="agent-row-desc">{agent.desc}</div>
                  </div>

                  {/* Click hint */}
                  {isClickable && (
                    <div className="agent-view-hint">
                      {isSelected ? "↑" : "→"}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {status === "complete" && (
          <div className="stats-row">
            <div className="stat-item">
              <div className="stat-value">{papersFound}</div>
              <div className="stat-label">Papers</div>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <div className="stat-value">5</div>
              <div className="stat-label">Agents</div>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <div className="stat-value">1</div>
              <div className="stat-label">Report</div>
            </div>
          </div>
        )}
      </aside>

      {/* ── Right panel ── */}
      <div className="right-panel">

        {/* Agent detail view — shown when an agent is clicked */}
        {selectedAgent && (
          <div className="agent-detail">
            <div className="agent-detail-header">
              <div className="agent-detail-title">
                <span className="agent-detail-icon">{selectedAgentData?.icon}</span>
                {selectedAgentData?.label} Agent — Output
              </div>
              <button className="btn-close" onClick={() => setSelectedAgent(null)}>✕</button>
            </div>
            <div className="agent-detail-body">
              {selectedAgentLogs.length === 0 ? (
                <div className="detail-empty">No logs captured for this agent yet.</div>
              ) : (
                selectedAgentLogs.map((log, i) => (
                  <div className="detail-log-row" key={i}>
                    <span className="detail-log-text">{log.text}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Tab bar */}
        <div className="tab-bar">
          <button
            className={`tab ${activeTab === "logs" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("logs")}
          >
            Agent Activity
            {status === "running" && <span className="tab-live">● LIVE</span>}
          </button>
          <button
            className={`tab ${activeTab === "report" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("report")}
            disabled={!report}
          >
            Research Report
            {report && <span className="tab-ready">Ready</span>}
          </button>
        </div>

        {/* Logs tab */}
        {activeTab === "logs" && (
          <div className="logs-panel">
            {logs.length === 0 && status === "running" && (
              <div className="logs-starting">
                <div className="spinner" />
                Starting pipeline...
              </div>
            )}
            <div className="logs-feed">
              {logs.map((log, i) => (
                <div className="log-entry" key={i}>
                  <span className="log-agent-tag">{log.agent}</span>
                  <span className="log-message">{log.text}</span>
                </div>
              ))}
              {status === "running" && logs.length > 0 && (
                <div className="log-entry log-thinking">
                  <span className="log-agent-tag">{activeAgent}</span>
                  <span className="thinking-dots">
                    <span /><span /><span />
                  </span>
                </div>
              )}
              <div ref={logEndRef} />
            </div>
          </div>
        )}

        {/* Report tab */}
        {activeTab === "report" && (
          <div className="report-panel">
            {error && (
              <div className="error-banner">
                <strong>Pipeline Error:</strong> {error}
                <div className="error-hint">Make sure backend is running: <code>uvicorn main:app --reload</code></div>
              </div>
            )}
            {report ? (
              <>
                <div className="report-toolbar">
                  <div className="report-toolbar-info">
                    Research Report · {report.split(" ").length} words
                  </div>
                  <button className="btn-download" onClick={() => window.print()}>
                    ↓ Download PDF
                  </button>
                </div>
                <div className="report-content" id="printable-report">
                  <ReactMarkdown>{report}</ReactMarkdown>
                </div>
              </>
            ) : (
              <div className="report-waiting">
                <div className="spinner" />
                Report will appear here when the pipeline completes...
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
