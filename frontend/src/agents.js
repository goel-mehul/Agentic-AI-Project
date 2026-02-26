/**
 * Agent metadata used throughout the UI.
 * Centralizing this here means we only update one place
 * if we add or rename agents.
 */
export const AGENTS = [
  {
    id:    "planner",
    icon:  "◈",
    label: "Planner",
    desc:  "Decomposing research question into search queries"
  },
  {
    id:    "search",
    icon:  "◎",
    label: "Search",
    desc:  "Retrieving papers from arXiv & Semantic Scholar"
  },
  {
    id:    "critic",
    icon:  "◐",
    label: "Critic",
    desc:  "Evaluating evidence quality and contradictions"
  },
  {
    id:    "writer",
    icon:  "◉",
    label: "Writer",
    desc:  "Synthesizing structured research report"
  },
  {
    id:    "fact_checker",
    icon:  "◑",
    label: "Fact Checker",
    desc:  "Verifying claims against source evidence"
  },
]

export const AGENT_IDS = AGENTS.map(a => a.id)