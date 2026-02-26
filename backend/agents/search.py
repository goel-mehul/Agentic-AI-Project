"""
search.py — The Search Agent
=============================

WHAT IT DOES:
    1. Takes the search queries from the Planner
    2. Fetches matching papers from arXiv (free, no API key)
    3. Also queries Semantic Scholar (free, no API key)
    4. Stores all paper abstracts in ChromaDB (local vector database)
    5. Retrieves the top-k most semantically relevant chunks

WHY IT EXISTS:
    Raw search returns too many loosely related papers. ChromaDB lets us
    find the MOST relevant sections using semantic similarity — meaning
    we compare meaning, not just keywords. This is the RAG pattern.

HOW IT FITS:
    Second node in the pipeline.
    Reads:  research_plan (from Planner), research_question
    Writes: raw_papers, retrieved_chunks, agent_logs

WHAT YOU'RE LEARNING:
    - How to call external APIs (arXiv, Semantic Scholar)
    - How vector databases work (ChromaDB)
    - The RAG (Retrieval-Augmented Generation) pattern
    - How to deduplicate and clean data
"""

import requests
import chromadb
from chromadb.utils import embedding_functions
from .state import ResearchState

try:
    import arxiv
except ImportError:
    arxiv = None


# ── ChromaDB Setup ────────────────────────────────────────────────────────────
# ChromaDB is a local vector database. It runs entirely on your machine —
# no external service, no API key, completely free.
#
# The embedding function converts text into vectors (lists of numbers).
# Similar texts get similar vectors. This is how semantic search works.

chroma_client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()


# ── arXiv Search ──────────────────────────────────────────────────────────────

def _search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv for academic papers.

    arXiv is a free preprint server used by researchers in CS, Physics,
    Math, and more. The Python `arxiv` library wraps their API.
    No authentication needed — completely open.
    """
    if arxiv is None:
        print("[Search] arxiv package not installed, skipping")
        return []

    papers = []
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for result in search.results():
            papers.append({
                "paper_id": result.entry_id,
                "title": result.title,
                "authors": [str(a) for a in result.authors[:3]],
                "abstract": result.summary,
                "published": str(result.published.date()),
                "source": "arxiv",
                "url": result.pdf_url or result.entry_id,
                "relevance_score": 1.0
            })
    except Exception as e:
        print(f"[Search] arXiv error for '{query}': {e}")

    return papers


# ── Semantic Scholar Search ───────────────────────────────────────────────────

def _search_semantic_scholar(query: str, max_results: int = 3) -> list[dict]:
    """
    Search Semantic Scholar's free public API.

    Semantic Scholar indexes 200M+ papers and provides citation data.
    The basic search API requires no key (though there are rate limits).
    We use it as a second source to complement arXiv.
    """
    papers = []
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,year,url,citationCount"
        }
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code == 200:
            for p in resp.json().get("data", []):
                # Skip papers without abstracts — not useful for our RAG step
                if not p.get("abstract"):
                    continue
                papers.append({
                    "paper_id": p.get("paperId", ""),
                    "title": p.get("title", ""),
                    "authors": [a["name"] for a in p.get("authors", [])[:3]],
                    "abstract": p.get("abstract", ""),
                    "published": str(p.get("year", "Unknown")),
                    "source": "semantic_scholar",
                    "url": p.get("url", ""),
                    "relevance_score": 0.9,
                    "citation_count": p.get("citationCount", 0)
                })
    except Exception as e:
        print(f"[Search] Semantic Scholar error for '{query}': {e}")

    return papers


# ── ChromaDB: Store + Retrieve ────────────────────────────────────────────────

def _store_and_retrieve(
    papers: list[dict],
    research_question: str,
    session_id: str,
    top_k: int = 8
) -> list[dict]:
    """
    Store papers in ChromaDB and retrieve the most relevant ones.

    HOW THIS WORKS:
    1. Each paper's title + abstract gets converted to a vector embedding
       (a list of ~384 numbers that captures the semantic meaning)
    2. The research question also gets converted to a vector
    3. ChromaDB finds the papers whose vectors are closest to the question vector
    4. "Closest" = most semantically similar = most relevant

    This is the core of RAG. We retrieve BEFORE generating so the Writer
    only sees the most useful evidence.
    """
    if not papers:
        return []

    # Fresh collection per session to avoid data leaking between runs
    collection_name = f"papers_{session_id[:8]}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    # Build lists for ChromaDB batch insert
    docs, ids, metadatas = [], [], []
    seen_ids = set()

    for i, paper in enumerate(papers):
        uid = f"doc_{i}"
        if uid in seen_ids or not paper.get("abstract"):
            continue
        seen_ids.add(uid)

        # We embed title + abstract together for richer semantic signal
        text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        docs.append(text)
        ids.append(uid)
        metadatas.append({
            "title":    paper["title"],
            "authors":  ", ".join(paper["authors"]),
            "published": paper["published"],
            "source":   paper["source"],
            "url":      paper["url"],
            "paper_id": paper["paper_id"]
        })

    if not docs:
        return []

    # Insert all papers into ChromaDB
    collection.add(documents=docs, ids=ids, metadatas=metadatas)

    # Query: find top_k papers most similar to the research question
    results = collection.query(
        query_texts=[research_question],
        n_results=min(top_k, len(docs))
    )

    # Format results for the next agents
    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({"content": doc, "metadata": meta})

    return chunks


# ── Agent Function ────────────────────────────────────────────────────────────

def search_agent(state: ResearchState) -> ResearchState:
    iteration = state.get("search_iteration", 0)
    gap_queries = state.get("gap_queries", [])
    question = state["research_question"]
    session_id = state["session_id"]

    state["current_agent"] = "search"

    # Decide which queries to run
    if iteration == 0:
        # First pass: use the Planner's queries
        queries = state["research_plan"]
        state["agent_logs"] = [f"🔍 Search: Pass 1 — running {len(queries)} planned queries..."]
    else:
        # Subsequent pass: target gaps found by the Critic
        queries = gap_queries
        state["agent_logs"] = [
            f"🔄 Search: Pass {iteration + 1} — running {len(queries)} gap-filling queries...",
            f"🎯 Targeting gaps: {', '.join(gap_queries[:3])}"
        ]

    # Track which papers we already have to avoid duplicates
    existing_titles = {p.get("title", "").lower() for p in state.get("raw_papers", [])}
    new_papers = []

    for query in queries:
        state["agent_logs"] = [f"🔍 Search: Querying for '{query}'"]
        arxiv_papers = _search_arxiv(query, max_results=4)
        ss_papers    = _search_semantic_scholar(query, max_results=2)

        for paper in arxiv_papers + ss_papers:
            title = paper.get("title", "").lower()
            if title and title not in existing_titles:
                existing_titles.add(title)
                new_papers.append(paper)

    # Merge with existing papers
    all_papers = state.get("raw_papers", []) + new_papers

    state["raw_papers"] = all_papers
    state["search_iteration"] = iteration + 1

    # Update citation counts
    citation_counts = state.get("citation_counts", {})
    for paper in new_papers:
        pid = paper.get("paper_id", "")
        count = paper.get("citation_count", 0)
        if pid and count:
            citation_counts[pid] = count
    state["citation_counts"] = citation_counts

    state["agent_logs"] = [
        f"📚 Search: {len(new_papers)} new papers found (total: {len(all_papers)})",
        "🧮 Search: Re-indexing vector store with new papers..."
    ]

    # Re-index everything in ChromaDB and retrieve
    chunks = _store_and_retrieve(all_papers, question, session_id)
    state["retrieved_chunks"] = chunks

    state["agent_logs"] = [
        f"✅ Search: Retrieved {len(chunks)} most relevant sections"
    ]

    return state