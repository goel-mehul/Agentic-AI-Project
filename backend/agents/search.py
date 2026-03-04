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
import time
import io
import re 
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
        time.sleep(3)  # ← add this — respects arXiv's rate limit
    except Exception as e:
        print(f"[Search] arXiv error for '{query}': {e}")
        time.sleep(5)  # ← longer wait after an error before retrying
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

# ── PDF Download & Extraction ─────────────────────────────────────────────────

def _download_and_extract_pdf(url: str) -> str | None:
    """
    Download a PDF from arXiv and extract clean text.
    Returns None if download fails or text is too short to be useful.
    """
    try:
        import fitz  # pymupdf
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None

        # Load PDF from bytes
        doc = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        # Clean up the text
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # collapse excessive newlines
        full_text = re.sub(r'[ \t]+', ' ', full_text)      # collapse spaces

        # Must be substantial to be worth using
        if len(full_text.split()) < 500:
            return None

        return full_text.strip()

    except Exception as e:
        print(f"[Search] PDF extraction failed for {url}: {e}")
        return None


def _chunk_by_section(full_text: str, paper_title: str) -> list[str]:
    """
    Split full paper text into meaningful chunks by section.
    Falls back to fixed-size chunking if sections aren't detectable.
    Each chunk is prefixed with the paper title for context.
    """
    # Common section headers in academic papers
    section_pattern = re.compile(
        r'\n(?:Abstract|Introduction|Background|Related Work|'
        r'Methodology|Methods|Approach|Experiments|Results|'
        r'Evaluation|Discussion|Conclusion|Limitations|'
        r'References|Acknowledgements)\s*\n',
        re.IGNORECASE
    )

    sections = section_pattern.split(full_text)
    section_names = section_pattern.findall(full_text)

    chunks = []

    if len(sections) >= 3:
        # Successfully split by sections — skip references and acknowledgements
        skip_sections = {'references', 'acknowledgements', 'acknowledgments'}
        for i, section_text in enumerate(sections):
            section_name = section_names[i - 1].strip() if i > 0 else "Introduction"
            if section_name.lower() in skip_sections:
                continue
            text = section_text.strip()
            if len(text.split()) < 50:  # skip tiny sections
                continue
            chunk = f"Title: {paper_title}\nSection: {section_name}\n\n{text[:2000]}"
            chunks.append(chunk)
    else:
        # Fallback: fixed-size chunks of ~400 words with overlap
        words = full_text.split()
        chunk_size = 400
        overlap = 50
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:
                continue
            chunk = f"Title: {paper_title}\n\n{' '.join(chunk_words)}"
            chunks.append(chunk)

    return chunks

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

    # Sort papers by citation count — top 3 get full PDF treatment
    sorted_papers = sorted(
        papers,
        key=lambda p: p.get("citation_count", 0),
        reverse=True
    )
    top_paper_ids = {
        p.get("paper_id") for p in sorted_papers[:3]
        if p.get("source") == "arxiv"  # only arXiv has free PDFs
    }

    chunk_counter = 0
    for i, paper in enumerate(papers):
        if not paper.get("abstract"):
            continue

        meta = {
            "title":     paper["title"],
            "authors":   ", ".join(paper["authors"]),
            "published": paper["published"],
            "source":    paper["source"],
            "url":       paper["url"],
            "paper_id":  paper["paper_id"]
        }

        # Try full PDF for top 3 arXiv papers
        if paper.get("paper_id") in top_paper_ids and paper.get("url"):
            print(f"[Search] Downloading full PDF: {paper['title'][:50]}...")
            full_text = _download_and_extract_pdf(paper["url"])
            if full_text:
                section_chunks = _chunk_by_section(full_text, paper["title"])
                for j, chunk in enumerate(section_chunks):
                    uid = f"doc_{i}_chunk_{j}"
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        docs.append(chunk)
                        ids.append(uid)
                        metadatas.append(meta)
                        chunk_counter += 1
                print(f"[Search] Extracted {len(section_chunks)} sections from full paper")
                continue  # skip abstract fallback

        # Fallback: abstract only
        uid = f"doc_{i}"
        if uid not in seen_ids:
            seen_ids.add(uid)
            text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            docs.append(text)
            ids.append(uid)
            metadatas.append(meta)
            chunk_counter += 1

    if not docs:
        return []

    # Insert all papers into ChromaDB
    collection.add(documents=docs, ids=ids, metadatas=metadatas)

    # Dynamic retrieval: between 8 and 16 chunks based on available evidence
    min_chunks = 8
    max_chunks = 16
    n_results = min(max_chunks, max(min_chunks, len(docs)))
    n_results = min(n_results, len(docs))  # can't retrieve more than we have

    results = collection.query(
        query_texts=[research_question],
        n_results=n_results
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