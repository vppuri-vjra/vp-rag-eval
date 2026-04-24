"""
graph_pipeline.py — Graph RAG Pipeline (Step 6f)

How Graph RAG differs from all previous pipelines:

  Fixed pipelines:
    Question → vector search → top-3 chunks → generate

  Graph RAG:
    Question → vector search → entry doc (top-1)
             → graph traversal → neighbor docs (connected by shared concepts)
             → retrieve best chunk from each neighbor
             → combined context (entry + neighbors) → generate

Why this helps:
  Some questions need knowledge from CONNECTED documents, not just the nearest one.
  Example:
    Q9 "pan sauce" → vector lands on reduction doc
                   → graph: reduction ↔ deglazing (share: fond, pan sauce, liquid reduction)
                   → traversal pulls deglazing chunk too
                   → combined context has the right answer

Graph structure (built by build_graph.py):
  Nodes  = 20 docs
  Edges  = docs sharing >= 2 concepts
  Weight = number of shared concepts

Usage:
    python3 scripts/build_graph.py     # run once to build the graph
    python3 scripts/graph_pipeline.py  # run to evaluate

Output:
    results/graph_results_<timestamp>.json
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import anthropic
import chromadb
import networkx as nx
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
QUESTIONS_CSV  = ROOT / "data" / "questions.csv"
GRAPH_FILE     = ROOT / "data" / "knowledge_graph.json"
CHROMA_DIR     = ROOT / "chroma_db"
RESULTS_DIR    = ROOT / "results"

load_dotenv(ROOT / ".env")

MODEL            = "claude-opus-4-5"
MAX_TOKENS       = 512
COLLECTION_NAME  = "cooking_techniques"
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"

TOP_K_ENTRY      = 3   # chunks to retrieve from entry doc
TOP_K_NEIGHBOR   = 1   # best chunk from each neighbor doc
MAX_NEIGHBORS    = 2   # max neighbor docs to include (keeps context bounded)

GROUNDED_PROMPT = """\
Answer the question using ONLY the information provided below.
Do not use outside knowledge. If the answer is not in the provided \
context, say "I cannot answer this from the provided information."

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_questions() -> list[dict]:
    with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def load_graph(graph_file: Path) -> tuple[nx.Graph, dict]:
    """Load graph from JSON. Returns (NetworkX graph, doc_concepts dict)."""
    data  = json.loads(graph_file.read_text(encoding="utf-8"))
    G     = nx.Graph()

    # Add nodes
    for doc_id, concepts in data["doc_concepts"].items():
        G.add_node(doc_id, concepts=concepts)

    # Add edges
    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["weight"],
            shared_concepts=edge["shared_concepts"],
        )

    return G, data["doc_concepts"]


def vector_retrieve(collection, question: str, top_k: int,
                    where: dict = None) -> list[dict]:
    """
    Retrieve top-k chunks from ChromaDB.
    Optional `where` filter to restrict to a specific doc_id.
    """
    kwargs = dict(query_texts=[question], n_results=top_k)
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    chunks  = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "chunk_id": f"{meta['doc_id']}__{meta['section'].replace(' ', '_').lower()}",
            "doc_id":   meta["doc_id"],
            "topic":    meta["topic"],
            "section":  meta["section"],
            "content":  doc,
            "distance": round(dist, 4),
            "source":   "entry",
        })
    return chunks


def graph_retrieve(collection, G: nx.Graph, question: str,
                   entry_doc_id: str) -> list[dict]:
    """
    Graph RAG retrieval:
      1. Get entry chunks from the top vector-search doc
      2. Find neighbors in graph (sorted by edge weight)
      3. Retrieve best chunk from each neighbor doc
      4. Return combined context
    """
    all_chunks = []
    seen_docs  = {entry_doc_id}

    # ── Entry: get top-K chunks from the entry doc ────────────────────────────
    entry_chunks = vector_retrieve(
        collection, question, TOP_K_ENTRY,
        where={"doc_id": entry_doc_id}
    )
    for c in entry_chunks:
        c["source"] = "entry"
    all_chunks.extend(entry_chunks)

    # ── Graph traversal: get neighbors sorted by shared concept count ─────────
    if entry_doc_id in G:
        neighbors = sorted(
            G.neighbors(entry_doc_id),
            key=lambda n: G[entry_doc_id][n]["weight"],
            reverse=True,
        )[:MAX_NEIGHBORS]

        for neighbor_id in neighbors:
            if neighbor_id in seen_docs:
                continue
            seen_docs.add(neighbor_id)

            # Get best chunk from this neighbor doc
            neighbor_chunks = vector_retrieve(
                collection, question, TOP_K_NEIGHBOR,
                where={"doc_id": neighbor_id}
            )
            for c in neighbor_chunks:
                c["source"]          = "graph_neighbor"
                c["neighbor_of"]     = entry_doc_id
                c["shared_concepts"] = G[entry_doc_id][neighbor_id]["shared_concepts"]
            all_chunks.extend(neighbor_chunks)

    return all_chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        label = chunk.get("source", "")
        note  = f" [via graph: {', '.join(chunk.get('shared_concepts', [])[:2])}]" \
                if label == "graph_neighbor" else ""
        context_parts.append(
            f"[Source {i}: {chunk['topic']} — {chunk['section']}{note}]\n{chunk['content']}"
        )
    return GROUNDED_PROMPT.format(context="\n\n".join(context_parts), question=question)


def generate_answer(client: anthropic.Anthropic, prompt: str) -> tuple[str, int]:
    start   = time.time()
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    duration_ms = round((time.time() - start) * 1000)
    return message.content[0].text.strip(), duration_ms


def find_latest_standard_results() -> Path:
    files = sorted(RESULTS_DIR.glob("rag_results_*.json"))
    if not files:
        raise FileNotFoundError("No rag_results_*.json found")
    return files[-1]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph not found: {GRAPH_FILE}\n"
            "Run: python3 scripts/build_graph.py first"
        )

    client     = anthropic.Anthropic(api_key=api_key)
    collection = load_collection()
    questions  = load_questions()
    total      = len(questions)

    print(f"\n🕸️  Loading knowledge graph...")
    G, doc_concepts = load_graph(GRAPH_FILE)
    print(f"   ✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    std_file    = find_latest_standard_results()
    std_data    = json.loads(std_file.read_text(encoding="utf-8"))
    std_results = {r["id"]: r for r in std_data["results"]}

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"🕸️  VP RAG Eval — Graph RAG Pipeline (Step 6f)")
    print(f"   Entry    : vector search top-{TOP_K_ENTRY} chunks from entry doc")
    print(f"   Neighbors: up to {MAX_NEIGHBORS} neighbor docs via graph, top-{TOP_K_NEIGHBOR} chunk each")
    print(f"   Compare  : {std_file.name}\n")
    print(f"   {'Q':<4} {'Expected':<25} {'Std':^6} {'Graph':^6} {'Entry doc':<30} {'Neighbors'}")
    print(f"   {'-'*90}")

    results = []

    for row in questions:
        qid        = row["id"]
        question   = row["question"]
        expected   = row["expected_doc_id"]
        difficulty = row["difficulty"]

        # Step 1: vector search to find entry doc (top-1 doc from standard search)
        entry_chunks = vector_retrieve(collection, question, top_k=1)
        entry_doc_id = entry_chunks[0]["doc_id"] if entry_chunks else ""

        # Step 2: graph retrieval — entry + neighbors
        start  = time.time()
        chunks = graph_retrieve(collection, G, question, entry_doc_id)

        retrieved_doc_ids = list(dict.fromkeys(c["doc_id"] for c in chunks))
        retrieval_correct = expected in retrieved_doc_ids
        top_hit           = retrieved_doc_ids[0] if retrieved_doc_ids else ""

        neighbor_ids = [c["doc_id"] for c in chunks if c["source"] == "graph_neighbor"]

        # Step 3: generate
        prompt = build_prompt(question, chunks)
        try:
            answer, duration_ms = generate_answer(client, prompt)
            status = "success"
        except Exception as e:
            answer, duration_ms, status = f"ERROR: {e}", 0, "error"

        std         = std_results.get(qid, {})
        std_correct = std.get("retrieval_correct", False)
        std_icon    = "✅" if std_correct       else "❌"
        gr_icon     = "✅" if retrieval_correct else "❌"
        change      = " ⬆️FIXED" if not std_correct and retrieval_correct else \
                      " ⬇️BROKE" if std_correct and not retrieval_correct else ""

        neighbor_str = ", ".join(n.replace("doc_", "").replace("_", " ") for n in neighbor_ids) or "—"
        print(f"   Q{qid:<3} {expected:<25} {std_icon:^6} {gr_icon:^6} {entry_doc_id:<30} {neighbor_str}{change}")

        results.append({
            "id":                qid,
            "question":          question,
            "difficulty":        difficulty,
            "expected_doc_id":   expected,
            "entry_doc_id":      entry_doc_id,
            "neighbor_doc_ids":  neighbor_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_correct": retrieval_correct,
            "top_hit_doc_id":    top_hit,
            "num_chunks":        len(chunks),
            "chunks":            chunks,
            "answer":            answer,
            "status":            status,
            "duration_ms":       duration_ms,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    std_correct_count = sum(1 for r in std_data["results"] if r["retrieval_correct"])
    gr_correct_count  = sum(1 for r in results if r["retrieval_correct"])
    std_pct = round(std_correct_count / total * 100, 1)
    gr_pct  = round(gr_correct_count  / total * 100, 1)
    diff    = round(gr_pct - std_pct, 1)
    diff_str = f"+{diff}%" if diff > 0 else f"{diff}%"

    print(f"\n{'='*60}")
    print(f"  Standard RAG  : {std_pct}%  ({std_correct_count}/{total})")
    print(f"  Graph RAG     : {gr_pct}%  ({gr_correct_count}/{total})  {diff_str}")
    print(f"{'='*60}")

    output = {
        "metadata": {
            "timestamp":          timestamp,
            "model":              MODEL,
            "embedding_model":    EMBEDDING_MODEL,
            "pattern":            "Graph RAG",
            "graph_nodes":        G.number_of_nodes(),
            "graph_edges":        G.number_of_edges(),
            "top_k_entry":        TOP_K_ENTRY,
            "top_k_neighbor":     TOP_K_NEIGHBOR,
            "max_neighbors":      MAX_NEIGHBORS,
            "total":              total,
            "retrieval_correct":  gr_correct_count,
            "retrieval_accuracy": f"{gr_pct}%",
            "standard_accuracy":  f"{std_pct}%",
            "delta":              diff_str,
            "compare_against":    std_file.name,
        },
        "results": results,
    }
    output_file = RESULTS_DIR / f"graph_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
