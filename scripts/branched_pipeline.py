"""
branched_pipeline.py — Branched RAG (Hybrid Search) Pipeline

Two retrieval paths run in parallel, results merged:

  Path A — Vector search (semantic):
    Query → 1024-dim vector → ChromaDB cosine similarity → top-5 chunks
    Good at: meaning, synonyms, paraphrases

  Path B — Keyword search (BM25):
    Query → tokenized → BM25 term frequency scoring → top-5 chunks
    Good at: exact terms, rare words, domain-specific phrases

  Merge:
    Deduplicate by chunk_id → keep top-3 by combined score

Usage:
    python3 scripts/branched_pipeline.py

Output:
    results/branched_results_<timestamp>.json
    Prints side-by-side comparison with standard pipeline results
"""

import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
QUESTIONS_CSV  = ROOT / "data" / "questions.csv"
CHUNKS_FILE    = ROOT / "data" / "chunks.json"
CHROMA_DIR     = ROOT / "chroma_db"
RESULTS_DIR    = ROOT / "results"

load_dotenv(ROOT / ".env")

MODEL           = "claude-opus-4-5"
MAX_TOKENS      = 512
COLLECTION_NAME = "cooking_techniques"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
TOP_K_EACH      = 5   # retrieve top-5 from each path
TOP_K_FINAL     = 3   # keep top-3 after merge

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
    import csv
    with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def tokenize(text: str) -> list[str]:
    """Simple tokenizer — lowercase, split on non-alphanumeric."""
    return re.findall(r'[a-z0-9]+', text.lower())


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    """Build BM25 index from all chunks."""
    corpus = [tokenize(c["content"]) for c in chunks]
    return BM25Okapi(corpus)


# ── Path A — Vector search ────────────────────────────────────────────────────
def vector_retrieve(collection, question: str, top_k: int) -> list[dict]:
    results = collection.query(query_texts=[question], n_results=top_k)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "chunk_id":     f"{meta['doc_id']}__{meta['section'].replace(' ', '_').lower()}",
            "doc_id":       meta["doc_id"],
            "topic":        meta["topic"],
            "section":      meta["section"],
            "content":      doc,
            "vector_dist":  round(dist, 4),
            "vector_rank":  None,   # set after sorting
            "bm25_score":   None,
            "bm25_rank":    None,
            "source":       "vector",
        })
    for i, c in enumerate(chunks):
        c["vector_rank"] = i + 1
    return chunks


# ── Path B — BM25 keyword search ──────────────────────────────────────────────
def bm25_retrieve(bm25: BM25Okapi, all_chunks: list[dict], question: str, top_k: int) -> list[dict]:
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)

    # Pair scores with chunks, sort descending
    scored = sorted(
        [(score, chunk) for score, chunk in zip(scores, all_chunks)],
        key=lambda x: x[0],
        reverse=True,
    )[:top_k]

    results = []
    for rank, (score, chunk) in enumerate(scored, 1):
        results.append({
            "chunk_id":    chunk["chunk_id"],
            "doc_id":      chunk["doc_id"],
            "topic":       chunk["topic"],
            "section":     chunk["section"],
            "content":     chunk["content"],
            "vector_dist": None,
            "vector_rank": None,
            "bm25_score":  round(float(score), 4),
            "bm25_rank":   rank,
            "source":      "bm25",
        })
    return results


# ── Merge — deduplicate, score, keep top-K ────────────────────────────────────
def merge(vector_chunks: list[dict], bm25_chunks: list[dict],
          top_k: int, total_candidates: int) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) — standard merge for hybrid search.
    Score = 1/(rank + 60) for each path. Sums across paths.
    Higher score = more relevant overall.
    """
    K = 60  # RRF constant — dampens effect of very high ranks
    scores: dict[str, float] = {}
    by_id:  dict[str, dict]  = {}

    for rank, chunk in enumerate(vector_chunks, 1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rank + K)
        by_id[cid]  = chunk

    for rank, chunk in enumerate(bm25_chunks, 1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rank + K)
        if cid not in by_id:
            by_id[cid] = chunk
        else:
            # Merge scores into existing entry
            by_id[cid]["bm25_score"] = chunk["bm25_score"]
            by_id[cid]["bm25_rank"]  = chunk["bm25_rank"]
            by_id[cid]["source"]     = "both"

    # Sort by RRF score, return top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    merged = []
    for cid, rrf_score in ranked:
        chunk = dict(by_id[cid])
        chunk["rrf_score"] = round(rrf_score, 6)
        merged.append(chunk)
    return merged


# ── Generation ────────────────────────────────────────────────────────────────
def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['topic']} — {chunk['section']}]\n{chunk['content']}"
        )
    return GROUNDED_PROMPT.format(context="\n\n".join(context_parts), question=question)


def generate_answer(client: anthropic.Anthropic, prompt: str) -> tuple[str, int]:
    start = time.time()
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

    client     = anthropic.Anthropic(api_key=api_key)
    collection = load_collection()
    all_chunks = load_chunks()
    questions  = load_questions()
    total      = len(questions)

    print(f"\n🌿 Building BM25 index over {len(all_chunks)} chunks...")
    bm25 = build_bm25(all_chunks)
    print(f"   ✅ BM25 index ready\n")

    # Load standard results for comparison
    std_file    = find_latest_standard_results()
    std_data    = json.loads(std_file.read_text(encoding="utf-8"))
    std_results = {r["id"]: r for r in std_data["results"]}

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"🌿 VP RAG Eval — Branched RAG (Hybrid Search)")
    print(f"   Path A : Vector search — {EMBEDDING_MODEL} (top-{TOP_K_EACH})")
    print(f"   Path B : BM25 keyword search (top-{TOP_K_EACH})")
    print(f"   Merge  : Reciprocal Rank Fusion → top-{TOP_K_FINAL}")
    print(f"   Compare: {std_file.name}\n")
    print(f"   {'Q':<4} {'Expected':<25} {'Standard':^10} {'Branched':^10} {'Sources':^12}")
    print(f"   {'-'*65}")

    results = []

    for row in questions:
        qid        = row["id"]
        question   = row["question"]
        expected   = row["expected_doc_id"]
        difficulty = row["difficulty"]

        # Path A — vector
        vector_chunks = vector_retrieve(collection, question, TOP_K_EACH)

        # Path B — BM25
        bm25_chunks = bm25_retrieve(bm25, all_chunks, question, TOP_K_EACH)

        # Merge
        chunks = merge(vector_chunks, bm25_chunks, TOP_K_FINAL, len(all_chunks))

        retrieved_doc_ids = [c["doc_id"] for c in chunks]
        retrieval_correct = expected in retrieved_doc_ids
        top_hit           = retrieved_doc_ids[0] if retrieved_doc_ids else ""
        sources           = "+".join(sorted(set(c["source"] for c in chunks)))

        # Generate
        prompt = build_prompt(question, chunks)
        try:
            answer, duration_ms = generate_answer(client, prompt)
            status = "success"
        except Exception as e:
            answer, duration_ms, status = f"ERROR: {e}", 0, "error"

        # Compare
        std         = std_results.get(qid, {})
        std_correct = std.get("retrieval_correct", False)
        std_icon    = "✅" if std_correct       else "❌"
        br_icon     = "✅" if retrieval_correct else "❌"
        change      = "⬆️  FIXED" if not std_correct and retrieval_correct else \
                      "⬇️  BROKE" if std_correct and not retrieval_correct else ""

        print(f"   Q{qid:<3} {expected:<25} {std_icon:^10} {br_icon:^10} {sources:<12} {change}")

        results.append({
            "id":                qid,
            "question":          question,
            "difficulty":        difficulty,
            "expected_doc_id":   expected,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_correct": retrieval_correct,
            "top_hit_doc_id":    top_hit,
            "top_hit_topic":     chunks[0]["topic"]    if chunks else "",
            "top_hit_section":   chunks[0]["section"]  if chunks else "",
            "chunks":            chunks,
            "answer":            answer,
            "status":            status,
            "duration_ms":       duration_ms,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    std_correct_count = sum(1 for r in std_data["results"] if r["retrieval_correct"])
    br_correct_count  = sum(1 for r in results if r["retrieval_correct"])
    std_pct = round(std_correct_count / total * 100, 1)
    br_pct  = round(br_correct_count  / total * 100, 1)
    diff    = round(br_pct - std_pct, 1)
    diff_str = f"+{diff}%" if diff > 0 else f"{diff}%"

    print(f"\n{'='*55}")
    print(f"  Standard RAG  : {std_pct}%  ({std_correct_count}/{total})")
    print(f"  Branched RAG  : {br_pct}%  ({br_correct_count}/{total})  {diff_str}")
    print(f"{'='*55}")

    output = {
        "metadata": {
            "timestamp":          timestamp,
            "model":              MODEL,
            "embedding_model":    EMBEDDING_MODEL,
            "bm25":               "BM25Okapi (rank-bm25)",
            "merge_method":       "Reciprocal Rank Fusion (RRF, K=60)",
            "top_k_each":         TOP_K_EACH,
            "top_k_final":        TOP_K_FINAL,
            "total":              total,
            "retrieval_correct":  br_correct_count,
            "retrieval_accuracy": f"{br_pct}%",
            "standard_accuracy":  f"{std_pct}%",
            "delta":              diff_str,
            "compare_against":    std_file.name,
        },
        "results": results,
    }
    output_file = RESULTS_DIR / f"branched_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
