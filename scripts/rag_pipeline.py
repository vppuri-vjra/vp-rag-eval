"""
rag_pipeline.py — RAG Pipeline for VP RAG Eval

For each question:
  1. RETRIEVE  — search ChromaDB for top 3 relevant chunks
  2. BUILD     — inject chunks into Claude's context (grounded prompt)
  3. GENERATE  — Claude answers using only retrieved content
  4. SAVE      — store question + chunks + answer + metadata

Usage:
    python3 scripts/rag_pipeline.py

Output:
    results/rag_results_<timestamp>.json
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
QUESTIONS_CSV  = ROOT / "data" / "questions.csv"
CHROMA_DIR     = ROOT / "chroma_db"
RESULTS_DIR    = ROOT / "results"

load_dotenv(ROOT / ".env")

MODEL            = "claude-opus-4-5"
MAX_TOKENS       = 512
COLLECTION_NAME  = "cooking_techniques"
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
TOP_K            = 3  # number of chunks to retrieve per question


# ── Prompt template ───────────────────────────────────────────────────────────
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
    """Load ChromaDB collection with same embedding function used at index time."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )


def retrieve(collection, question: str, top_k: int = TOP_K) -> list[dict]:
    """Search ChromaDB and return top K chunks with metadata."""
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "chunk_id":        f"{meta['doc_id']}__{meta['section'].replace(' ', '_').lower()}",
            "doc_id":          meta["doc_id"],
            "topic":           meta["topic"],
            "section":         meta["section"],
            "content":         doc,
            "distance":        round(dist, 4),  # cosine distance — lower = more similar
        })
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build grounded prompt — inject retrieved chunks as context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['topic']} — {chunk['section']}]\n{chunk['content']}"
        )
    context = "\n\n".join(context_parts)
    return GROUNDED_PROMPT.format(context=context, question=question)


def generate(client: anthropic.Anthropic, prompt: str) -> tuple[str, float]:
    """Send prompt to Claude and return answer + duration."""
    start = time.time()
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    duration_ms = round((time.time() - start) * 1000)
    return message.content[0].text.strip(), duration_ms


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

    client     = anthropic.Anthropic(api_key=api_key)
    collection = load_collection()
    questions  = load_questions()
    total      = len(questions)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n🔗 VP RAG Eval — RAG Pipeline")
    print(f"   Model      : {MODEL}")
    print(f"   Questions  : {total}")
    print(f"   Top-K      : {TOP_K} chunks per question")
    print(f"   Grounded   : Yes — Claude restricted to retrieved context\n")

    results = []

    for i, row in enumerate(questions, 1):
        qid        = row["id"]
        question   = row["question"]
        expected   = row["expected_doc_id"]
        difficulty = row["difficulty"]

        print(f"[{i:02d}/{total}] Q{qid} ({difficulty}) — {question[:60]}...")

        # Step 1 — Retrieve
        chunks = retrieve(collection, question)
        retrieved_doc_ids = [c["doc_id"] for c in chunks]
        top_hit           = retrieved_doc_ids[0] if retrieved_doc_ids else ""
        retrieval_correct = expected in retrieved_doc_ids

        # Step 2 — Build prompt
        prompt = build_prompt(question, chunks)

        # Step 3 — Generate
        try:
            answer, duration_ms = generate(client, prompt)
            status = "success"
        except Exception as e:
            answer      = f"ERROR: {e}"
            duration_ms = 0
            status      = "error"

        # Show retrieval result
        retrieval_icon = "✅" if retrieval_correct else "❌"
        print(f"         Retrieved : {[c['doc_id'] for c in chunks]}")
        print(f"         Expected  : {expected} {retrieval_icon}")
        print(f"         Top hit   : {chunks[0]['topic']} — {chunks[0]['section']}")
        print(f"         {duration_ms}ms\n")

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
    total_correct = sum(1 for r in results if r["retrieval_correct"])
    retrieval_acc = round(total_correct / total * 100, 1)

    output = {
        "metadata": {
            "timestamp":       timestamp,
            "model":           MODEL,
            "top_k":           TOP_K,
            "total":           total,
            "retrieval_correct": total_correct,
            "retrieval_accuracy": f"{retrieval_acc}%",
        },
        "results": results,
    }

    output_file = RESULTS_DIR / f"rag_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'='*55}")
    print(f"  Retrieval Accuracy : {retrieval_acc}% ({total_correct}/{total})")
    print(f"{'='*55}")
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
