"""
hyde_pipeline.py — HyDE (Hypothetical Document Embeddings) RAG Pipeline

Standard RAG:
  question → vector → retrieve → generate

HyDE RAG:
  question → Claude generates hypothetical answer → vector → retrieve → generate

The hypothetical answer reads like document content — so its vector is closer
to actual document chunks than a question vector would be.

This fixes semantic overlap failures where the question words match the wrong doc.

Usage:
    python3 scripts/hyde_pipeline.py

Output:
    results/hyde_results_<timestamp>.json
    Prints side-by-side comparison with standard pipeline results
"""

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
TOP_K            = 3

# ── HyDE prompt — asks Claude to write a hypothetical answer ─────────────────
HYDE_PROMPT = """\
You are a culinary expert writing content for a cooking techniques reference guide.

Write a short passage (3-5 sentences) that directly answers the following question.
Write it as if it were a section of a cookbook — factual, specific, instructional.
Do not say "I" or reference the question. Just write the answer as document content.

Question: {question}

Passage:"""

# ── Grounded generation prompt (same as standard pipeline) ───────────────────
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


def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def generate_hypothesis(client: anthropic.Anthropic, question: str) -> str:
    """Ask Claude to write a hypothetical answer — used as the search query."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
    )
    return message.content[0].text.strip()


def retrieve(collection, query_text: str, top_k: int = TOP_K) -> list[dict]:
    """Search ChromaDB using query_text (question or hypothesis)."""
    results = collection.query(query_texts=[query_text], n_results=top_k)
    chunks = []
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
        })
    return chunks


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
    questions  = load_questions()
    total      = len(questions)

    # Load standard pipeline results for comparison
    std_file    = find_latest_standard_results()
    std_data    = json.loads(std_file.read_text(encoding="utf-8"))
    std_results = {r["id"]: r for r in std_data["results"]}

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n🔮 VP RAG Eval — HyDE Pipeline")
    print(f"   Model          : {MODEL}")
    print(f"   Embedding      : {EMBEDDING_MODEL}")
    print(f"   Questions      : {total}")
    print(f"   Compare against: {std_file.name}\n")
    print(f"   {'Q':<4} {'Expected':<25} {'Standard':^10} {'HyDE':^10}")
    print(f"   {'-'*55}")

    results = []

    for row in questions:
        qid        = row["id"]
        question   = row["question"]
        expected   = row["expected_doc_id"]
        difficulty = row["difficulty"]

        # Step 1 — Generate hypothesis
        hypothesis = generate_hypothesis(client, question)

        # Step 2 — Retrieve using hypothesis instead of question
        chunks = retrieve(collection, hypothesis)
        retrieved_doc_ids = [c["doc_id"] for c in chunks]
        retrieval_correct = expected in retrieved_doc_ids
        top_hit           = retrieved_doc_ids[0] if retrieved_doc_ids else ""

        # Step 3 — Build grounded prompt with retrieved chunks + ORIGINAL question
        prompt = build_prompt(question, chunks)

        # Step 4 — Generate answer
        try:
            answer, duration_ms = generate_answer(client, prompt)
            status = "success"
        except Exception as e:
            answer, duration_ms, status = f"ERROR: {e}", 0, "error"

        # Compare with standard
        std = std_results.get(qid, {})
        std_correct  = std.get("retrieval_correct", False)
        std_icon  = "✅" if std_correct  else "❌"
        hyde_icon = "✅" if retrieval_correct else "❌"

        # Flag changes
        if not std_correct and retrieval_correct:
            change = "⬆️  FIXED"
        elif std_correct and not retrieval_correct:
            change = "⬇️  BROKE"
        else:
            change = ""

        print(f"   Q{qid:<3} {expected:<25} {std_icon:^10} {hyde_icon:^10} {change}")

        results.append({
            "id":                qid,
            "question":          question,
            "difficulty":        difficulty,
            "expected_doc_id":   expected,
            "hypothesis":        hypothesis,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_correct": retrieval_correct,
            "top_hit_doc_id":    top_hit,
            "top_hit_topic":     chunks[0]["topic"]   if chunks else "",
            "top_hit_section":   chunks[0]["section"] if chunks else "",
            "chunks":            chunks,
            "answer":            answer,
            "status":            status,
            "duration_ms":       duration_ms,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    std_correct_count  = sum(1 for r in std_data["results"] if r["retrieval_correct"])
    hyde_correct_count = sum(1 for r in results if r["retrieval_correct"])
    std_pct  = round(std_correct_count  / total * 100, 1)
    hyde_pct = round(hyde_correct_count / total * 100, 1)
    diff     = round(hyde_pct - std_pct, 1)
    diff_str = f"+{diff}%" if diff > 0 else f"{diff}%"

    print(f"\n{'='*55}")
    print(f"  Standard RAG : {std_pct}%  ({std_correct_count}/{total})")
    print(f"  HyDE RAG     : {hyde_pct}%  ({hyde_correct_count}/{total})  {diff_str}")
    print(f"{'='*55}")

    # Save
    output = {
        "metadata": {
            "timestamp":          timestamp,
            "model":              MODEL,
            "embedding_model":    EMBEDDING_MODEL,
            "top_k":              TOP_K,
            "total":              total,
            "retrieval_correct":  hyde_correct_count,
            "retrieval_accuracy": f"{hyde_pct}%",
            "standard_accuracy":  f"{std_pct}%",
            "delta":              diff_str,
            "compare_against":    std_file.name,
        },
        "results": results,
    }
    output_file = RESULTS_DIR / f"hyde_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
