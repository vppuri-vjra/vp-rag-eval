"""
agentic_pipeline.py — Agentic RAG Pipeline (Step 6e)

Fixed pipelines (everything we built so far):
  Question → always retrieve once → always same query → generate → done

Agentic RAG:
  Question → Claude thinks → call retrieve tool? with what query?
                           → reads results → enough? → answer OR retrieve again

What the agent controls:
  1. WHETHER to retrieve   — might answer simple questions directly
  2. WHAT query to use     — can rephrase, expand, or decompose the question
  3. HOW MANY TIMES        — can retrieve multiple times if first attempt insufficient

Implementation:
  Uses Anthropic tool use API — Claude receives a `retrieve` tool definition.
  When Claude calls the tool, we execute ChromaDB search and return results.
  Loop continues until Claude emits a final text answer (no more tool calls).

Why this matters:
  Fixed pipeline: "pan sauce" → retrieves top-3 → always gets reduction doc wrong
  Agent:          "pan sauce" → thinks → calls retrieve("deglazing fond technique")
                             → gets right doc → answers correctly

Usage:
    python3 scripts/agentic_pipeline.py

Output:
    results/agentic_results_<timestamp>.json
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
MAX_TOKENS       = 1024
COLLECTION_NAME  = "cooking_techniques"
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
MAX_ITERATIONS   = 5   # safety limit — agent can retrieve at most 5 times per question

# ── Tool definition — what Claude sees ───────────────────────────────────────
RETRIEVE_TOOL = {
    "name": "retrieve",
    "description": (
        "Search the cooking techniques knowledge base for relevant information. "
        "Use this tool to find content about specific cooking methods, techniques, or concepts. "
        "You can call this tool multiple times with different queries if needed. "
        "Always retrieve before answering — do not answer from memory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query. Can be the original question, a rephrased version, "
                    "or a more specific sub-query to find relevant content. "
                    "Write the query as if searching for document content, not as a question."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to retrieve (1-5). Default 3.",
                "default": 3,
            },
        },
        "required": ["query"],
    },
}

# ── System prompt — defines agent behaviour ───────────────────────────────────
SYSTEM_PROMPT = """\
You are a cooking techniques expert assistant with access to a knowledge base.

Rules:
1. ALWAYS use the retrieve tool before answering — never answer from your own knowledge
2. If the first retrieval doesn't return relevant content, try a different query
3. You may retrieve multiple times with different queries if needed
4. Once you have enough context, give a clear, concise answer
5. If the knowledge base doesn't contain the answer after 2-3 retrieval attempts, say so

When forming queries: write them as document search terms, not as questions.
Example: instead of "How do I make a pan sauce?" use "pan sauce deglazing fond technique"
"""

# ── Grounded answer instruction (injected after retrieval) ────────────────────
GROUNDING_REMINDER = (
    "\n\nIMPORTANT: Answer using ONLY the information from the retrieved context above. "
    "Do not use outside knowledge."
)


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


def execute_retrieve(collection, query: str, top_k: int = 3) -> dict:
    """Execute ChromaDB search. Returns dict Claude will see as tool result."""
    top_k = max(1, min(top_k, 5))  # clamp 1-5
    results = collection.query(query_texts=[query], n_results=top_k)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "doc_id":   meta["doc_id"],
            "topic":    meta["topic"],
            "section":  meta["section"],
            "content":  doc,
            "distance": round(dist, 4),
        })

    # Format as readable text for Claude
    formatted = []
    for i, c in enumerate(chunks, 1):
        formatted.append(
            f"[Result {i}: {c['topic']} — {c['section']} | doc_id: {c['doc_id']} | distance: {c['distance']}]\n{c['content']}"
        )

    return {
        "query":       query,
        "top_k":       top_k,
        "results":     chunks,
        "text":        "\n\n".join(formatted),
    }


def find_latest_standard_results() -> Path:
    files = sorted(RESULTS_DIR.glob("rag_results_*.json"))
    if not files:
        raise FileNotFoundError("No rag_results_*.json found")
    return files[-1]


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_agent(client: anthropic.Anthropic, collection, question: str) -> dict:
    """
    Run the agentic RAG loop for one question.

    Loop:
      1. Send question + retrieve tool to Claude
      2. If Claude calls retrieve → execute → send results back
      3. If Claude emits text → that's the final answer
      4. Repeat up to MAX_ITERATIONS

    Returns dict with: answer, tool_calls, retrieved_doc_ids, iterations
    """
    messages = [{"role": "user", "content": question + GROUNDING_REMINDER}]

    tool_calls      = []   # log every retrieve call the agent makes
    all_retrieved   = []   # all chunks across all retrieval calls
    iterations      = 0
    final_answer    = ""

    start = time.time()

    while iterations < MAX_ITERATIONS:
        iterations += 1

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[RETRIEVE_TOOL],
            messages=messages,
        )

        # Add assistant response to message history
        messages.append({"role": "assistant", "content": response.content})

        # ── Check stop reason ─────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            # Claude finished — extract text answer
            for block in response.content:
                if hasattr(block, "text"):
                    final_answer = block.text.strip()
            break

        if response.stop_reason != "tool_use":
            # Unexpected stop — treat as done
            break

        # ── Process tool calls ────────────────────────────────────────────────
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            query = tool_input.get("query", question)
            top_k = tool_input.get("top_k", 3)

            # Execute retrieval
            result = execute_retrieve(collection, query, top_k)
            all_retrieved.extend(result["results"])

            tool_calls.append({
                "iteration": iterations,
                "query":     query,
                "top_k":     top_k,
                "doc_ids":   [c["doc_id"] for c in result["results"]],
            })

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result["text"],
            })

        # Send tool results back to Claude
        messages.append({"role": "user", "content": tool_results})

    duration_ms = round((time.time() - start) * 1000)

    # Collect all retrieved doc_ids across all calls
    retrieved_doc_ids = list(dict.fromkeys(c["doc_id"] for c in all_retrieved))

    return {
        "answer":            final_answer,
        "tool_calls":        tool_calls,
        "retrieved_doc_ids": retrieved_doc_ids,
        "all_chunks":        all_retrieved,
        "iterations":        iterations,
        "duration_ms":       duration_ms,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

    client     = anthropic.Anthropic(api_key=api_key)
    collection = load_collection()
    questions  = load_questions()
    total      = len(questions)

    std_file    = find_latest_standard_results()
    std_data    = json.loads(std_file.read_text(encoding="utf-8"))
    std_results = {r["id"]: r for r in std_data["results"]}

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n🤖 VP RAG Eval — Agentic RAG Pipeline (Step 6e)")
    print(f"   Model     : {MODEL}")
    print(f"   Embeddings: {EMBEDDING_MODEL}")
    print(f"   Agent     : tool_use loop, up to {MAX_ITERATIONS} retrieve calls per question")
    print(f"   Compare   : {std_file.name}\n")
    print(f"   {'Q':<4} {'Expected':<25} {'Std':^6} {'Agent':^6} {'Calls':^6} {'Queries used'}")
    print(f"   {'-'*75}")

    results = []

    for row in questions:
        qid        = row["id"]
        question   = row["question"]
        expected   = row["expected_doc_id"]
        difficulty = row["difficulty"]

        agent_result = run_agent(client, collection, question)

        retrieved_doc_ids = agent_result["retrieved_doc_ids"]
        retrieval_correct = expected in retrieved_doc_ids
        top_hit           = retrieved_doc_ids[0] if retrieved_doc_ids else ""
        num_calls         = len(agent_result["tool_calls"])
        queries           = " → ".join(tc["query"][:30] for tc in agent_result["tool_calls"])

        std         = std_results.get(qid, {})
        std_correct = std.get("retrieval_correct", False)
        std_icon    = "✅" if std_correct       else "❌"
        ag_icon     = "✅" if retrieval_correct else "❌"
        change      = " ⬆️FIXED" if not std_correct and retrieval_correct else \
                      " ⬇️BROKE" if std_correct and not retrieval_correct else ""

        print(f"   Q{qid:<3} {expected:<25} {std_icon:^6} {ag_icon:^6} {num_calls:^6} {queries}{change}")

        results.append({
            "id":                qid,
            "question":          question,
            "difficulty":        difficulty,
            "expected_doc_id":   expected,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_correct": retrieval_correct,
            "top_hit_doc_id":    top_hit,
            "tool_calls":        agent_result["tool_calls"],
            "num_retrieve_calls": num_calls,
            "answer":            agent_result["answer"],
            "iterations":        agent_result["iterations"],
            "status":            "success",
            "duration_ms":       agent_result["duration_ms"],
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    std_correct_count = sum(1 for r in std_data["results"] if r["retrieval_correct"])
    ag_correct_count  = sum(1 for r in results if r["retrieval_correct"])
    std_pct = round(std_correct_count / total * 100, 1)
    ag_pct  = round(ag_correct_count  / total * 100, 1)
    diff    = round(ag_pct - std_pct, 1)
    diff_str = f"+{diff}%" if diff > 0 else f"{diff}%"
    avg_calls = round(sum(r["num_retrieve_calls"] for r in results) / total, 1)

    print(f"\n{'='*60}")
    print(f"  Standard RAG  : {std_pct}%  ({std_correct_count}/{total})")
    print(f"  Agentic RAG   : {ag_pct}%  ({ag_correct_count}/{total})  {diff_str}")
    print(f"  Avg retrieve calls per question: {avg_calls}")
    print(f"{'='*60}")

    output = {
        "metadata": {
            "timestamp":           timestamp,
            "model":               MODEL,
            "embedding_model":     EMBEDDING_MODEL,
            "pattern":             "Agentic RAG — tool_use loop",
            "max_iterations":      MAX_ITERATIONS,
            "avg_retrieve_calls":  avg_calls,
            "total":               total,
            "retrieval_correct":   ag_correct_count,
            "retrieval_accuracy":  f"{ag_pct}%",
            "standard_accuracy":   f"{std_pct}%",
            "delta":               diff_str,
            "compare_against":     std_file.name,
        },
        "results": results,
    }
    output_file = RESULTS_DIR / f"agentic_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
