"""
langchain_pipeline.py — LangChain RAG Pipeline (Step 7)

Rebuilds the standard BGE-large RAG pipeline using LangChain abstractions:

  Manual pipeline (what we built)   →   LangChain equivalent
  ─────────────────────────────────────────────────────────────
  chromadb.PersistentClient          →   Chroma vectorstore
  SentenceTransformerEmbeddingFn     →   HuggingFaceEmbeddings
  GROUNDED_PROMPT.format(...)        →   PromptTemplate
  anthropic.Anthropic(...)           →   ChatAnthropic
  manual retrieve → format → gen     →   LCEL chain ( | pipe operator )

Why LangChain?
  - Industry-standard framework: chains, retrievers, memory, agents
  - LCEL (LangChain Expression Language) — composable pipelines with |
  - swap retrievers / models without rewriting orchestration logic
  - foundation for LlamaIndex, LangGraph, production patterns

LCEL Pipeline:
  retriever | format_docs → prompt → llm → output_parser

Comparison: runs against same questions.csv, compares to latest rag_results_*.json

Usage:
    python3 scripts/langchain_pipeline.py

Output:
    results/langchain_results_<timestamp>.json
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

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

# ── Grounded prompt (same constraint as all other pipelines) ──────────────────
GROUNDED_TEMPLATE = """\
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


def format_docs(docs) -> str:
    """Convert LangChain Document objects into the [Source N: ...] format."""
    parts = []
    for i, doc in enumerate(docs, 1):
        topic   = doc.metadata.get("topic", "")
        section = doc.metadata.get("section", "")
        parts.append(f"[Source {i}: {topic} — {section}]\n{doc.page_content}")
    return "\n\n".join(parts)


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

    # ── Step 1: Load embedding model (same BGE-large as other pipelines) ──────
    print(f"\n🦜 Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"   ✅ Embeddings ready\n")

    # ── Step 2: Connect to existing ChromaDB collection ───────────────────────
    # LangChain's Chroma wraps the same chroma_db/ we built in build_index.py
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # ── Step 3: Create retriever (LangChain abstraction over vector search) ───
    # .as_retriever() returns a Runnable — plug into any LCEL chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # ── Step 4: LLM ───────────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        api_key=api_key,
    )

    # ── Step 5: Prompt template ───────────────────────────────────────────────
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=GROUNDED_TEMPLATE,
    )

    # ── Step 6: LCEL chain ( | pipe operator ) ────────────────────────────────
    #
    # How LCEL works:
    #   retriever          → takes "question", returns list[Document]
    #   format_docs        → converts Document list to string context
    #   prompt             → fills {context} and {question} into template
    #   llm                → calls Claude, returns AIMessage
    #   StrOutputParser()  → extracts .content string from AIMessage
    #
    # RunnablePassthrough() passes "question" through unchanged so both
    # the retriever AND the prompt template receive it.
    #
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    questions = load_questions()
    total     = len(questions)

    std_file    = find_latest_standard_results()
    std_data    = json.loads(std_file.read_text(encoding="utf-8"))
    std_results = {r["id"]: r for r in std_data["results"]}

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"🦜 VP RAG Eval — LangChain Pipeline (Step 7)")
    print(f"   Embeddings : HuggingFaceEmbeddings ({EMBEDDING_MODEL})")
    print(f"   Vectorstore: Chroma (existing chroma_db/)")
    print(f"   Retriever  : .as_retriever(k={TOP_K})")
    print(f"   LLM        : ChatAnthropic ({MODEL})")
    print(f"   Chain      : LCEL  retriever | format_docs | prompt | llm | parser")
    print(f"   Compare    : {std_file.name}\n")
    print(f"   {'Q':<4} {'Expected':<25} {'Standard':^10} {'LangChain':^10}")
    print(f"   {'-'*55}")

    results = []

    for row in questions:
        qid        = row["id"]
        question   = row["question"]
        expected   = row["expected_doc_id"]
        difficulty = row["difficulty"]

        # Retrieve docs separately so we can log retrieved_doc_ids
        start = time.time()
        docs  = retriever.invoke(question)
        retrieved_doc_ids = [d.metadata.get("doc_id", "") for d in docs]
        retrieval_correct = expected in retrieved_doc_ids
        top_hit           = retrieved_doc_ids[0] if retrieved_doc_ids else ""

        # Generate answer via chain
        try:
            answer      = rag_chain.invoke(question)
            duration_ms = round((time.time() - start) * 1000)
            status      = "success"
        except Exception as e:
            answer, duration_ms, status = f"ERROR: {e}", 0, "error"

        # Compare with standard
        std         = std_results.get(qid, {})
        std_correct = std.get("retrieval_correct", False)
        std_icon    = "✅" if std_correct       else "❌"
        lc_icon     = "✅" if retrieval_correct else "❌"
        change      = "⬆️  FIXED" if not std_correct and retrieval_correct else \
                      "⬇️  BROKE" if std_correct and not retrieval_correct else ""

        print(f"   Q{qid:<3} {expected:<25} {std_icon:^10} {lc_icon:^10} {change}")

        results.append({
            "id":                qid,
            "question":          question,
            "difficulty":        difficulty,
            "expected_doc_id":   expected,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_correct": retrieval_correct,
            "top_hit_doc_id":    top_hit,
            "top_hit_topic":     docs[0].metadata.get("topic", "")    if docs else "",
            "top_hit_section":   docs[0].metadata.get("section", "")  if docs else "",
            "answer":            answer,
            "status":            status,
            "duration_ms":       duration_ms,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    std_correct_count = sum(1 for r in std_data["results"] if r["retrieval_correct"])
    lc_correct_count  = sum(1 for r in results if r["retrieval_correct"])
    std_pct = round(std_correct_count / total * 100, 1)
    lc_pct  = round(lc_correct_count  / total * 100, 1)
    diff    = round(lc_pct - std_pct, 1)
    diff_str = f"+{diff}%" if diff > 0 else f"{diff}%"

    print(f"\n{'='*55}")
    print(f"  Standard RAG  : {std_pct}%  ({std_correct_count}/{total})")
    print(f"  LangChain RAG : {lc_pct}%  ({lc_correct_count}/{total})  {diff_str}")
    print(f"{'='*55}")
    print(f"\n  Key: Same BGE-large model + Chroma DB → expect same retrieval accuracy")
    print(f"  What changed: orchestration layer only (manual → LCEL chain)")

    output = {
        "metadata": {
            "timestamp":          timestamp,
            "framework":          "LangChain",
            "model":              MODEL,
            "embedding_model":    EMBEDDING_MODEL,
            "vectorstore":        "Chroma (existing chroma_db/)",
            "chain_type":         "LCEL (retriever | format_docs | prompt | llm | parser)",
            "top_k":              TOP_K,
            "total":              total,
            "retrieval_correct":  lc_correct_count,
            "retrieval_accuracy": f"{lc_pct}%",
            "standard_accuracy":  f"{std_pct}%",
            "delta":              diff_str,
            "compare_against":    std_file.name,
        },
        "results": results,
    }
    output_file = RESULTS_DIR / f"langchain_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
