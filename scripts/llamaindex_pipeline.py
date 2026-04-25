"""
llamaindex_pipeline.py — LlamaIndex RAG Pipeline (Step 8)

Rebuilds the BGE-large RAG pipeline using LlamaIndex abstractions.

Manual pipeline (what we built)        LlamaIndex equivalent
──────────────────────────────────────────────────────────────
Read txt files ourselves           →   SimpleDirectoryReader("data/docs/")
Chunk docs with custom script      →   SentenceSplitter (built-in, smarter)
SentenceTransformerEmbeddingFn     →   HuggingFaceEmbedding(model_name=...)
chromadb.PersistentClient(...)     →   ChromaVectorStore + StorageContext
retrieve → format → generate loop  →   index.as_query_engine().query(question)
Custom grounded prompt             →   text_qa_template with same constraint

Key LlamaIndex concepts:
  Document      — raw text + metadata loaded from file
  Node          — a chunk (LlamaIndex calls chunks "nodes")
  VectorIndex   — stores nodes as vectors, handles search
  QueryEngine   — retriever + response synthesizer in one object
  StorageContext — wires together the vector store, docstore, index

Usage:
    python3 scripts/llamaindex_pipeline.py

Output:
    results/llamaindex_results_<timestamp>.json
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import chromadb
from dotenv import load_dotenv

# ── LlamaIndex imports ────────────────────────────────────────────────────────
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── Config ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
QUESTIONS_CSV  = ROOT / "data" / "questions.csv"
DOCS_DIR       = ROOT / "data" / "docs"
CHROMA_DIR     = ROOT / "chroma_db"
RESULTS_DIR    = ROOT / "results"

load_dotenv(ROOT / ".env")

MODEL            = "claude-opus-4-5"
MAX_TOKENS       = 512
COLLECTION_NAME  = "cooking_techniques_llamaindex"   # separate collection — fresh index
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
TOP_K            = 3

# ── Grounded prompt — same constraint as all other pipelines ──────────────────
GROUNDED_TEMPLATE = PromptTemplate(
    "Answer the question using ONLY the information provided below.\n"
    "Do not use outside knowledge. If the answer is not in the provided "
    "context, say 'I cannot answer this from the provided information.'\n\n"
    "--- CONTEXT ---\n"
    "{context_str}\n"
    "--- END CONTEXT ---\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_questions() -> list[dict]:
    with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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

    # ── Step 1: Configure global LlamaIndex settings ──────────────────────────
    # Settings replaces ServiceContext (old LlamaIndex API)
    # All components read from here — no need to pass them everywhere
    print(f"\n🦙 Configuring LlamaIndex settings...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm         = Anthropic(model=MODEL, max_tokens=MAX_TOKENS, api_key=api_key)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    print(f"   Embedding : {EMBEDDING_MODEL}")
    print(f"   LLM       : {MODEL}")
    print(f"   Chunking  : SentenceSplitter (chunk_size=512, overlap=50)\n")

    # ── Step 2: Load documents ────────────────────────────────────────────────
    # SimpleDirectoryReader reads all .txt files, attaches filename as metadata
    print(f"🦙 Loading documents from {DOCS_DIR.name}/...")
    reader = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        required_exts=[".txt"],
        filename_as_id=True,
    )
    documents = reader.load_data()
    print(f"   ✅ Loaded {len(documents)} documents\n")

    # ── Step 3: Connect ChromaDB and build VectorStoreIndex ───────────────────
    # ChromaVectorStore wraps our existing chroma_db/ directory
    # StorageContext wires the vector store into LlamaIndex's pipeline
    print(f"🦙 Building VectorStoreIndex in ChromaDB...")
    chroma_client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)

    # VectorStoreIndex.from_documents():
    #   1. Splits documents into nodes (chunks) using Settings.node_parser
    #   2. Embeds each node using Settings.embed_model
    #   3. Stores in ChromaDB
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"   ✅ Index built\n")

    # ── Step 4: Create query engine ───────────────────────────────────────────
    # as_query_engine() = retriever + response synthesizer
    # similarity_top_k — how many nodes to retrieve
    # text_qa_template — our grounded prompt
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        text_qa_template=GROUNDED_TEMPLATE,
    )

    # ── Load questions and standard results ───────────────────────────────────
    questions   = load_questions()
    total       = len(questions)
    std_file    = find_latest_standard_results()
    std_data    = json.loads(std_file.read_text(encoding="utf-8"))
    std_results = {r["id"]: r for r in std_data["results"]}

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"🦙 VP RAG Eval — LlamaIndex Pipeline (Step 8)")
    print(f"   Docs ingested : {len(documents)}")
    print(f"   Retriever     : VectorStoreIndex.as_query_engine(top_k={TOP_K})")
    print(f"   Compare       : {std_file.name}\n")
    print(f"   {'Q':<4} {'Expected':<25} {'Std':^6} {'LlamaIdx':^8} {'Top hit'}")
    print(f"   {'-'*70}")

    results = []

    for row in questions:
        qid        = row["id"]
        question   = row["question"]
        expected   = row["expected_doc_id"]
        difficulty = row["difficulty"]

        start = time.time()

        # query() — retrieve top-K nodes → build prompt → call LLM → return response
        response    = query_engine.query(question)
        duration_ms = round((time.time() - start) * 1000)
        answer      = str(response).strip()

        # Extract retrieved doc_ids from source nodes
        source_nodes      = response.source_nodes
        retrieved_doc_ids = []
        for node in source_nodes:
            fname = node.metadata.get("file_name", "")
            # file_name is like "doc_01_blanching.txt" → strip extension
            doc_id = Path(fname).stem if fname else ""
            if doc_id and doc_id not in retrieved_doc_ids:
                retrieved_doc_ids.append(doc_id)

        retrieval_correct = expected in retrieved_doc_ids
        top_hit           = retrieved_doc_ids[0] if retrieved_doc_ids else ""

        std         = std_results.get(qid, {})
        std_correct = std.get("retrieval_correct", False)
        std_icon    = "✅" if std_correct       else "❌"
        li_icon     = "✅" if retrieval_correct else "❌"
        change      = " ⬆️FIXED" if not std_correct and retrieval_correct else \
                      " ⬇️BROKE" if std_correct and not retrieval_correct else ""

        print(f"   Q{qid:<3} {expected:<25} {std_icon:^6} {li_icon:^8} {top_hit}{change}")

        results.append({
            "id":                qid,
            "question":          question,
            "difficulty":        difficulty,
            "expected_doc_id":   expected,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_correct": retrieval_correct,
            "top_hit_doc_id":    top_hit,
            "answer":            answer,
            "status":            "success",
            "duration_ms":       duration_ms,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    std_correct_count = sum(1 for r in std_data["results"] if r["retrieval_correct"])
    li_correct_count  = sum(1 for r in results if r["retrieval_correct"])
    std_pct  = round(std_correct_count / total * 100, 1)
    li_pct   = round(li_correct_count  / total * 100, 1)
    diff     = round(li_pct - std_pct, 1)
    diff_str = f"+{diff}%" if diff > 0 else f"{diff}%"

    print(f"\n{'='*60}")
    print(f"  Standard RAG   : {std_pct}%  ({std_correct_count}/{total})")
    print(f"  LlamaIndex RAG : {li_pct}%  ({li_correct_count}/{total})  {diff_str}")
    print(f"{'='*60}")
    print(f"\n  Key differences vs manual pipeline:")
    print(f"  - SimpleDirectoryReader loaded docs in 1 line vs custom file reader")
    print(f"  - SentenceSplitter chunks by sentence boundaries, not section headers")
    print(f"  - query_engine.query() replaces retrieve → format → generate loop")

    output = {
        "metadata": {
            "timestamp":          timestamp,
            "framework":          "LlamaIndex",
            "model":              MODEL,
            "embedding_model":    EMBEDDING_MODEL,
            "chunking":           "SentenceSplitter(chunk_size=512, overlap=50)",
            "index_type":         "VectorStoreIndex",
            "vector_store":       "ChromaDB",
            "top_k":              TOP_K,
            "total":              total,
            "retrieval_correct":  li_correct_count,
            "retrieval_accuracy": f"{li_pct}%",
            "standard_accuracy":  f"{std_pct}%",
            "delta":              diff_str,
            "compare_against":    std_file.name,
        },
        "results": results,
    }
    output_file = RESULTS_DIR / f"llamaindex_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
