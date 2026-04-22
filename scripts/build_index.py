"""
build_index.py — Loads document chunks into ChromaDB vector database

Converts each chunk to a vector embedding using sentence-transformers
and stores it in a local ChromaDB collection for semantic search.

Usage:
    python3 scripts/build_index.py

Output:
    chroma_db/  — local vector database (excluded from GitHub via .gitignore)
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ── Config ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CHUNKS_FILE = ROOT / "data" / "chunks.json"
CHROMA_DIR  = ROOT / "chroma_db"

COLLECTION_NAME  = "cooking_techniques"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"  # fast, lightweight, 384 dimensions


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load chunks
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"\n📦 VP RAG Eval — Build ChromaDB Index")
    print(f"   Embedding model : {EMBEDDING_MODEL}")
    print(f"   Chunks to load  : {len(chunks)}")
    print(f"   Output dir      : {CHROMA_DIR}\n")

    # Set up ChromaDB with local persistence
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if it exists (clean rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   ♻️  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    # Create embedding function using sentence-transformers
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Create collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for search
    )

    print(f"   ✅ Collection '{COLLECTION_NAME}' created\n")
    print(f"   Loading chunks into ChromaDB...")

    # Add chunks in batches
    BATCH_SIZE = 20
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        collection.add(
            ids        = [c["chunk_id"] for c in batch],
            documents  = [c["content"]  for c in batch],
            metadatas  = [
                {
                    "doc_id":  c["doc_id"],
                    "topic":   c["topic"],
                    "section": c["section"],
                }
                for c in batch
            ],
        )
        print(f"   Loaded chunks {i+1}–{min(i+BATCH_SIZE, len(chunks))} of {len(chunks)}")

    # Verify
    count = collection.count()
    print(f"\n{'='*55}")
    print(f"  Index built successfully")
    print(f"  Collection  : {COLLECTION_NAME}")
    print(f"  Chunks      : {count}")
    print(f"  Dimensions  : 384 (all-MiniLM-L6-v2)")
    print(f"  Similarity  : cosine")
    print(f"{'='*55}")

    # Quick sanity test — search for a sample query
    print(f"\n🔍 Sanity test — searching: 'how long to blanch green beans'")
    results = collection.query(
        query_texts=["how long to blanch green beans"],
        n_results=3,
    )
    for j, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        print(f"\n  Result {j}: [{meta['topic']}] {meta['section']}")
        print(f"  Preview : {doc[:100]}...")

    print(f"\n✅ ChromaDB index ready at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
