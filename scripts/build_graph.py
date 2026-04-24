"""
build_graph.py — Build Knowledge Graph for Graph RAG (Step 6f)

What this does:
  1. Reads all 20 cooking docs from data/docs/
  2. Calls Claude once per doc to extract key concepts (20 API calls)
  3. Builds a NetworkX graph:
       - Nodes  = documents (doc_id)
       - Edges  = two docs share >= 2 concepts → they are connected
  4. Saves data/knowledge_graph.json

Why concepts not vectors:
  Vector similarity connects docs that sound alike.
  Concept overlap connects docs that share underlying ideas:
    - maillard_reaction + deglazing share: fond, browning, flavor
    - reduction + deglazing share: pan sauce, fond, liquid
  These connections are what Graph RAG uses to expand retrieval context.

Usage:
    python3 scripts/build_graph.py

Output:
    data/knowledge_graph.json
    Prints the graph: nodes, edges, avg degree
"""

import json
import os
from pathlib import Path

import anthropic
import networkx as nx
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DOCS_DIR     = ROOT / "data" / "docs"
GRAPH_FILE   = ROOT / "data" / "knowledge_graph.json"

load_dotenv(ROOT / ".env")

MODEL              = "claude-opus-4-5"
MIN_SHARED_CONCEPTS = 2   # edge threshold — docs sharing >= N concepts get connected

CONCEPT_PROMPT = """\
Read this cooking technique document and extract the key concepts.

Return ONLY a JSON array of strings — the core cooking concepts, techniques,
ingredients, processes, and scientific terms mentioned.
Include 8-15 concepts. Be specific (e.g. "fond" not "cooking").

Document:
{text}

JSON array:"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_docs() -> list[dict]:
    """Load all .txt docs from data/docs/."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        doc_id = path.stem
        text   = path.read_text(encoding="utf-8")
        docs.append({"doc_id": doc_id, "text": text, "path": str(path)})
    return docs


def extract_concepts(client: anthropic.Anthropic, doc_id: str, text: str) -> list[str]:
    """Ask Claude to extract key concepts from a document."""
    prompt = CONCEPT_PROMPT.format(text=text[:2000])  # first 2000 chars — enough for concepts
    message = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # Parse JSON — Claude should return a clean array
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        concepts = json.loads(raw.strip())
        if isinstance(concepts, list):
            return [str(c).lower().strip() for c in concepts]
    except Exception:
        pass

    # Fallback: split by comma or newline
    concepts = [c.strip().strip('"').lower() for c in raw.replace("[", "").replace("]", "").split(",")]
    return [c for c in concepts if c]


def build_graph(doc_concepts: dict[str, list[str]], min_shared: int) -> nx.Graph:
    """
    Build NetworkX graph from doc concepts.
    Nodes = doc_ids
    Edges = docs sharing >= min_shared concepts
    Edge weight = number of shared concepts
    """
    G = nx.Graph()
    doc_ids = list(doc_concepts.keys())

    # Add all docs as nodes
    for doc_id, concepts in doc_concepts.items():
        G.add_node(doc_id, concepts=concepts)

    # Add edges for docs with sufficient concept overlap
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            id_a, id_b  = doc_ids[i], doc_ids[j]
            concepts_a  = set(doc_concepts[id_a])
            concepts_b  = set(doc_concepts[id_b])
            shared      = concepts_a & concepts_b

            if len(shared) >= min_shared:
                G.add_edge(id_a, id_b,
                           weight=len(shared),
                           shared_concepts=sorted(shared))

    return G


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=api_key)
    docs   = load_docs()

    print(f"\n🕸️  Building Knowledge Graph for Graph RAG")
    print(f"   Docs      : {len(docs)}")
    print(f"   Model     : {MODEL}")
    print(f"   Edge rule : docs sharing >= {MIN_SHARED_CONCEPTS} concepts\n")

    # ── Step 1: Extract concepts from each doc ────────────────────────────────
    doc_concepts: dict[str, list[str]] = {}

    for doc in docs:
        concepts = extract_concepts(client, doc["doc_id"], doc["text"])
        doc_concepts[doc["doc_id"]] = concepts
        print(f"   {doc['doc_id']:<35} → {', '.join(concepts[:5])}{'...' if len(concepts) > 5 else ''}")

    # ── Step 2: Build graph ───────────────────────────────────────────────────
    print(f"\n   Building graph (edge threshold: {MIN_SHARED_CONCEPTS} shared concepts)...")
    G = build_graph(doc_concepts, MIN_SHARED_CONCEPTS)

    # ── Step 3: Print graph stats ─────────────────────────────────────────────
    print(f"\n   Nodes : {G.number_of_nodes()}")
    print(f"   Edges : {G.number_of_edges()}")
    if G.number_of_nodes() > 0:
        degrees    = dict(G.degree())
        avg_degree = round(sum(degrees.values()) / len(degrees), 1)
        print(f"   Avg degree : {avg_degree} connections per doc")

    print(f"\n   Key connections found:")
    # Show edges sorted by weight (most shared concepts first)
    edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
    for a, b, data in edges[:10]:
        print(f"   {a:<35} ↔  {b:<35}  ({data['weight']} shared: {', '.join(data['shared_concepts'][:3])})")

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "total_docs":    len(docs),
            "total_nodes":   G.number_of_nodes(),
            "total_edges":   G.number_of_edges(),
            "edge_threshold": MIN_SHARED_CONCEPTS,
            "model":         MODEL,
        },
        "doc_concepts": doc_concepts,
        "edges": [
            {
                "source":          a,
                "target":          b,
                "weight":          data["weight"],
                "shared_concepts": data["shared_concepts"],
            }
            for a, b, data in G.edges(data=True)
        ],
    }

    GRAPH_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Knowledge graph saved to: {GRAPH_FILE}")


if __name__ == "__main__":
    main()
