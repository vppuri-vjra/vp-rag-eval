"""
visualize_graph.py — Visualize the Knowledge Graph

Reads data/knowledge_graph.json and draws:
  - Nodes  = documents (shortened labels)
  - Edges  = shared concepts (thicker = more shared)
  - Labels on edges = shared concept names

Output:
    results/knowledge_graph.png

Usage:
    python3 scripts/visualize_graph.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
GRAPH_FILE = ROOT / "data" / "knowledge_graph.json"
OUTPUT     = ROOT / "results" / "knowledge_graph.png"

# Shorten doc_id labels for readability
def short_label(doc_id: str) -> str:
    return doc_id.replace("doc_", "").replace("_", "\n")


def main():
    data = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))

    G = nx.Graph()

    for doc_id in data["doc_concepts"]:
        G.add_node(doc_id)

    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["weight"],
            shared=", ".join(edge["shared_concepts"][:2]),  # top 2 for label
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    # spring_layout spreads nodes by connection strength
    pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")

    # ── Draw edges — width proportional to shared concept count ──────────────
    edges     = G.edges(data=True)
    weights   = [d["weight"] for _, _, d in edges]
    max_w     = max(weights) if weights else 1

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        lw = 1.5 + (d["weight"] / max_w) * 5  # thickness 1.5–6.5
        alpha = 0.4 + (d["weight"] / max_w) * 0.5
        ax.plot([x0, x1], [y0, y1], color="#4a9eff", linewidth=lw, alpha=alpha, zorder=1)

        # Edge label — shared concepts
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, d["shared"][:30], fontsize=5.5, color="#aacfff",
                ha="center", va="center", alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.15", fc="#0f1117", ec="none", alpha=0.7))

    # ── Draw nodes ────────────────────────────────────────────────────────────
    # Highlight the Q9-relevant cluster
    q9_cluster = {"doc_09_reduction", "doc_11_deglazing", "doc_02_sauteing",
                  "doc_04_braising", "doc_06_maillard_reaction"}

    for node in G.nodes():
        x, y   = pos[node]
        degree = G.degree(node)
        color  = "#ff6b6b" if node in q9_cluster else "#4a9eff"
        size   = 0.06 + degree * 0.015

        circle = plt.Circle((x, y), size, color=color, zorder=3, alpha=0.9)
        ax.add_patch(circle)

        label = short_label(node)
        ax.text(x, y - size - 0.04, label, fontsize=7, color="white",
                ha="center", va="top", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="#1a1d27", ec="none", alpha=0.8))

    # ── Legend ────────────────────────────────────────────────────────────────
    red_patch  = mpatches.Patch(color="#ff6b6b", label="Q9 cluster (pan sauce fix path)")
    blue_patch = mpatches.Patch(color="#4a9eff", label="Other docs")
    ax.legend(handles=[red_patch, blue_patch], loc="lower left",
              facecolor="#1a1d27", edgecolor="#4a9eff", labelcolor="white", fontsize=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    nodes = data["metadata"]["total_nodes"]
    edges_count = data["metadata"]["total_edges"]
    ax.set_title(
        f"VP RAG Eval — Knowledge Graph\n"
        f"{nodes} docs (nodes)  ·  {edges_count} connections (edges)  ·  "
        f"edge = ≥2 shared concepts  ·  thickness = strength",
        color="white", fontsize=11, pad=15
    )

    ax.axis("off")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    OUTPUT.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"✅ Graph saved to: {OUTPUT}")
    print(f"   {nodes} nodes  ·  {edges_count} edges")
    print(f"   Red nodes = Q9 fix cluster (reduction ↔ deglazing ↔ maillard ↔ sauteing ↔ braising)")


if __name__ == "__main__":
    main()
