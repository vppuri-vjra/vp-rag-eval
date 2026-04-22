"""
retrieval_eval.py — Retrieval Quality Evaluation

Reads the RAG pipeline results and produces a structured CSV report:
  - Per-question: was the expected doc retrieved? rank? distance?
  - Summary: accuracy by difficulty (easy / medium / hard)

Usage:
    python3 scripts/retrieval_eval.py
    python3 scripts/retrieval_eval.py --results results/rag_results_20260422_143630.json

Output:
    results/retrieval_eval.csv
"""

import argparse
import csv
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_latest_results() -> Path:
    files = sorted(RESULTS_DIR.glob("rag_results_*.json"))
    if not files:
        raise FileNotFoundError("No rag_results_*.json found in results/")
    return files[-1]


def rank_of_expected(expected_doc_id: str, chunks: list[dict]) -> int | None:
    """Return 1-based rank of expected doc in retrieved chunks, or None."""
    for i, chunk in enumerate(chunks, 1):
        if chunk["doc_id"] == expected_doc_id:
            return i
    return None


def distance_of_expected(expected_doc_id: str, chunks: list[dict]) -> float | None:
    """Return cosine distance of first chunk matching expected doc, or None."""
    for chunk in chunks:
        if chunk["doc_id"] == expected_doc_id:
            return chunk["distance"]
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                        help="Path to rag_results_*.json (default: latest)")
    args = parser.parse_args()

    results_file = Path(args.results) if args.results else find_latest_results()
    data = json.loads(results_file.read_text(encoding="utf-8"))

    meta    = data["metadata"]
    results = data["results"]
    total   = len(results)

    print(f"\n📊 VP RAG Eval — Retrieval Evaluation")
    print(f"   Source  : {results_file.name}")
    print(f"   Model   : {meta['model']}")
    print(f"   Top-K   : {meta['top_k']}")
    print(f"   Total   : {total} questions\n")

    # ── Per-question rows ─────────────────────────────────────────────────────
    rows = []
    for r in results:
        chunks   = r["chunks"]
        exp_doc  = r["expected_doc_id"]
        rank     = rank_of_expected(exp_doc, chunks)
        dist     = distance_of_expected(exp_doc, chunks)
        got_doc  = chunks[0]["doc_id"] if chunks else ""
        correct  = rank is not None

        rows.append({
            "id":               r["id"],
            "difficulty":       r["difficulty"],
            "question":         r["question"],
            "expected_doc_id":  exp_doc,
            "got_doc_id":       got_doc,
            "retrieval_correct": "yes" if correct else "no",
            "rank_of_expected": rank if rank else "not found",
            "distance_of_expected": dist if dist is not None else "—",
            "top_hit_distance": chunks[0]["distance"] if chunks else "—",
            "top_hit_topic":    chunks[0]["topic"]    if chunks else "",
        })

    # ── Accuracy by difficulty ────────────────────────────────────────────────
    difficulties = ["easy", "medium", "hard"]
    stats = {}
    for d in difficulties:
        subset  = [r for r in rows if r["difficulty"] == d]
        correct = sum(1 for r in subset if r["retrieval_correct"] == "yes")
        stats[d] = {"total": len(subset), "correct": correct,
                    "pct": round(correct / len(subset) * 100, 1) if subset else 0}

    overall_correct = sum(1 for r in rows if r["retrieval_correct"] == "yes")
    overall_pct     = round(overall_correct / total * 100, 1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"  {'Difficulty':<10} {'Questions':>10} {'Correct':>10} {'Accuracy':>10}")
    print(f"  {'-'*44}")
    for d in difficulties:
        s = stats[d]
        print(f"  {d.capitalize():<10} {s['total']:>10} {s['correct']:>10} {s['pct']:>9}%")
    print(f"  {'-'*44}")
    print(f"  {'OVERALL':<10} {total:>10} {overall_correct:>10} {overall_pct:>9}%\n")

    # ── Failures ──────────────────────────────────────────────────────────────
    failures = [r for r in rows if r["retrieval_correct"] == "no"]
    if failures:
        print(f"  ❌ Retrieval failures ({len(failures)}):")
        for f in failures:
            print(f"     Q{f['id']} ({f['difficulty']}) — expected: {f['expected_doc_id']}")
            print(f"            got: {f['got_doc_id']}  |  top-hit-dist: {f['top_hit_distance']}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    output_csv = RESULTS_DIR / "retrieval_eval.csv"
    fieldnames = [
        "id", "difficulty", "question", "expected_doc_id", "got_doc_id",
        "retrieval_correct", "rank_of_expected", "distance_of_expected",
        "top_hit_distance", "top_hit_topic",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Append summary rows
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        f.write("\n")
        f.write("# SUMMARY BY DIFFICULTY\n")
        f.write("difficulty,total,correct,accuracy\n")
        for d in difficulties:
            s = stats[d]
            f.write(f"{d},{s['total']},{s['correct']},{s['pct']}%\n")
        f.write(f"overall,{total},{overall_correct},{overall_pct}%\n")

    print(f"\n{'='*55}")
    print(f"  Retrieval Accuracy : {overall_pct}% ({overall_correct}/{total})")
    print(f"{'='*55}")
    print(f"\n✅ Saved: {output_csv}")


if __name__ == "__main__":
    main()
