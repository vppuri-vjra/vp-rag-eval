"""
faithfulness_eval.py — Faithfulness Evaluation

Faithfulness = did Claude answer using ONLY the retrieved chunks,
               or did it introduce outside knowledge (hallucination)?

This script reads judge_results_*.json (produced by llm_judge.py),
extracts the GROUNDED score per question, and cross-references with
retrieval accuracy to identify patterns.

No additional Claude API calls needed — judge already scored GROUNDED.

Usage:
    python3 scripts/faithfulness_eval.py
    python3 scripts/faithfulness_eval.py --judge results/judge_results_20260422_161419.json

Output:
    results/faithfulness_eval.csv
"""

import argparse
import csv
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_latest_judge() -> Path:
    files = sorted(RESULTS_DIR.glob("judge_results_*.json"))
    if not files:
        raise FileNotFoundError("No judge_results_*.json found in results/")
    return files[-1]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=str, default=None,
                        help="Path to judge_results_*.json (default: latest)")
    args = parser.parse_args()

    judge_file = Path(args.judge) if args.judge else find_latest_judge()
    data       = json.loads(judge_file.read_text(encoding="utf-8"))
    results    = data["results"]
    total      = len(results)

    print(f"\n🔍 VP RAG Eval — Faithfulness Evaluation")
    print(f"   Source : {judge_file.name}")
    print(f"   Total  : {total} questions\n")

    # ── Build per-question rows ───────────────────────────────────────────────
    rows = []
    for r in results:
        retrieval_correct = r["retrieval_correct"]
        grounded          = r.get("grounded", "UNKNOWN")
        verdict           = r.get("verdict", "UNKNOWN")

        # Faithfulness category
        if grounded == "PASS" and retrieval_correct:
            category = "grounded + correct retrieval"
        elif grounded == "PASS" and not retrieval_correct:
            category = "grounded + wrong retrieval (safe refusal or partial)"
        elif grounded == "FAIL" and retrieval_correct:
            category = "hallucinated despite correct retrieval"
        else:
            category = "hallucinated + wrong retrieval"

        rows.append({
            "id":                r["id"],
            "difficulty":        r["difficulty"],
            "question":          r["question"],
            "retrieval_correct": "yes" if retrieval_correct else "no",
            "grounded":          grounded,
            "verdict":           verdict,
            "faithfulness":      "PASS" if grounded == "PASS" else "FAIL",
            "category":          category,
            "reason":            r.get("reason", ""),
        })

    # ── Summary stats ─────────────────────────────────────────────────────────
    faithful     = sum(1 for r in rows if r["faithfulness"] == "PASS")
    unfaithful   = sum(1 for r in rows if r["faithfulness"] == "FAIL")
    faith_pct    = round(faithful / total * 100, 1)

    # Cross-tab: retrieval x faithfulness
    both_pass    = sum(1 for r in rows if r["retrieval_correct"] == "yes" and r["faithfulness"] == "PASS")
    ret_fail_faith_pass = sum(1 for r in rows if r["retrieval_correct"] == "no"  and r["faithfulness"] == "PASS")
    ret_pass_faith_fail = sum(1 for r in rows if r["retrieval_correct"] == "yes" and r["faithfulness"] == "FAIL")
    both_fail    = sum(1 for r in rows if r["retrieval_correct"] == "no"  and r["faithfulness"] == "FAIL")

    print(f"  {'Metric':<40} {'Count':>6} {'%':>6}")
    print(f"  {'-'*54}")
    print(f"  {'Faithful (GROUNDED=PASS)':<40} {faithful:>6} {faith_pct:>5}%")
    print(f"  {'Unfaithful (GROUNDED=FAIL)':<40} {unfaithful:>6} {round(unfaithful/total*100,1):>5}%")
    print(f"\n  Cross-tab — Retrieval × Faithfulness:")
    print(f"  {'Retrieval PASS + Faithful PASS':<40} {both_pass:>6}")
    print(f"  {'Retrieval FAIL + Faithful PASS':<40} {ret_fail_faith_pass:>6}  ← grounding constraint worked")
    print(f"  {'Retrieval PASS + Faithful FAIL':<40} {ret_pass_faith_fail:>6}  ← hallucinated despite right chunks")
    print(f"  {'Retrieval FAIL + Faithful FAIL':<40} {both_fail:>6}  ← worst case")

    # By difficulty
    print(f"\n  By difficulty:")
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in rows if r["difficulty"] == diff]
        f_pass = sum(1 for r in subset if r["faithfulness"] == "PASS")
        f_pct  = round(f_pass / len(subset) * 100, 1) if subset else 0
        print(f"    {diff.capitalize():<8}: {f_pass}/{len(subset)} faithful ({f_pct}%)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    output_csv = RESULTS_DIR / "faithfulness_eval.csv"
    fieldnames = [
        "id", "difficulty", "question",
        "retrieval_correct", "grounded", "verdict",
        "faithfulness", "category", "reason",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Append summary
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        f.write("\n# SUMMARY\n")
        f.write(f"faithful,{faithful},{faith_pct}%\n")
        f.write(f"unfaithful,{unfaithful},{round(unfaithful/total*100,1)}%\n")
        f.write(f"\n# CROSS-TAB\n")
        f.write(f"retrieval_pass+faithful_pass,{both_pass}\n")
        f.write(f"retrieval_fail+faithful_pass,{ret_fail_faith_pass}\n")
        f.write(f"retrieval_pass+faithful_fail,{ret_pass_faith_fail}\n")
        f.write(f"retrieval_fail+faithful_fail,{both_fail}\n")

    print(f"\n{'='*55}")
    print(f"  Faithfulness Rate : {faith_pct}% ({faithful}/{total})")
    print(f"{'='*55}")
    print(f"\n✅ Saved: {output_csv}")


if __name__ == "__main__":
    main()
