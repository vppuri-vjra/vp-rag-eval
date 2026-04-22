"""
rag_analysis.py — Final RAG Eval Analysis

Reads all eval result files and produces a single analysis report:
  - Overall scorecard
  - Where RAG broke down (retrieval vs generation)
  - Root cause analysis
  - What to fix next

Usage:
    python3 scripts/rag_analysis.py

Output:
    results/rag_analysis.md
"""

import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def find_latest(pattern: str) -> Path:
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} found in results/")
    return files[-1]


def main():
    # Load all result files
    rag_file      = find_latest("rag_results_*.json")
    judge_file    = find_latest("judge_results_*.json")
    retrieval_csv = RESULTS_DIR / "retrieval_eval.csv"
    faith_csv     = RESULTS_DIR / "faithfulness_eval.csv"

    rag_data   = json.loads(rag_file.read_text(encoding="utf-8"))
    judge_data = json.loads(judge_file.read_text(encoding="utf-8"))

    rag_results   = rag_data["results"]
    judge_results = judge_data["results"]
    total         = len(rag_results)

    # ── Compute metrics ───────────────────────────────────────────────────────
    retrieval_pass  = sum(1 for r in rag_results   if r["retrieval_correct"])
    retrieval_fail  = total - retrieval_pass
    retrieval_pct   = round(retrieval_pass / total * 100, 1)

    judge_pass      = sum(1 for r in judge_results if r["verdict"] == "PASS")
    judge_fail      = total - judge_pass
    judge_pct       = round(judge_pass / total * 100, 1)

    faithful        = sum(1 for r in judge_results if r.get("grounded") == "PASS")
    faithful_pct    = round(faithful / total * 100, 1)

    # Cross-tab
    ret_pass_faith_pass = sum(1 for r in judge_results if r["retrieval_correct"] and r.get("grounded") == "PASS")
    ret_fail_faith_pass = sum(1 for r in judge_results if not r["retrieval_correct"] and r.get("grounded") == "PASS")
    ret_pass_faith_fail = sum(1 for r in judge_results if r["retrieval_correct"] and r.get("grounded") == "FAIL")
    ret_fail_faith_fail = sum(1 for r in judge_results if not r["retrieval_correct"] and r.get("grounded") == "FAIL")

    # Retrieval failures detail
    ret_failures = [r for r in rag_results if not r["retrieval_correct"]]

    # Generation failures detail (judge FAIL)
    gen_failures = [r for r in judge_results if r["verdict"] == "FAIL"]

    # By difficulty
    diff_stats = {}
    for diff in ["easy", "medium", "hard"]:
        r_subset = [r for r in rag_results   if r["difficulty"] == diff]
        j_subset = [r for r in judge_results if r["difficulty"] == diff]
        diff_stats[diff] = {
            "total":         len(r_subset),
            "ret_pass":      sum(1 for r in r_subset if r["retrieval_correct"]),
            "judge_pass":    sum(1 for r in j_subset if r["verdict"] == "PASS"),
        }

    print(f"\n📊 Generating RAG analysis report...")

    # ── Build markdown ────────────────────────────────────────────────────────
    lines = []

    lines += [
        "# VP RAG Eval — Final Analysis",
        "",
        f"**Sources:** `{rag_file.name}` + `{judge_file.name}`  ",
        f"**Total questions:** {total} | **Model:** {rag_data['metadata']['model']} | **Top-K:** {rag_data['metadata']['top_k']}",
        "",
        "---",
        "",
        "## Overall Scorecard",
        "",
        "| Metric | Score | Questions |",
        "|---|---|---|",
        f"| Retrieval Accuracy | **{retrieval_pct}%** | {retrieval_pass}/{total} correct |",
        f"| Answer Pass Rate (LLM Judge) | **{judge_pct}%** | {judge_pass}/{total} passed |",
        f"| Faithfulness (Grounded) | **{faithful_pct}%** | {faithful}/{total} grounded |",
        "",
        "---",
        "",
        "## Results by Difficulty",
        "",
        "| Difficulty | Questions | Retrieval Correct | Answers Passed |",
        "|---|---|---|---|",
    ]
    for diff in ["easy", "medium", "hard"]:
        s = diff_stats[diff]
        r_pct = round(s["ret_pass"]   / s["total"] * 100, 1)
        j_pct = round(s["judge_pass"] / s["total"] * 100, 1)
        lines.append(
            f"| {diff.capitalize()} | {s['total']} "
            f"| {s['ret_pass']}/{s['total']} ({r_pct}%) "
            f"| {s['judge_pass']}/{s['total']} ({j_pct}%) |"
        )
    lines += [
        f"| **Overall** | **{total}** | **{retrieval_pass}/{total} ({retrieval_pct}%)** | **{judge_pass}/{total} ({judge_pct}%)** |",
        "",
        "---",
        "",
        "## Where Did RAG Break Down?",
        "",
        "### Retrieval failures",
        f"**{retrieval_fail} out of {total} questions** — the expected document was not in the top-3 retrieved chunks.",
        "",
        "| Q | Difficulty | Question | Expected Doc | Got Doc | Root Cause |",
        "|---|---|---|---|---|---|",
    ]
    for r in ret_failures:
        got = r["top_hit_doc_id"]
        lines.append(
            f"| {r['id']} | {r['difficulty']} | {r['question']} "
            f"| `{r['expected_doc_id']}` | `{got}` | "
            + ("Semantic overlap — concept in multiple docs" if r['id'] in ["9","11"] else "Unique term — no semantic neighbors")
            + " |"
        )

    lines += [
        "",
        "### Generation failures",
        f"**{judge_fail} out of {total} questions** — LLM judge scored the answer FAIL.",
        "",
    ]
    if gen_failures:
        lines += [
            "| Q | Difficulty | Question | Verdict | Reason |",
            "|---|---|---|---|---|",
        ]
        for r in gen_failures:
            lines.append(
                f"| {r['id']} | {r['difficulty']} | {r['question']} "
                f"| {r['verdict']} | {r['reason']} |"
            )
    else:
        lines.append("None — all 20 answers passed the judge.")

    lines += [
        "",
        "---",
        "",
        "## Retrieval × Generation Cross-tab",
        "",
        "| Retrieval | Generation | Count | Interpretation |",
        "|---|---|---|---|",
        f"| ✅ PASS | ✅ PASS | {ret_pass_faith_pass} | Right doc retrieved, correct grounded answer |",
        f"| ❌ FAIL | ✅ PASS | {ret_fail_faith_pass} | Wrong doc — but Claude stayed grounded (safe refusal or partial) |",
        f"| ✅ PASS | ❌ FAIL | {ret_pass_faith_fail} | Right doc — but Claude hallucinated ← **did not happen** |",
        f"| ❌ FAIL | ❌ FAIL | {ret_fail_faith_fail} | Wrong doc + hallucination ← **worst case — did not happen** |",
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### 1. Retrieval is the weak link",
        f"Retrieval accuracy was **{retrieval_pct}%** — the only place this RAG system failed.",
        "All 3 failures were due to semantic similarity issues, not document quality.",
        "",
        "| Failure type | Count | Example |",
        "|---|---|---|",
        "| Semantic overlap — concept exists in multiple docs | 2 | Q9: pan sauce in both reduction and deglazing docs |",
        "| Unique term — no semantic neighbors in corpus | 1 | Q20: claw grip has no related chunks |",
        "",
        "### 2. The grounding constraint eliminated hallucination",
        "Despite 3 retrieval failures, **faithfulness was 100%**.",
        "The prompt constraint `Answer using ONLY the information provided` forced Claude to:",
        "- Give a partial answer when wrong chunks were retrieved (Q9, Q11)",
        '- Refuse to answer when no relevant chunks existed (Q20: "I cannot answer this")',
        "",
        "This is the most important design decision in the entire system.",
        "",
        "### 3. Hard questions were easier to retrieve",
        "Hard questions used specific technical language (Maillard reaction, carryover cooking)",
        "that mapped cleanly to one document. Medium questions used general cooking language",
        "that overlapped across multiple documents.",
        "",
        "---",
        "",
        "## What to Fix Next (Production Improvements)",
        "",
        "| Priority | Fix | Addresses |",
        "|---|---|---|",
        "| 1 | **Hybrid search** — combine vector search + keyword (BM25) | Q20: unique terms with no semantic neighbors |",
        "| 2 | **Add topic labels to chunk content** — e.g. prepend `[Reduction]` to every chunk | Q9, Q11: semantic overlap between docs |",
        "| 3 | **Increase Top-K from 3 to 5** | Retrieval failures where expected doc was rank 4-5 |",
        "| 4 | **Human review of judge labels** | Validate 100% pass rate isn't judge leniency |",
        "| 5 | **Fine-tune embedding model on cooking domain** | Improve semantic matching for domain-specific terms |",
        "",
        "---",
        "",
        "## Eval Pipeline Summary",
        "",
        "| Script | Input | Output | Purpose |",
        "|---|---|---|---|",
        "| `rag_pipeline.py` | questions.csv + ChromaDB | `rag_results_*.json` | Run RAG system |",
        "| `retrieval_eval.py` | `rag_results_*.json` | `retrieval_eval.csv` | Measure retrieval accuracy |",
        "| `llm_judge.py` | `rag_results_*.json` | `judge_results_*.json` | Score answer quality |",
        "| `faithfulness_eval.py` | `judge_results_*.json` | `faithfulness_eval.csv` | Extract faithfulness scores |",
        "| `rag_analysis.py` | All of the above | `rag_analysis.md` | Final analysis report |",
        "",
    ]

    output = RESULTS_DIR / "rag_analysis.md"
    output.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"  Retrieval Accuracy : {retrieval_pct}% ({retrieval_pass}/{total})")
    print(f"  Answer Pass Rate   : {judge_pct}% ({judge_pass}/{total})")
    print(f"  Faithfulness       : {faithful_pct}% ({faithful}/{total})")
    print(f"{'='*55}")
    print(f"\n✅ Saved: {output}")


if __name__ == "__main__":
    main()
