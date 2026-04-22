"""
llm_judge.py — LLM-as-Judge for RAG Answer Quality

For each question in the RAG results:
  1. Load the question, retrieved chunks, and Claude's answer
  2. Send all three to Claude acting as a judge
  3. Judge scores: CORRECT / COMPLETE / GROUNDED → overall VERDICT
  4. Save per-question results + summary

Usage:
    python3 scripts/llm_judge.py
    python3 scripts/llm_judge.py --results results/rag_results_20260422_143630.json

Output:
    results/judge_results_<timestamp>.json
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
RESULTS_DIR  = ROOT / "results"
JUDGE_PROMPT = ROOT / "prompts" / "judge_prompt.txt"

load_dotenv(ROOT / ".env")

MODEL      = "claude-opus-4-5"
MAX_TOKENS = 256


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_latest_results() -> Path:
    files = sorted(RESULTS_DIR.glob("rag_results_*.json"))
    if not files:
        raise FileNotFoundError("No rag_results_*.json found in results/")
    return files[-1]


def build_context(chunks: list[dict]) -> str:
    """Reconstruct the context string from saved chunks."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['topic']} — {chunk['section']}]\n{chunk['content']}"
        )
    return "\n\n".join(parts)


def call_judge(client: anthropic.Anthropic, prompt: str) -> tuple[str, int]:
    """Send judge prompt to Claude, return raw response + duration_ms."""
    start = time.time()
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    duration_ms = round((time.time() - start) * 1000)
    return message.content[0].text.strip(), duration_ms


def parse_judge_output(raw: str) -> dict:
    """Parse structured judge response into a dict."""
    result = {
        "correct":  None,
        "complete": None,
        "grounded": None,
        "verdict":  None,
        "reason":   None,
        "parse_error": False,
    }
    try:
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("CORRECT:"):
                result["correct"]  = line.split(":", 1)[1].strip()
            elif line.startswith("COMPLETE:"):
                result["complete"] = line.split(":", 1)[1].strip()
            elif line.startswith("GROUNDED:"):
                result["grounded"] = line.split(":", 1)[1].strip()
            elif line.startswith("VERDICT:"):
                result["verdict"]  = line.split(":", 1)[1].strip()
            elif line.startswith("REASON:"):
                result["reason"]   = line.split(":", 1)[1].strip()
    except Exception:
        result["parse_error"] = True
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                        help="Path to rag_results_*.json (default: latest)")
    args = parser.parse_args()

    results_file = Path(args.results) if args.results else find_latest_results()
    data         = json.loads(results_file.read_text(encoding="utf-8"))
    rag_results  = data["results"]
    total        = len(rag_results)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

    client       = anthropic.Anthropic(api_key=api_key)
    judge_tmpl   = JUDGE_PROMPT.read_text(encoding="utf-8")
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n⚖️  VP RAG Eval — LLM Judge")
    print(f"   Source  : {results_file.name}")
    print(f"   Model   : {MODEL}")
    print(f"   Total   : {total} questions\n")

    judge_results = []

    for i, r in enumerate(rag_results, 1):
        qid      = r["id"]
        question = r["question"]
        answer   = r["answer"]
        chunks   = r["chunks"]
        context  = build_context(chunks)

        print(f"[{i:02d}/{total}] Q{qid} ({r['difficulty']}) — {question[:55]}...")

        # Build judge prompt
        prompt = judge_tmpl.format(
            question=question,
            context=context,
            answer=answer,
        )

        # Call judge
        try:
            raw, duration_ms = call_judge(client, prompt)
            parsed = parse_judge_output(raw)
            status = "success"
        except Exception as e:
            raw        = f"ERROR: {e}"
            parsed     = {"correct": None, "complete": None, "grounded": None,
                          "verdict": "ERROR", "reason": str(e), "parse_error": True}
            duration_ms = 0
            status      = "error"

        verdict_icon = "✅" if parsed.get("verdict") == "PASS" else "❌"
        print(f"         CORRECT={parsed['correct']}  COMPLETE={parsed['complete']}  GROUNDED={parsed['grounded']}")
        print(f"         VERDICT={parsed['verdict']} {verdict_icon}  |  {parsed['reason']}")
        print(f"         {duration_ms}ms\n")

        judge_results.append({
            "id":               qid,
            "question":         question,
            "difficulty":       r["difficulty"],
            "retrieval_correct": r["retrieval_correct"],
            "answer":           answer,
            "correct":          parsed["correct"],
            "complete":         parsed["complete"],
            "grounded":         parsed["grounded"],
            "verdict":          parsed["verdict"],
            "reason":           parsed["reason"],
            "parse_error":      parsed.get("parse_error", False),
            "judge_raw":        raw,
            "duration_ms":      duration_ms,
            "status":           status,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    passed  = sum(1 for r in judge_results if r["verdict"] == "PASS")
    failed  = sum(1 for r in judge_results if r["verdict"] == "FAIL")
    errors  = sum(1 for r in judge_results if r["status"] == "error")
    pct     = round(passed / total * 100, 1)

    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        subset  = [r for r in judge_results if r["difficulty"] == diff]
        d_pass  = sum(1 for r in subset if r["verdict"] == "PASS")
        d_pct   = round(d_pass / len(subset) * 100, 1) if subset else 0
        print(f"   {diff.capitalize():<8}: {d_pass}/{len(subset)} passed ({d_pct}%)")

    output = {
        "metadata": {
            "timestamp":       timestamp,
            "model":           MODEL,
            "source_file":     results_file.name,
            "total":           total,
            "passed":          passed,
            "failed":          failed,
            "errors":          errors,
            "pass_rate":       f"{pct}%",
        },
        "results": judge_results,
    }

    output_file = RESULTS_DIR / f"judge_results_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"  Answer Pass Rate : {pct}% ({passed}/{total})")
    print(f"{'='*55}")
    print(f"\n✅ Saved: {output_file}")


if __name__ == "__main__":
    main()
