# VP RAG Eval — Final Analysis

**Sources:** `rag_results_20260422_143630.json` + `judge_results_20260422_161419.json`  
**Total questions:** 20 | **Model:** claude-opus-4-5 | **Top-K:** 3

---

## Overall Scorecard

| Metric | Score | Questions |
|---|---|---|
| Retrieval Accuracy | **85.0%** | 17/20 correct |
| Answer Pass Rate (LLM Judge) | **100.0%** | 20/20 passed |
| Faithfulness (Grounded) | **100.0%** | 20/20 grounded |

---

## Results by Difficulty

| Difficulty | Questions | Retrieval Correct | Answers Passed |
|---|---|---|---|
| Easy | 7 | 6/7 (85.7%) | 7/7 (100.0%) |
| Medium | 9 | 7/9 (77.8%) | 9/9 (100.0%) |
| Hard | 4 | 4/4 (100.0%) | 4/4 (100.0%) |
| **Overall** | **20** | **17/20 (85.0%)** | **20/20 (100.0%)** |

---

## Where Did RAG Break Down?

### Retrieval failures
**3 out of 20 questions** — the expected document was not in the top-3 retrieved chunks.

| Q | Difficulty | Question | Expected Doc | Got Doc | Root Cause |
|---|---|---|---|---|---|
| 9 | medium | How do I build a pan sauce after searing meat? | `doc_09_reduction` | `doc_11_deglazing` | Semantic overlap — concept in multiple docs |
| 11 | medium | What is fond and why is it important for flavor? | `doc_11_deglazing` | `doc_02_sauteing` | Semantic overlap — concept in multiple docs |
| 20 | easy | What is the claw grip and why is it important? | `doc_20_knife_skills` | `doc_12_marinating` | Unique term — no semantic neighbors |

### Generation failures
**0 out of 20 questions** — LLM judge scored the answer FAIL.

None — all 20 answers passed the judge.

---

## Retrieval × Generation Cross-tab

| Retrieval | Generation | Count | Interpretation |
|---|---|---|---|
| ✅ PASS | ✅ PASS | 17 | Right doc retrieved, correct grounded answer |
| ❌ FAIL | ✅ PASS | 3 | Wrong doc — but Claude stayed grounded (safe refusal or partial) |
| ✅ PASS | ❌ FAIL | 0 | Right doc — but Claude hallucinated ← **did not happen** |
| ❌ FAIL | ❌ FAIL | 0 | Wrong doc + hallucination ← **worst case — did not happen** |

---

## Key Findings

### 1. Retrieval is the weak link
Retrieval accuracy was **85.0%** — the only place this RAG system failed.
All 3 failures were due to semantic similarity issues, not document quality.

| Failure type | Count | Example |
|---|---|---|
| Semantic overlap — concept exists in multiple docs | 2 | Q9: pan sauce in both reduction and deglazing docs |
| Unique term — no semantic neighbors in corpus | 1 | Q20: claw grip has no related chunks |

### 2. The grounding constraint eliminated hallucination
Despite 3 retrieval failures, **faithfulness was 100%**.
The prompt constraint `Answer using ONLY the information provided` forced Claude to:
- Give a partial answer when wrong chunks were retrieved (Q9, Q11)
- Refuse to answer when no relevant chunks existed (Q20: "I cannot answer this")

This is the most important design decision in the entire system.

### 3. Hard questions were easier to retrieve
Hard questions used specific technical language (Maillard reaction, carryover cooking)
that mapped cleanly to one document. Medium questions used general cooking language
that overlapped across multiple documents.

---

## What to Fix Next (Production Improvements)

| Priority | Fix | Addresses |
|---|---|---|
| 1 | **Hybrid search** — combine vector search + keyword (BM25) | Q20: unique terms with no semantic neighbors |
| 2 | **Add topic labels to chunk content** — e.g. prepend `[Reduction]` to every chunk | Q9, Q11: semantic overlap between docs |
| 3 | **Increase Top-K from 3 to 5** | Retrieval failures where expected doc was rank 4-5 |
| 4 | **Human review of judge labels** | Validate 100% pass rate isn't judge leniency |
| 5 | **Fine-tune embedding model on cooking domain** | Improve semantic matching for domain-specific terms |

---

## Eval Pipeline Summary

| Script | Input | Output | Purpose |
|---|---|---|---|
| `rag_pipeline.py` | questions.csv + ChromaDB | `rag_results_*.json` | Run RAG system |
| `retrieval_eval.py` | `rag_results_*.json` | `retrieval_eval.csv` | Measure retrieval accuracy |
| `llm_judge.py` | `rag_results_*.json` | `judge_results_*.json` | Score answer quality |
| `faithfulness_eval.py` | `judge_results_*.json` | `faithfulness_eval.csv` | Extract faithfulness scores |
| `rag_analysis.py` | All of the above | `rag_analysis.md` | Final analysis report |
