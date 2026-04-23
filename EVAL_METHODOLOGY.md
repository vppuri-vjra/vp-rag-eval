# VP RAG Eval — Overall Eval Methodology

**Course:** Understanding Evals  
**Project:** VP RAG Eval (cooking techniques knowledge base)  
**Model:** claude-opus-4-5 | **Documents:** 20 | **Queries:** 20 | **Date:** April 2026

---

## What is RAG

RAG (Retrieval Augmented Generation) solves the problem of LLMs not knowing your private or recent data.

```
User asks a question
        ↓
Search a document store for relevant chunks
        ↓
Inject those chunks into Claude's context
        ↓
Claude answers using only the retrieved content
```

RAG has two parts — both can fail:
| Part | Question | Failure mode |
|---|---|---|
| **Retrieval** | Did it find the right document? | Wrong chunks retrieved — garbage in, garbage out |
| **Generation** | Did Claude answer correctly from those docs? | Hallucinated beyond what the docs say |

---

## What is the RAG Pipeline

The RAG pipeline connects three previously separate pieces:

```
ChromaDB (97 chunks as vectors)  +  Questions CSV  +  Claude API
                    ↓
              RAG Pipeline (rag_pipeline.py)
```

**For each question, the pipeline runs 4 steps:**

```
Question: "How long should I blanch green beans?"
        ↓
Step 1 — RETRIEVE
  Search ChromaDB for top 3 most relevant chunks
  → returns: Blanching/TIMING GUIDE, Blanching/HOW TO BLANCH, Steaming/STEAMING TIMES

        ↓
Step 2 — BUILD PROMPT
  Inject retrieved chunks into Claude's context:
  "Answer using ONLY the information below. Do not use outside knowledge.

   [Blanching TIMING GUIDE chunk]
   [Blanching HOW TO BLANCH chunk]
   [Steaming STEAMING TIMES chunk]

   Question: How long should I blanch green beans?"

        ↓
Step 3 — GENERATE
  Claude answers using only retrieved content
  → "Green beans should be blanched for 2-3 minutes..."

        ↓
Step 4 — SAVE
  Store: question + retrieved chunks + Claude's answer + metadata
```

**Why "Answer using ONLY the information below" matters:**
This constraint forces Claude to stay within the retrieved content — it cannot use outside knowledge. This is what makes faithfulness measurable. If Claude answers from general knowledge instead of the retrieved chunks, that is a faithfulness failure.

**What gets saved for each question:**

| Field saved | Used in which eval step |
|---|---|
| Question | All steps |
| Retrieved chunk IDs | Step 7 — retrieval accuracy |
| Retrieved chunk content | Step 9 — faithfulness |
| Claude's answer | Step 8 — answer correctness |
| Expected doc (from questions.csv) | Step 7 — compare against retrieved |

---

## What rag_pipeline.py Contains

**4 functions — one per pipeline stage:**

| Function | What it does |
|---|---|
| `retrieve()` | Searches ChromaDB, returns top 3 chunks with doc_id, topic, section, content, distance score |
| `build_prompt()` | Takes 3 chunks + question, assembles grounded prompt with "Answer using ONLY this information" |
| `generate()` | Sends grounded prompt to Claude API, returns answer + response time |
| `main()` | Loops through all 20 questions, calls the 3 functions, saves everything to JSON |

**JSON output structure per question:**
```json
{
  "id": "1",
  "question": "How long should I blanch green beans?",
  "difficulty": "easy",
  "expected_doc_id": "doc_01_blanching",
  "retrieved_doc_ids": ["doc_01_blanching", "doc_19_steaming", "..."],
  "retrieval_correct": true,
  "top_hit_doc_id": "doc_01_blanching",
  "top_hit_topic": "Blanching",
  "top_hit_section": "TIMING GUIDE",
  "chunks": [...],
  "answer": "Green beans should be blanched for 2-3 minutes...",
  "duration_ms": 3200
}
```

**The key design decision — grounding constraint:**
```
"Answer using ONLY the information provided below.
Do not use outside knowledge."
```
Without this, Claude answers from training data and faithfulness cannot be measured.
This one constraint is what makes Steps 8 and 9 possible.

---

## Document Store

**20 cooking technique documents** — one topic per file.

| File | Topic |
|---|---|
| `doc_01_blanching.txt` | Blanching |
| `doc_02_sauteing.txt` | Sautéing |
| `doc_03_emulsification.txt` | Emulsification |
| `doc_04_braising.txt` | Braising |
| `doc_05_caramelization.txt` | Caramelization |
| `doc_06_maillard_reaction.txt` | Maillard Reaction |
| `doc_07_brining.txt` | Brining |
| `doc_08_tempering_chocolate.txt` | Tempering Chocolate |
| `doc_09_reduction.txt` | Reduction |
| `doc_10_poaching.txt` | Poaching |
| `doc_11_deglazing.txt` | Deglazing |
| `doc_12_marinating.txt` | Marinating |
| `doc_13_rendering_fat.txt` | Rendering Fat |
| `doc_14_blooming_spices.txt` | Blooming Spices |
| `doc_15_resting_meat.txt` | Resting Meat |
| `doc_16_folding.txt` | Folding vs Stirring |
| `doc_17_seasoning_cast_iron.txt` | Seasoning Cast Iron |
| `doc_18_making_roux.txt` | Making a Roux |
| `doc_19_steaming.txt` | Steaming |
| `doc_20_knife_skills.txt` | Knife Skills |

---

## Chunking Strategy

**Method chosen: Section-based chunking**

Each document is split into chunks at section headers (lines in ALL CAPS followed by a colon, e.g. `PURPOSE:`, `HOW TO BLANCH:`).

### Why section-based over alternatives

| Strategy | How | Why not chosen |
|---|---|---|
| Fixed size (200 words) | Split every N words regardless of content | Cuts mid-sentence, breaks context |
| Sliding window | Overlapping chunks of fixed size | Redundant chunks, more storage |
| Semantic | AI decides natural break points | Most accurate but complex to build |
| **Section-based** ✅ | Split on document headers | Clean, coherent, matches our doc structure |

### Chunk design
- Each chunk = one section from one document
- Chunk includes: document name + section header + section content
- Minimum chunk size: 3 lines (avoids empty or trivial chunks)
- Metadata stored with each chunk: `doc_id`, `topic`, `section`

### Why chunking matters
Storing whole documents in a vector database causes:
- Retrieval confusion — long docs cover many topics, search cannot identify the relevant part
- Context window bloat — too many long docs fill Claude's context with irrelevant content
- Lower retrieval precision — the right answer is buried in a large chunk

---

## Knowledge Progression — Full Table

| Step | Project / Topic | Before this step | What it does | What you learn | Status |
|------|----------------|-----------------|--------------|----------------|--------|
| 1 | VP Recipe Agent | No eval system | Build a recipe chatbot, run 20 test queries, measure pass/fail with a regex checker | System prompt design, bulk testing, TPR/TNR, confusion matrix, failure taxonomy, 3 Gulfs | ✅ Done |
| 2 | VP Substitution Agent | Checker not yet built | Build an ingredient substitution chatbot, iterate checker from V1→V4 until 0 false positives | Rule-based checker design, checker iteration, human review, ground truth labeling | ✅ Done |
| 3 | VP Substitution Agent — LLM Judge | Regex checker only — no meaning-based eval | Send all 20 responses to Claude acting as a judge — score each on 4 criteria | LLM-as-judge pattern, structured scoring, judge vs human agreement rate | ✅ Done |
| 4 | VP Substitution Agent — A/B Testing | One prompt, no comparison | Run two versions of the system prompt through the same 20 queries, compare scores | A/B prompt testing, one variable at a time, data-driven prompt decisions | ✅ Done |
| 5 | VP RAG Eval | No RAG — Claude answers from training data only | Build full RAG pipeline — 20 docs → chunk → embed (all-MiniLM-L6-v2, **384 dims**) → retrieve → generate → evaluate | Chunking, vector embeddings — each chunk becomes 384 numbers, ChromaDB stores and searches by cosine similarity | ✅ Done |
| 6a | Swap Embedding Model | **384 dims**, retrieval accuracy **85%** (17/20) | Replace all-MiniLM-L6-v2 with BGE-large (**1024 dims**), rebuild index, re-run 20 questions | More dimensions = richer vector = better at separating similar concepts. 85% → 95%. Q11 and Q20 fixed | ✅ Done |
| 6b | RAG Pattern — HyDE | **1024 dims**, retrieval accuracy **95%** (19/20), Q9 still failing | Ask Claude to write a hypothetical answer first, embed that (**1024 dims**) as the search query instead of the question | Question and document live in different vector spaces — hypothesis bridges the gap. 95% → 100%. Q9 fixed | ✅ Done |
| 6c | RAG Pattern — Re-ranking | **100%** with HyDE — testing a different pattern on BGE baseline (95%) | Retrieve top-10 with BGE-large (**1024 dim** vectors), re-score all 10 with cross-encoder, keep top-3 | Cross-encoder reads both texts together — no fixed dims. More stages ≠ better. Went to 90% (-5%) | ✅ Done |
| 6d | RAG Pattern — Branched RAG | Single retrieval path | Vector (BGE, 1024d) + BM25 in parallel, merge via RRF → top-3 | No single path covers everything — BM25 fixes exact terms, vector fixes meaning. Held at 95% | ✅ Done |
| 6e | RAG Pattern — Agentic RAG | Fixed pipeline — always retrieves, always same way | Let an agent decide whether to retrieve, what to retrieve, and how many times | Moving from fixed pipeline to dynamic decision-making | ⬅️ Next |
| 6f | RAG Pattern — Graph RAG | Flat chunks in vector DB | Store knowledge as a graph (entities + relationships) instead of flat chunks | Structured knowledge retrieval — better for connected concepts | — |
| 7 | LangChain | Everything wired manually | Rebuild the same RAG pipeline using LangChain abstractions | Framework fluency — chains, retrievers, prompt templates, memory | — |
| 8 | LlamaIndex | LangChain only | Use LlamaIndex for advanced chunking and data ingestion | Better chunking strategies, multi-modal, complex document pipelines | — |
| 9 | Agentic Eval | Eval for single Q→A only | Evaluate a multi-step agent — not just one question → one answer | Trace-level evaluation, tool use, non-deterministic chain scoring | — |
| 10 | Fine-tuning Eval | No baseline comparison framework | Compare model before and after fine-tuning on the same eval set | Regression testing, eval-driven fine-tune validation | — |
| 11 | DeepLearning.AI Prompt Engineering | Ad-hoc prompting | Work through structured course modules with exercises | Chain-of-thought, few-shot, prompt chaining, structured techniques | — |
| 12 | Production Eval Patterns | One-off experiments | Wire evals into CI/CD, build dashboards, add human review at scale | Evals in production — not just experiments but ongoing quality gates | — |
| 13 | Portfolio — Tell Phase | Work done but not articulated | Write about what you built, publish on LinkedIn, prep interview stories | Articulate AI eval work confidently to hiring managers and peers | — |

---

## End-to-End Eval Steps (RAG Pipeline)

| Step | Part | Purpose | What | Owner | Status | Location |
|------|------|---------|------|-------|--------|----------|
| 1 | Setup | Create knowledge base | 20 cooking technique documents | Claude | ✅ Done | `data/docs/` / GitHub |
| 2 | Setup | Split documents into searchable pieces | Section-based chunking script | Claude | ✅ Done | `scripts/chunk_docs.py` / GitHub |
| 3 | Setup | Build vector database | Load 97 chunks into ChromaDB | Claude | ✅ Done | `scripts/build_index.py` / GitHub |
| 4 | Setup | Create test questions | 20 questions with expected doc + difficulty | Both | ✅ Done | `data/questions.csv` / GitHub |
| 5 | Pipeline | Build RAG pipeline | Retrieve + generate script | Claude | ✅ Done | `scripts/rag_pipeline.py` / GitHub |
| 6 | Eval | Run 20 questions through pipeline | Bulk test → JSON results | Vipin | ✅ Done | `results/rag_results_20260422_143630.json` / GitHub |
| 7 | Eval | Measure retrieval quality | Did the right chunk come back? | Claude | ✅ Done | `results/retrieval_eval.csv` / GitHub |
| 8 | Eval | Measure generation quality | LLM-judge — correct answer? | Claude | ✅ Done | `results/judge_results_*.json` / GitHub |
| 9 | Eval | Measure faithfulness | Did Claude stay within retrieved docs? | Claude | ✅ Done | `results/faithfulness_eval.csv` / GitHub |
| 10 | Eval | Compare retrieval vs generation failures | Where does RAG break down? | Both | ✅ Done | `results/rag_analysis.md` / GitHub |

---

## Expected Doc vs Got Doc — Why They Differ

### Expected Doc — human-defined ground truth
When we wrote the 20 questions in Step 4, we manually tagged each with the document we knew contained the answer:

```
id, question,                          expected_doc_id,    expected_section
9,  "How do I build a pan sauce?",     doc_09_reduction,   PAN SAUCE METHOD
```

This is our ground truth — we know the right answer is in the reduction doc because we wrote both the document and the question.

### Got Doc — ChromaDB decided at runtime
When the pipeline ran, ChromaDB searched the vector database and returned the top 3 closest chunks by cosine similarity. It returned `doc_11_deglazing` — not because it was wrong in a dumb way, but because deglazing chunks were vectorially closer to the query than reduction chunks.

ChromaDB does not know our ground truth. It just finds the closest vectors.

| | Expected Doc | Got Doc |
|---|---|---|
| Who decided | Us (humans) | ChromaDB (algorithm) |
| When | Step 4 — before running | Step 6 — at runtime |
| Based on | Knowledge of document content | Vector similarity |
| Purpose | Ground truth for eval | Actual retrieval result |

**The gap between Expected and Got is exactly what retrieval accuracy measures.**

---

## Retrieval Failure Analysis — Step 6 Results

**Retrieval Accuracy: 85% (17/20)**

### 3 Retrieval Failures

| Q | Question | Expected Doc | Expected Section | Expected Chunk ID | Got Doc | Why it failed |
|---|---|---|---|---|---|---|
| 9 | "How do I build a pan sauce?" | `doc_09_reduction` | PAN SAUCE METHOD | `doc_09_reduction__pan_sauce_method` | `doc_11_deglazing` | Pan sauce exists in both docs — semantic overlap |
| 11 | "What is fond and why important?" | `doc_11_deglazing` | WHAT IS FOND | `doc_11_deglazing__what_is_fond` | `doc_02_sauteing` | "Fond" mentioned in Sautéing doc — keyword confusion |
| 20 | "What is the claw grip?" | `doc_20_knife_skills` | THE CLAW GRIP | `doc_20_knife_skills__the_claw_grip` | `doc_12_marinating` | "Claw grip" too specific — no semantic neighbors in other chunks |

### Chunks retrieved for each failure

**Q9 — "How do I build a pan sauce after searing meat?"**

| Rank | Expected | Actually Retrieved | Section | Distance |
|---|---|---|---|---|
| 1 | `doc_09_reduction` / PAN SAUCE METHOD | `doc_11_deglazing` | DO NOT | 0.44 |
| 2 | `doc_09_reduction` / PAN SAUCE METHOD | `doc_04_braising` | HOW TO BRAISE | 0.47 |
| 3 | `doc_09_reduction` / PAN SAUCE METHOD | `doc_02_sauteing` | HOW TO SAUTÉ | 0.48 |

**Q11 — "What is fond and why is it important for flavor?"**

| Rank | Expected | Actually Retrieved | Section | Distance |
|---|---|---|---|---|
| 1 | `doc_11_deglazing` / WHAT IS FOND | `doc_02_sauteing` | PURPOSE | 0.46 |
| 2 | `doc_11_deglazing` / WHAT IS FOND | `doc_12_marinating` | THREE COMPONENTS OF A MARINADE | 0.49 |
| 3 | `doc_11_deglazing` / WHAT IS FOND | `doc_12_marinating` | WHAT A MARINADE DOES | 0.50 |

**Q20 — "What is the claw grip and why is it important?"**

| Rank | Expected | Actually Retrieved | Section | Distance |
|---|---|---|---|---|
| 1 | `doc_20_knife_skills` / THE CLAW GRIP | `doc_12_marinating` | WHAT A MARINADE DOES | 0.79 |
| 2 | `doc_20_knife_skills` / THE CLAW GRIP | `doc_10_poaching` | PURPOSE | 0.82 |
| 3 | `doc_20_knife_skills` / THE CLAW GRIP | `doc_15_resting_meat` | WHAT HAPPENS DURING RESTING | 0.83 |

### Failure patterns and fixes

| Failure type | Questions | Fix |
|---|---|---|
| Semantic overlap — concept exists in multiple docs | Q9, Q11 | Add topic labels to chunk content, or hybrid search |
| Unique term — no semantic neighbors | Q20 | Add keyword-based retrieval as fallback (hybrid search) |

---

## Claude as Answerer vs Claude as Judge

In this eval, Claude plays two completely different roles across two separate steps. Same model — different prompt, different job.

### Role 1 — Answerer (Step 6)

Claude is the system being evaluated.

```
Prompt says:
"Answer the question using ONLY the information provided below.
Do not use outside knowledge."

Input:  question + retrieved chunks
Output: an answer
```

Claude's job: *produce an answer grounded in the retrieved content.*

---

### Role 2 — Judge (Step 8)

Claude is the evaluator reviewing the system's output.

```
Prompt says:
"You are an expert evaluator reviewing answers produced by a RAG system.
Your job is to evaluate the answer on three criteria."

Input:  question + retrieved chunks + the answer Claude gave in Step 6
Output: CORRECT / COMPLETE / GROUNDED scores + VERDICT + REASON
```

Claude's job: *score someone else's answer against the source material.*

---

### Why this works — Claude has no memory

Each API call is completely independent. Claude does not remember what it said in Step 6. When it receives the Step 8 judge prompt, it has no idea it was the one who generated the answer. It just reads the prompt and plays the role defined there.

**The prompt IS the role.** Change the prompt → change the behavior.

| | Step 6 — Answerer | Step 8 — Judge |
|---|---|---|
| Claude's role | Generate an answer | Score an answer |
| Input | Question + chunks | Question + chunks + answer |
| Output | Answer text | PASS/FAIL scores + reasoning |
| What it doesn't know | Nothing hidden | It was the one who wrote the answer |
| Defined by | `GROUNDED_PROMPT` in rag_pipeline.py | `judge_prompt.txt` in prompts/ |

### Why use the same model for both roles

- Consistency — same capability level evaluating its own tier of output
- Cost — no need to spin up a separate evaluation model
- Precedent — this is the standard LLM-as-judge pattern used in production evals

The risk: a model may be biased toward its own style of answers. In production you would validate LLM-judge scores against human labels to check for this bias. That is Step 10 in our eval.

---

## Retrieval Eval — Step 7 Results

**Script:** `scripts/retrieval_eval.py`
**Output:** `results/retrieval_eval.csv`

### Accuracy by difficulty

| Difficulty | Questions | Correct | Accuracy |
|---|---|---|---|
| Easy | 7 | 6 | 85.7% |
| Medium | 9 | 7 | 77.8% |
| Hard | 4 | 4 | 100.0% |
| **OVERALL** | **20** | **17** | **85.0%** |

### Key insight — Hard questions got 100%

Hard questions in this eval were questions where the answer was in a very specific doc and the question used clear topic-specific language (e.g. "Why can't you brown meat by boiling it?" → Maillard reaction doc). The model found these easily.

Medium questions had the most failures because those questions used general cooking language that overlapped between multiple docs ("pan sauce", "fond").

### What retrieval_eval.py produces per question

| Column | What it means |
|---|---|
| `retrieval_correct` | yes / no — was expected doc in top-3? |
| `rank_of_expected` | 1, 2, or 3 — or "not found" |
| `distance_of_expected` | Cosine distance of the expected chunk if found |
| `top_hit_distance` | Distance of the chunk that ranked #1 |

---

## LLM Judge — Step 8 Results

**Script:** `scripts/llm_judge.py`
**Judge prompt:** `prompts/judge_prompt.txt`
**Output:** `results/judge_results_20260422_161419.json`

### Pass rate by difficulty

| Difficulty | Questions | Passed | Pass Rate |
|---|---|---|---|
| Easy | 7 | 7 | 100% |
| Medium | 9 | 9 | 100% |
| Hard | 4 | 4 | 100% |
| **OVERALL** | **20** | **20** | **100%** |

### Key insight — 100% pass rate despite 3 retrieval failures

This is the most important finding of the eval. Even the 3 questions where retrieval failed (Q9, Q11, Q20) received a PASS from the judge. Why?

| Q | Retrieval result | What Claude did | Judge verdict |
|---|---|---|---|
| Q9 | Wrong doc retrieved — got deglazing instead of reduction | Answered from deglazing chunks — answer was still useful and grounded | PASS |
| Q11 | Wrong doc retrieved — got sautéing instead of deglazing | Answered partially from sautéing, noted limitations | PASS |
| Q20 | Completely wrong doc — no relevant content | Said "I cannot answer this from the provided information" | PASS ✅ |

**Q20 is the standout case:** The grounded constraint ("Answer using ONLY the information provided") worked exactly as designed. Claude had no relevant content, so it refused to hallucinate. The judge correctly rewarded this.

**The lesson:** A well-designed grounding constraint turns retrieval failures into safe "I don't know" responses rather than confident wrong answers.

---

## Faithfulness Eval — Step 9 Results

**Script:** `scripts/faithfulness_eval.py`
**Output:** `results/faithfulness_eval.csv`
**Source:** Extracts `GROUNDED` scores from `judge_results_*.json` — no extra API calls needed.

### Faithfulness rate

| Difficulty | Questions | Faithful | Rate |
|---|---|---|---|
| Easy | 7 | 7 | 100% |
| Medium | 9 | 9 | 100% |
| Hard | 4 | 4 | 100% |
| **OVERALL** | **20** | **20** | **100%** |

### Full results — Question + Answerer + Judge

| Q | Diff | Question | Retrieval | Answerer Response (summary) | Faithful | Judge Reason |
|---|---|---|---|---|---|---|
| 1 | easy | How long should I blanch green beans? | ✅ | Blanch green beans for **2-3 minutes** | ✅ | Correctly states 2-3 min, matches Source 1 exactly |
| 2 | medium | Why does overcrowding a pan prevent browning when sautéing? | ✅ | Food **steams instead of browns**; noted chunks don't explain the mechanism | ✅ | Correctly identifies steaming outcome, acknowledges limitation, no hallucination |
| 3 | medium | What ingredient in egg yolks makes mayonnaise stable? | ✅ | **Lecithin** in egg yolks stabilises mayonnaise | ✅ | Correctly identifies lecithin from Source 1, no extra claims |
| 4 | easy | What cuts of beef work best for braising? | ✅ | **Short ribs and brisket** — collagen-rich, tenderised by slow cooking | ✅ | Correct cuts and explanation, stays within chunks |
| 5 | easy | How do I caramelize onions properly and how long does it take? | ✅ | Medium-low heat, stir occasionally, **45-60 minutes**, don't rush | ✅ | All info from chunks, no outside knowledge added |
| 6 | hard | Why can't you brown meat by boiling it? | ✅ | Maillard reaction needs **>280°F and dry surface**; boiling = 212°F max + wet | ✅ | Both temperature and moisture conditions correct per Source 1 |
| 7 | easy | What is the salt to water ratio for a basic wet brine? | ✅ | **1 cup kosher salt per 1 gallon of water** | ✅ | Direct match to Source 1, no extras |
| 8 | medium | How do I test if chocolate is properly tempered? | ✅ | Dip knife tip — sets in **3-5 minutes** with **glossy finish** | ✅ | Exact extraction from Source 1 |
| 9 | medium | How do I build a pan sauce after searing meat? | ❌ | Partial answer from deglazing chunks (what to avoid) + noted full steps not available | ✅ | Used what was available, correctly flagged limitation — no hallucination |
| 10 | hard | Why does vinegar help when poaching eggs? | ✅ | **"I cannot answer this"** — chunks mention vinegar but don't explain why | ✅ | Correct refusal — chunks had the fact but not the explanation |
| 11 | medium | What is fond and why is it important for flavor? | ❌ | **Fond = browned bits** on pan bottom, used for building sauces; noted detail not in chunks | ✅ | Correct partial definition from sautéing chunk, acknowledged limitation |
| 12 | easy | How long should I marinate fish or seafood? | ✅ | **15-30 minutes only** — acid "cooks" fish, makes it mushy longer | ✅ | Time and reason both from chunks |
| 13 | medium | What is the difference between wet and dry rendering fat? | ✅ | Wet = water, lighter colour, less flavour. Dry = no water, more flavour, darker | ✅ | Complete and accurate summary of Source 1 |
| 14 | medium | What happens if I burn spices while blooming them? | ✅ | Turn **bitter**, result is **irreversible** — must start over | ✅ | Direct match to Source 1 |
| 15 | hard | How much does internal temperature rise after removing meat from heat? | ✅ | Small cuts **3-5°F**, medium roasts **5-10°F**, large roasts **10-15°F** | ✅ | All three ranges reproduced accurately |
| 16 | medium | Why should I use a spatula instead of a whisk when folding egg whites? | ✅ | Whisk is **"too aggressive"** and **breaks bubbles** | ✅ | Correct, grounded in Source 1 |
| 17 | easy | How do I restore a rusty cast iron pan? | ✅ | Scrub with steel wool → wash → dry → re-season (full 8-step process) | ✅ | Combines two chunks correctly, stays within retrieved content |
| 18 | medium | What happens if I add cold liquid to a roux? | ✅ | **Causes lumps** — add warm or hot liquid instead | ✅ | Direct citation from Source 1 |
| 19 | hard | How do aromatics added to steaming water affect the food? | ✅ | Steam carries **subtle flavour** into food; place aromatics under/around basket | ✅ | Accurate extraction from Source 2 |
| 20 | easy | What is the claw grip and why is it important? | ❌ | **"I cannot answer this"** — retrieved chunks had no relevant content | ✅ | Correct refusal — grounding constraint prevented hallucination |

### Cross-tab — Retrieval × Faithfulness

**✅ Retrieval PASS + ✅ Faithful PASS — 17 questions**

| Q | Question | Answerer Response | Judge Reason |
|---|---|---|---|
| 1 | How long should I blanch green beans? | Blanch for **2-3 minutes** | Correct, matches Source 1 exactly |
| 2 | Why does overcrowding a pan prevent browning? | Food **steams instead of browns**; noted mechanism not in chunks | Correct outcome, acknowledged limitation, no hallucination |
| 3 | What ingredient in egg yolks makes mayonnaise stable? | **Lecithin** in egg yolks stabilises mayonnaise | Correct, directly from Source 1 |
| 4 | What cuts of beef work best for braising? | **Short ribs and brisket** — collagen-rich, tenderised by slow cooking | Correct cuts and reason, stays within chunks |
| 5 | How do I caramelize onions and how long does it take? | Medium-low heat, stir occasionally, **45-60 minutes** | All info from chunks, nothing added |
| 6 | Why can't you brown meat by boiling it? | Maillard needs **>280°F and dry surface**; boiling = 212°F max + wet | Both conditions correct per Source 1 |
| 7 | What is the salt to water ratio for a basic wet brine? | **1 cup kosher salt per 1 gallon of water** | Direct match to Source 1 |
| 8 | How do I test if chocolate is properly tempered? | Dip knife tip — sets in **3-5 minutes** with **glossy finish** | Exact extraction from Source 1 |
| 10 | Why does vinegar help when poaching eggs? | **"I cannot answer this"** — chunks mention vinegar but don't explain why | Correct refusal — chunks had fact but not explanation |
| 12 | How long should I marinate fish or seafood? | **15-30 minutes only** — acid "cooks" fish, makes it mushy | Time and reason both from chunks |
| 13 | What is the difference between wet and dry rendering fat? | Wet = water, lighter, less flavour. Dry = no water, more flavour, darker | Complete and accurate summary |
| 14 | What happens if I burn spices while blooming them? | Turn **bitter**, result is **irreversible** — must start over | Direct match to Source 1 |
| 15 | How much does internal temperature rise after removing meat? | Small **3-5°F**, medium **5-10°F**, large **10-15°F** | All three ranges reproduced accurately |
| 16 | Why use a spatula instead of a whisk when folding egg whites? | Whisk is **"too aggressive"** and **breaks bubbles** | Correct, grounded in Source 1 |
| 17 | How do I restore a rusty cast iron pan? | Scrub → wash → dry → re-season (full 8-step process) | Combines two chunks correctly |
| 18 | What happens if I add cold liquid to a roux? | **Causes lumps** — add warm or hot liquid | Direct citation from Source 1 |
| 19 | How do aromatics added to steaming water affect the food? | Steam carries **subtle flavour**; place aromatics under/around basket | Accurate extraction from Source 2 |

---

**❌ Retrieval FAIL + ✅ Faithful PASS — 3 questions** ← grounding constraint worked

| Q | Question | Answerer Response | Judge Reason |
|---|---|---|---|
| 9 | How do I build a pan sauce after searing meat? | Partial answer from deglazing chunks (what to avoid); flagged full steps not available | Used what was available, correctly flagged limitation — no hallucination |
| 11 | What is fond and why is it important for flavor? | **Fond = browned bits** on pan bottom, used for sauces; noted detail not in chunks | Correct partial definition from sautéing chunk, acknowledged limitation |
| 20 | What is the claw grip and why is it important? | **"I cannot answer this"** — retrieved chunks had no relevant content | Correct refusal — grounding constraint prevented hallucination |

---

**✅ Retrieval PASS + ❌ Faithful FAIL — 0 questions** ← never hallucinated despite right chunks

**❌ Retrieval FAIL + ❌ Faithful FAIL — 0 questions** ← worst case never happened

### Key insight — the grounding constraint eliminated hallucination

The 3 retrieval failures (Q9, Q11, Q20) all landed in the **"Retrieval FAIL + Faithful PASS"** box. Claude answered from whatever chunks it received — it did not reach outside them.

This is the grounding constraint doing its job:
```
"Answer using ONLY the information provided below.
Do not use outside knowledge."
```

Without this constraint, a retrieval failure would likely cause a hallucination. With it, retrieval failure → partial answer or safe refusal.

---

## New Metrics (vs Substitution Eval)

| Metric | Measures | New concept |
|---|---|---|
| **Retrieval accuracy** | Did the right document/chunk come back? | ✅ New |
| **Faithfulness** | Did Claude answer using only retrieved content? | ✅ New |
| **Answer correctness** | Is the answer factually right? | ✅ New |
| Judge agreement rate | Does LLM-judge match human labels? | Same as substitution eval |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `sentence-transformers` | Convert text chunks to vectors (embeddings) |
| `ChromaDB` | Local vector database — stores and searches chunks |
| `Claude API` | Generation — answers questions using retrieved chunks |
| `llm_judge.py` | Evaluates answer quality and faithfulness |

---

## How Text → Vector Conversion Works

### Step by step

**1. Tokenization — text split into pieces**
```
"how long to blanch green beans"
→ ["how", "long", "to", "blanch", "green", "beans"]
```

**2. Neural network processes tokens**
The embedding model is a small BERT-style neural network trained on millions of sentences to understand meaning. Each token is converted to a vector, then all tokens are combined into one vector representing the whole sentence.

**3. Output — 384 numbers**
```
"how long to blanch green beans"
→ [0.23, -0.87, 0.45, 0.12, -0.33, 0.67, ... × 384]
```

**4. Similar meaning = close vectors**
```
"how long to blanch green beans"  → [0.23, -0.87, 0.45, ...]  ← similar
"blanching time for vegetables"   → [0.21, -0.85, 0.47, ...]  ← similar
"how to temper chocolate"         → [-0.45, 0.33, -0.12, ...] ← far away
```
The model learned that "blanch" and "blanching time" are related — their vectors end up close in 384-dimensional space.

**5. Cosine similarity — how search works**
ChromaDB measures the angle between the query vector and every stored chunk vector. Small angle = similar meaning = high relevance score. The top N closest chunks are returned.

---

### Why you cannot trace where 0.23 came from

A common question: *"how did the model come up with 0.23 for the word 'how'?"*

The honest answer: **you cannot trace it back.** Here's why:

`all-MiniLM-L6-v2` has 22 million parameters (weights) learned during training on 1 billion+ sentence pairs. When text goes in:

```
"how long to blanch green beans"
  → vocabulary lookup → initial vector
  → passed through 6 transformer layers
  → each layer applies matrix multiplications using 22M weights
  → output: 384 numbers including 0.23
```

The 0.23 is the result of millions of multiplications and additions. There is no human-readable explanation for why it is specifically 0.23.

**"how" alone means nothing — context changes everything:**
```
"how long to blanch green beans" → [0.23, -0.87, 0.45, ...]
"how to temper chocolate"        → [0.31, -0.12, 0.78, ...]
```
Same word "how", different sentence, different vector. The model reads the whole sentence together — not word by word.

**The 384 dimensions are not interpretable individually.**
No one knows what each dimension represents — not even the researchers who built the model. What matters is the pattern across all 384 numbers together. Two sentences with similar meaning end up with similar patterns.

**Analogy — GPS coordinates:**
- New York: 40.71°N, 74.00°W
- Boston: 42.36°N, 71.06°W

You cannot explain why New York's latitude is exactly 40.71 — it just is, based on the coordinate system. But you can tell New York and Boston are closer to each other than New York and Tokyo. Vectors work the same way — individual numbers are not meaningful, the **distance between vectors** is what matters for search.

---

## Embedding Model — all-MiniLM-L6-v2

| Property | Value |
|---|---|
| **Model name** | `all-MiniLM-L6-v2` |
| **Made by** | Microsoft (via sentence-transformers library) |
| **Architecture** | MiniLM — distilled (compressed) version of BERT |
| **Output dimensions** | 384 |
| **Max input tokens** | 256 tokens (~200 words) |
| **Size** | ~80MB — fast to load and run |
| **Trained on** | 1 billion+ sentence pairs from web, books, QA datasets |
| **Similarity metric** | Cosine similarity |
| **Speed** | Very fast — suitable for local use without GPU |

### Why this model for our eval
- No API key required — runs locally
- Fast enough for 97 chunks without GPU
- Good semantic understanding for cooking/English text
- Industry standard for RAG prototypes and evals

### Limitations
- Max 256 tokens — chunks longer than ~200 words get truncated
- Not fine-tuned on cooking domain — general purpose only
- Smaller models can miss nuanced meaning vs larger models (e.g. OpenAI `text-embedding-3-large`)

### Production alternatives
| Model | Dimensions | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast, local, free — good for prototypes ← we started here |
| `BAAI/bge-large-en-v1.5` | 1024 | Better semantic understanding, still local and free ← Step 6a |
| `all-mpnet-base-v2` | 768 | Better accuracy, slower — still local |
| OpenAI `text-embedding-3-small` | 1536 | API call, costs money, higher accuracy |
| OpenAI `text-embedding-3-large` | 3072 | Best accuracy, higher cost |
| Anthropic (no embedding API) | — | Anthropic does not offer an embedding model |

---

## Step 6a — Embedding Model Swap: all-MiniLM-L6-v2 → BGE-large

**Change made:** One line in `scripts/build_index.py` and `scripts/rag_pipeline.py`

```python
# Before
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384 dimensions

# After
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 1024 dimensions
```

Then rebuilt ChromaDB index and re-ran all 20 questions.

### Results comparison

| | all-MiniLM-L6-v2 | BGE-large | Change |
|---|---|---|---|
| **Overall** | 85% (17/20) | **95% (19/20)** | **+10% ✅** |
| Easy | 85.7% (6/7) | 100% (7/7) | +14.3% ✅ |
| Medium | 77.8% (7/9) | 88.9% (8/9) | +11.1% ✅ |
| Hard | 100% (4/4) | 100% (4/4) | — |

### What changed per failure

| Q | Before (MiniLM) | After (BGE-large) | Why |
|---|---|---|---|
| Q9 — pan sauce | ❌ got deglazing | ❌ still got deglazing | Genuine content overlap — pan sauce concept lives in both docs. Distance 0.25 = model is very confident in wrong answer. Not a model problem. |
| Q11 — fond | ❌ got sautéing | ✅ **fixed** | BGE-large's richer 1024-dim vector understood that "fond" belongs in deglazing context, not sautéing |
| Q20 — claw grip | ❌ not in top-3 | ✅ **fixed** | BGE-large found knife_skills at rank 2 — richer representation of "claw grip" as a specific technique |

### Key insight — what more dimensions actually did

More dimensions = more detail encoded per chunk = better at separating similar-but-different concepts.

```
384 dimensions (MiniLM):  "pan sauce" ≈ "deglazing" ≈ "searing" (too similar)
1024 dimensions (BGE):    "pan sauce" ≈ "reduction", "deglazing" = related but distinct
```

Q11 and Q20 were fixable with a better model. Q9 is not — the content overlap is real regardless of model quality. That is a retrieval architecture problem, not a model problem. **HyDE or re-ranking is the fix for Q9.**

### Why Q9 is different

Distance score tells the story:

| | MiniLM distance | BGE distance |
|---|---|---|
| Q9 top hit (wrong) | 0.44 | **0.25** |

BGE-large is *more confident* in the wrong answer than MiniLM was. The pan sauce concept is so tightly encoded in the deglazing doc that a richer model doubles down on it. This confirms the failure is content-based, not model-based.

---

## Step 6b — HyDE (Hypothetical Document Embeddings)

**Script:** `scripts/hyde_pipeline.py`
**Output:** `results/hyde_results_20260422_211525.json`

### What HyDE does differently

Standard RAG searches with the question. HyDE searches with a hypothetical answer.

```
Standard RAG:
  "How do I build a pan sauce after searing meat?"
  → question vector → matches deglazing doc ❌

HyDE RAG:
  "How do I build a pan sauce after searing meat?"
        ↓ Claude generates hypothesis:
  "To build a pan sauce, remove the meat, deglaze with wine, add stock
   and reduce by half to concentrate the flavors and create a silky sauce..."
        ↓ hypothesis vector → matches reduction doc ✅
```

Questions ask. Documents answer. Hypothesis reads like a document — so it matches document content better than a question does.

### The HyDE prompt

```
You are a culinary expert writing content for a cooking techniques reference guide.
Write a short passage (3-5 sentences) that directly answers the following question.
Write it as if it were a section of a cookbook — factual, specific, instructional.
Do not say "I" or reference the question. Just write the answer as document content.
```

### Results

| | Standard RAG (MiniLM) | Standard RAG (BGE-large) | HyDE RAG (BGE-large) |
|---|---|---|---|
| **Embedding model** | all-MiniLM-L6-v2 | BGE-large | BGE-large |
| **Dimensions** | 384 | 1024 | 1024 |
| **Search query** | Original question | Original question | Claude's hypothetical answer |
| **Overall accuracy** | 85% (17/20) | 95% (19/20) | **100% (20/20)** |
| **Easy** | 85.7% (6/7) | 100% (7/7) | 100% (7/7) |
| **Medium** | 77.8% (7/9) | 88.9% (8/9) | **100% (9/9)** |
| **Hard** | 100% (4/4) | 100% (4/4) | 100% (4/4) |
| **Q9 — pan sauce** | ❌ | ❌ | ✅ **fixed** |
| **Q11 — fond** | ❌ | ✅ fixed | ✅ |
| **Q20 — claw grip** | ❌ | ✅ fixed | ✅ |

### What fixed each failure

| Failure | Root cause | Fix |
|---|---|---|
| Q11 — fond | Weak semantic representation — 384 dims couldn't separate fond/sautéing | Better model (BGE-large, 1024 dims) |
| Q20 — claw grip | No semantic neighbors — MiniLM couldn't place "claw grip" near knife skills | Better model (BGE-large, 1024 dims) |
| Q9 — pan sauce | Question vocabulary matched wrong doc — "pan sauce" ≈ deglazing | HyDE — hypothesis reads like reduction doc content |

### Key insight — they fix different types of problems

- **Better model** → fixes failures caused by weak or missing semantic representation
- **HyDE** → fixes failures caused by question-document vocabulary mismatch

Both were needed. Neither alone got to 100%.

---

## Step 6c — Re-ranking

**Script:** `scripts/rerank_pipeline.py`
**Output:** `results/rerank_results_20260422_212807.json`

### How re-ranking works

Two-stage retrieval — speed first, then accuracy:

```
Stage 1 — Bi-encoder (vector search):
  Query vector vs chunk vectors — compared SEPARATELY
  Fast — retrieve top-10 candidates

Stage 2 — Cross-encoder (re-ranker):
  [Query + Chunk] read TOGETHER in one pass
  Scores each of the 10 pairs 0-1 for relevance
  Return top-3 by score
```

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Input | Query vector + chunk vector separately | Query + chunk text together |
| Speed | Fast — pre-computed | Slow — must re-score each pair |
| Accuracy | Good | Better — sees full context |
| Role | First pass — cast wide net | Second pass — refine |

**Model used:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — free, local

### Results — re-ranking went DOWN

| Pipeline | Accuracy | Delta |
|---|---|---|
| Standard RAG (MiniLM) | 85% (17/20) | baseline |
| Standard RAG (BGE-large) | 95% (19/20) | +10% |
| HyDE RAG (BGE-large) | 100% (20/20) | +15% |
| **Re-ranking RAG** | **90% (18/20)** | **-5% ⚠️** |

**What broke:**
- Q9 — still failing (expected — re-ranking can't fix vocabulary mismatch, that's HyDE's job)
- Q20 — **newly broke** — claw grip was in the top-10 vector results but the cross-encoder scored it lower than unrelated chunks

### Teaching points — why re-ranking underperformed here

**1. Re-ranking is not always better**
The cross-encoder is trained on MS MARCO — a general web search dataset. "Claw grip" is a domain-specific term. The re-ranker didn't recognise it as relevant to knife skills and downranked the correct chunk.

**2. Re-ranking fixes a different problem than HyDE**
- HyDE fixes: question vocabulary ≠ document vocabulary
- Re-ranking fixes: vector search returns the right docs but in wrong order
On our dataset, vector search already had good ordering after BGE-large. Re-ranking added noise instead of signal.

**3. Tool mismatch — our corpus is too small**
Re-ranking shines when vector search returns 10-50 relevant-looking candidates that need to be sorted precisely. With 97 chunks and specific section-based docs, vector search was already precise enough. Re-ranking added a model that wasn't calibrated to our domain.

**4. More stages = more failure points**
Each stage can fail independently. Adding re-ranking added a second failure point without fixing the one we had (Q9).

### The real lesson

Re-ranking is the right tool when:
- Your corpus is large (1000s of chunks)
- Vector search returns many relevant-looking candidates in wrong order
- You have a domain-specific re-ranker or enough data to fine-tune one

Re-ranking is the wrong tool when:
- Your failure is vocabulary mismatch (use HyDE instead)
- Your corpus is small and vector search is already precise
- The re-ranker model is out-of-domain

### Full comparison across all patterns

| | Standard (MiniLM) | Standard (BGE) | HyDE (BGE) | Re-rank (BGE) |
|---|---|---|---|---|
| **Embedding** | 384d | 1024d | 1024d | 1024d |
| **Search query** | Question | Question | Hypothesis | Question |
| **Re-ranker** | — | — | — | Cross-encoder |
| **Overall** | 85% | 95% | **100%** | 90% |
| **Q9 — pan sauce** | ❌ | ❌ | ✅ | ❌ |
| **Q11 — fond** | ❌ | ✅ | ✅ | ✅ |
| **Q20 — claw grip** | ❌ | ✅ | ✅ | ❌ |
| **Best for** | Prototypes | Better semantics | Vocab mismatch | Large corpora |

---

## Why No Single Retrieval Path Is Enough — The Case for Branched RAG

Every retrieval method has a blind spot. This is why Branched RAG (Step 6d) exists.

| Retrieval path | How it works | Good at | Bad at |
|---|---|---|---|
| **Vector search (semantic)** | Converts query to 384 or 1024 dim vector, finds closest chunk vectors by cosine similarity | Understanding meaning — "pan sauce" finds related cooking concepts even without exact word match | Exact rare terms — "claw grip" had no semantic neighbors, so MiniLM couldn't find it |
| **Keyword search (BM25)** | Counts exact word matches, ranks by term frequency | Exact rare terms — "claw grip" found instantly if those exact words exist in a chunk | Understanding meaning — "build a pan sauce" won't match a chunk that says "reduction method" |

**The lesson from our failures:**

| Failure | What went wrong | Right fix |
|---|---|---|
| Q20 — claw grip (MiniLM) | "claw grip" had no semantic neighbors — vector search guessing | Keyword search (BM25) — finds exact term |
| Q9 — pan sauce | Question vocabulary ≠ document vocabulary | HyDE or Branched RAG |
| Q11 — fond (MiniLM) | 384 dims too coarse to separate fond/sautéing | Better model (1024 dims) |

**Branched RAG runs both paths and merges results:**
```
Question
    ├── Path A: Vector search  → top-3 by cosine similarity
    └── Path B: Keyword (BM25) → top-3 by term frequency
               ↓
         Merge + deduplicate → best chunks from both
               ↓
         Generate answer
```

This is also called **Hybrid Search** — the most common production RAG pattern.

---

## Step 6d — Branched RAG (Hybrid Search)

**Script:** `scripts/branched_pipeline.py`
**Output:** `results/branched_results_20260423_100705.json`

### How it works

```
Question
    ├── Path A: BGE-large vector search (top-5)   → semantic similarity
    └── Path B: BM25 keyword search (top-5)        → exact term frequency
               ↓
         Reciprocal Rank Fusion (RRF) — merge and score
               ↓
         Top-3 unique chunks by combined RRF score
               ↓
         Generate grounded answer
```

### Reciprocal Rank Fusion (RRF)

The merge formula used in production hybrid search:

```
RRF score = 1/(rank + 60)   ← for each path
            ↓ sum both paths
```

- K=60 dampens the effect of very high ranks
- Chunk ranked #1 in both paths → highest combined score
- Chunk only in one path → lower score but still included

### Results

| | Standard (BGE) | Branched RAG | HyDE |
|---|---|---|---|
| **Overall** | 95% (19/20) | **95% (19/20)** | 100% (20/20) |
| **Q9 — pan sauce** | ❌ | ❌ | ✅ |
| **Q20 — claw grip** | ✅ | ✅ (via BM25) | ✅ |

### What the Sources column tells us

| Source label | Meaning |
|---|---|
| `both` | Chunk appeared in top-5 of both paths — strong signal |
| `both+vector` | Mix of chunks from both paths and vector-only |
| `bm25+vector` | No chunk appeared in both — paths retrieved different chunks |
| `bm25+both+vector` | All three categories present in top-3 |

Q20 (claw grip) showed `bm25+vector` — BM25 found it by exact term match, vector search found something different. The merge brought the right chunk in.

### Key findings

**1. BM25 confirmed to fix Q20**
"Claw grip" appeared in BM25 top-5 by exact keyword match. Vector search had also fixed it with BGE-large. But this confirms: if we only had MiniLM (384 dims), BM25 would have rescued Q20.

**2. Q9 still not fixed**
"Pan sauce" + "searing" → BM25 also matched deglazing doc (it mentions "pan", "sauce", "searing" too). The overlap exists at the keyword level as well as the vector level. Only HyDE fixes this because it changes what gets searched, not how.

**3. 95% — same as standard BGE, better than re-ranking**
Branched RAG matched the BGE baseline without regression. Unlike re-ranking which dropped to 90%, branched RAG held steady. It's additive — BM25 adds keyword coverage without hurting semantic retrieval.

### When to use each pattern

| Pattern | Best when | Our result |
|---|---|---|
| Better model (BGE-large) | Corpus has domain-specific terms needing richer semantics | 85% → 95% ✅ |
| HyDE | Question vocabulary ≠ document vocabulary | 95% → 100% ✅ |
| Re-ranking | Large corpus, many candidates need ordering | 90% (-5%) ❌ wrong tool |
| **Branched RAG** | Mix of semantic + exact term queries in same corpus | 95% (stable) ✅ |
