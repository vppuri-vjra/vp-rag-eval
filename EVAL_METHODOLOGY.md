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

## End-to-End Eval Steps

| Step | Part | Purpose | What | Owner | Status | Location |
|------|------|---------|------|-------|--------|----------|
| 1 | Setup | Create knowledge base | 20 cooking technique documents | Claude | ✅ Done | `data/docs/` / GitHub |
| 2 | Setup | Split documents into searchable pieces | Section-based chunking script | Claude | ✅ Done | `scripts/chunk_docs.py` / GitHub |
| 3 | Setup | Build vector database | Load 97 chunks into ChromaDB | Claude | ✅ Done | `scripts/build_index.py` / GitHub |
| 4 | Setup | Create test questions | 20 questions with expected doc + difficulty | Both | ✅ Done | `data/questions.csv` / GitHub |
| 5 | Pipeline | Build RAG pipeline | Retrieve + generate script | Claude | ✅ Done | `scripts/rag_pipeline.py` / GitHub |
| 6 | Eval | Run 20 questions through pipeline | Bulk test → JSON results | Vipin | ✅ Done | `results/rag_results_20260422_143630.json` / GitHub |
| 7 | Eval | Measure retrieval quality | Did the right chunk come back? | Claude | ✅ Done | `results/retrieval_eval.csv` / GitHub |
| 8 | Eval | Measure generation quality | LLM-judge — correct answer? | Claude | ⬅️ Next | `results/judge_results_*.json` / GitHub |
| 9 | Eval | Measure faithfulness | Did Claude stay within retrieved docs? | Claude | — | `results/faithfulness_eval.csv` / GitHub |
| 10 | Eval | Compare retrieval vs generation failures | Where does RAG break down? | Both | — | `results/rag_analysis.md` / GitHub |

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
| `all-MiniLM-L6-v2` | 384 | Fast, local, free — good for prototypes |
| `all-mpnet-base-v2` | 768 | Better accuracy, slower — still local |
| OpenAI `text-embedding-3-small` | 1536 | API call, costs money, higher accuracy |
| OpenAI `text-embedding-3-large` | 3072 | Best accuracy, higher cost |
| Anthropic (no embedding API) | — | Anthropic does not offer an embedding model |
