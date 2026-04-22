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
| 3 | Setup | Build vector database | Load chunks into ChromaDB | Claude | — | `scripts/build_index.py` / local |
| 4 | Setup | Create test questions | 20 questions answerable from the docs | Both | — | `data/questions.csv` / GitHub |
| 5 | Pipeline | Build RAG pipeline | Retrieve + generate script | Claude | — | `scripts/rag_pipeline.py` / GitHub |
| 6 | Eval | Run 20 questions through pipeline | Bulk test → JSON results | Vipin | — | `results/rag_results_*.json` / GitHub |
| 7 | Eval | Measure retrieval quality | Did the right chunk come back? | Claude | — | `results/retrieval_eval.csv` / GitHub |
| 8 | Eval | Measure generation quality | LLM-judge — correct answer? | Claude | — | `results/judge_results_*.json` / GitHub |
| 9 | Eval | Measure faithfulness | Did Claude stay within retrieved docs? | Claude | — | `results/faithfulness_eval.csv` / GitHub |
| 10 | Eval | Compare retrieval vs generation failures | Where does RAG break down? | Both | — | `results/rag_analysis.md` / GitHub |

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
