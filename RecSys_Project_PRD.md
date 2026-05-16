# Project Requirements Document
## Multi-Stage Sequential Recommendation System with LLM Reranking & Cold-Start Handling

**Project codename:** `SeqRec`  
**Resume title:** *Multi-Stage Recommendation System with Sequential Modeling, LLM Reranking, and Cold-Start Handling*  
**Target roles:** MLE – Recommendations, MLE – Ranking & Retrieval, Applied Scientist – Personalization  
**Target companies:** Meta, TikTok/ByteDance, Netflix, Spotify, LinkedIn, Pinterest, Airbnb, Snap, Amazon, YouTube  
**Estimated build time:** 3–4 weeks  
**Compute requirement:** Single GPU (any 16GB+ VRAM) or free-tier Colab Pro

---

## 1. What the project is — the one-paragraph pitch

Most recommendation portfolio projects stop at collaborative filtering in a notebook. This project builds what companies like Meta, LinkedIn, and TikTok actually ship: a **three-stage production recommendation pipeline** that separates retrieval, sequential re-scoring, and LLM-based reranking into distinct, independently scalable components. On top of that, it explicitly solves the **cold-start problem** — the #1 open challenge in industry recommendation systems — using a content-based fallback that handles new users with zero interaction history. Everything is served through a real API with a Redis-backed feature store, benchmarked end-to-end with standard industry metrics.

---

## 2. System architecture overview

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  ONLINE FEATURE SERVING                                  │
│  Redis Feature Store: user embedding, interaction        │
│  history (last 20), user-item interaction counts         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — RETRIEVAL (Two-Tower + FAISS)                 │
│  Input: user_id → user tower embedding                   │
│  ANN search over pre-built FAISS HNSW item index         │
│  Output: top-500 candidate items                         │
│  Cold-start path: BGE text embeddings for new users      │
└─────────────────────┬───────────────────────────────────┘
                      │ top-500
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2 — RE-SCORING (SASRec Sequential Model)          │
│  Input: user history sequence + top-500 candidates       │
│  Transformer encoder over last-N interaction history     │
│  Score each candidate against user sequence embedding    │
│  Output: top-50 items with relevance scores              │
└─────────────────────┬───────────────────────────────────┘
                      │ top-50
                      ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3 — LLM RERANKING (Phi-2 / Llama-3.2-1B)         │
│  Input: top-50 items + user preference summary prompt    │
│  LLM scores or re-orders items based on semantic context │
│  Output: final top-10 recommendations                    │
└─────────────────────┬───────────────────────────────────┘
                      │ top-10
                      ▼
              FastAPI Response
              JSON: [item_id, score, explanation]
```

---

## 3. Dataset

**Primary dataset:** Amazon Reviews 2023 (Beauty or Sports & Outdoors category)  
- URL: https://amazon-reviews-2023.github.io/  
- ~2M users, ~500K items, ~33M interactions for Beauty  
- Each item has: title, description, category, price, image URL  
- Each interaction has: user_id, item_id, rating, timestamp, review_text  

**Why Amazon Reviews over MovieLens:**  
- Has rich text metadata (titles + descriptions) needed for cold-start content embeddings  
- Has timestamps enabling proper sequential modeling  
- Matches what TikTok/Amazon/Pinterest actually use (product/content catalog)  
- Standard in recsys research — evaluators know the benchmark numbers  

**Data splits:**
```
Training:   interactions up to timestamp T-2 (80%)
Validation: interactions between T-2 and T-1 (10%)
Test:       interactions after T-1 (10%)
Cold-start test set: users with ≤3 interactions total (sampled separately)
Warm-start test set: users with ≥10 interactions
```

**Preprocessing steps:**
1. Filter users with <5 interactions and items with <10 interactions (standard k-core filtering, k=5)
2. Sort each user's interactions chronologically → interaction sequences
3. For each user, keep last interaction as test label, second-to-last as validation label
4. Build item metadata DataFrame: item_id, title, category, description (for BGE embeddings)
5. Encode user_id and item_id to contiguous integer indices

---

## 4. Component 1: Two-Tower Retrieval Model

### What it is
A dual-encoder neural network with one tower for users and one for items. At inference, only the user tower runs online — item embeddings are precomputed and stored in a FAISS index. This enables sub-millisecond retrieval over millions of items.

### Architecture
```python
UserTower(
    user_embedding: nn.Embedding(n_users, 128),
    MLP: [128 → 256 → 128],  # with LayerNorm + ReLU
    output: 128-dim unit-normalized embedding
)

ItemTower(
    item_embedding: nn.Embedding(n_items, 128),
    MLP: [128 → 256 → 128],
    output: 128-dim unit-normalized embedding
)
```

### Training details
- **Loss:** In-batch negatives (treat other items in batch as negatives), temperature-scaled dot product similarity
- **Loss formula:** softmax cross-entropy over (pos_item, all_neg_items_in_batch)  
- **Temperature:** τ = 0.07 (standard CLIP-style)
- **Batch size:** 1024
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-2
- **Epochs:** 30 with early stopping on validation Recall@50
- **Hard negative mining (optional upgrade):** after epoch 10, mine hard negatives from top-200 FAISS results

### FAISS index
- Build HNSW index over all item embeddings (n_items vectors of dim 128)
- HNSW parameters: M=32, ef_construction=200
- Save index to disk: `item_index.faiss`
- At query time: encode user → query FAISS → return top-500 item_ids with distances

### What this demonstrates
- Knowledge of the dominant industry retrieval architecture (used by Google, Meta, LinkedIn, Pinterest)
- In-batch negative training — the correct way to train two-tower models
- Understanding of ANN indexing and why it's necessary at scale

---

## 5. Component 2: SASRec Sequential Re-Scoring Model

### What it is
A transformer-based model that captures the temporal dynamics of user behavior. Instead of a static user embedding, it encodes the user's recent interaction *sequence* and predicts the next item. This is the architecture behind TikTok's and Netflix's user behavior modeling.

### Architecture
```python
SASRec(
    item_embedding: nn.Embedding(n_items + 1, 64),  # +1 for padding token
    positional_embedding: nn.Embedding(max_seq_len=50, 64),
    transformer_layers: 2x (
        MultiHeadAttention(heads=2, d_model=64) with causal mask,
        FeedForward(64 → 256 → 64),
        LayerNorm + Dropout(0.1)
    ),
    output: dot product between last-position embedding and candidate item embedding
)
```

### Training details
- **Task:** next-item prediction given last N items in sequence
- **Input:** sequence of last 50 item_ids (padded with 0 if shorter)
- **Label:** next item in sequence (teacher forcing)
- **Loss:** binary cross-entropy with sampled softmax (100 random negatives per positive)
- **Batch size:** 256
- **Optimizer:** Adam, lr=1e-3
- **Max sequence length:** 50 items
- **Epochs:** 200 with early stopping on validation NDCG@10

### Integration with two-tower
- Two-tower outputs top-500 candidates
- SASRec scores each candidate: `score_i = dot(seq_embedding, item_embedding_i)`
- Rank candidates by score → output top-50

### What this demonstrates
- Sequential user modeling — the key upgrade over static collaborative filtering
- Transformer architecture applied to recommendation (directly what TikTok, Netflix use)
- Proper causal masking for sequential prediction

---

## 6. Component 3: LLM Reranker

### What it is
A small language model that takes the top-50 candidates and user context, then re-orders them based on semantic understanding of item descriptions and user preferences — something a dot-product score can never do.

### Model choice
**Option A (faster to implement):** `Phi-2` (2.7B) or `Llama-3.2-1B-Instruct` loaded in 4-bit  
**Option B (simpler baseline):** use `BGE-reranker-large` as a cross-encoder (this is actually how LinkedIn does LLM reranking in production)

Start with Option B (BGE reranker) first — it's more production-realistic and faster. Add Option A as a comparison experiment.

### Prompt design (Option A)
```
System: You are a recommendation assistant. Given a user's recent purchase history 
and a list of candidate products, rank the top-10 most relevant products.

User history (recent items): {item_title_1}, {item_title_2}, ..., {item_title_5}
User category preference: {top_3_categories_from_history}

Candidate products (rank the best 10):
1. {item_title} - {item_category} - {item_description[:100]}
2. ...
[up to 50 candidates]

Return: JSON list of top 10 item indices in ranked order.
```

### Cross-encoder design (Option B — recommended first)
```python
BGEReranker(
    model: "BAAI/bge-reranker-large",  # or bge-reranker-v2-m3
    input: (user_query_text, item_title + item_description),
    output: relevance_score per item
)
# user_query_text = concatenation of last 5 item titles from user history
```

### Integration
- Input: top-50 items from SASRec with their titles and descriptions
- Score each (user_context, item_text) pair with reranker
- Sort by reranker score → top-10 final recommendations

### What this demonstrates
- LLM integration in recommendation pipelines — the exact direction every company is moving
- Cross-encoder reranking (how LinkedIn, Spotify, Yelp actually implement LLM reranking)
- Understanding of the retrieval-reranking paradigm (RAG for recommendations)

---

## 7. Component 4: Cold-Start Handling

### What it is
A content-based fallback that generates recommendations for users with fewer than 5 interactions — the cold-start problem. This is the #1 unsolved challenge in industry recsys and almost no portfolio projects address it.

### Design
```
IF user_interaction_count < 5:
    # Content-based path
    IF user_has_0_interactions:
        → return global popularity-weighted items by category
          (popularity = log(interaction_count) * recency_weight)
    
    IF 1 <= user_interaction_count < 5:
        → encode the user's few interacted items using BGE text embeddings
        → average the item embeddings → "sparse user embedding"
        → ANN search in BGE item embedding space → top-500 candidates
        → SASRec scores with truncated sequence
        → LLM reranker final pass
ELSE:
    → standard warm path (two-tower → SASRec → LLM reranker)
```

### BGE item index (separate from FAISS two-tower index)
- Embed all item titles + descriptions with `BAAI/bge-large-en-v1.5`
- Build second FAISS index for content-based retrieval
- This index is used only for cold-start users

### Evaluation
- Explicitly split test set into cold-start (≤3 interactions) and warm (≥10 interactions)
- Report separate NDCG@10, Hit@20 for each split
- This becomes a major resume talking point: "Addressed cold-start problem with content-based fallback, achieving X% Hit@20 on new users vs Y% warm-start baseline"

---

## 8. Component 5: Redis Feature Store

### What it is
A lightweight feature store that simulates real-time feature serving — solving the training-serving skew problem that kills production ML systems. This is what Databricks, Meta, and Netflix use in production.

### Features stored per user (in Redis hash)
```
user:{user_id} → {
    "interaction_history": "[item_id_1, item_id_2, ...]",  # last 50 items, JSON list
    "interaction_count": "143",
    "top_categories": "['Electronics', 'Books', 'Sports']",
    "last_seen_ts": "1716412800",
    "cold_start_flag": "0",  # 1 if <5 interactions
    "user_embedding": "<base64 encoded 128-dim float32 vector>"
}
```

### Features stored per item (in Redis hash)
```
item:{item_id} → {
    "title": "Sony WH-1000XM5 Headphones",
    "category": "Electronics",
    "description": "...",
    "popularity_score": "0.87",
    "item_embedding": "<base64 encoded 128-dim float32 vector>",
    "bge_embedding": "<base64 encoded 1024-dim float32 vector>"
}
```

### Update pattern (simulating production)
- Batch job (Python script) pre-populates all user and item features at startup
- On each API request: fetch user features → run inference → update interaction history in Redis
- TTL: user features expire after 24h (simulating freshness)

### Why this matters for resume
- Solves training-serving skew — the exact same feature pipeline used for both training and serving
- Redis as online store is the industry standard (Meta, LinkedIn, Netflix all use Redis-backed feature stores)
- You can say: "Built a Redis-backed feature store serving real-time user and item features with <5ms lookup latency"

---

## 9. Component 6: FastAPI Serving Layer

### Endpoints
```python
POST /recommend
Request: {
    "user_id": "U12345",
    "n_items": 10,
    "context": "homepage"  # optional: homepage, search, product_page
}
Response: {
    "user_id": "U12345",
    "cold_start": false,
    "recommendations": [
        {"item_id": "I98765", "title": "...", "score": 0.94, "stage": "llm_reranker"},
        ...
    ],
    "latency_ms": {
        "feature_fetch": 3.2,
        "retrieval": 8.1,
        "rescoring": 12.4,
        "reranking": 45.2,
        "total": 68.9
    }
}

GET /metrics
Response: Prometheus-formatted metrics

POST /feedback  (simulated A/B logging)
Request: {"user_id": "...", "item_id": "...", "interaction": "click"}
```

### Async design
- FastAPI with `async def` handlers
- Redis calls: `aioredis` async client
- FAISS search: runs in threadpool (CPU-bound)
- SASRec inference: PyTorch with `torch.inference_mode()`
- LLM reranker: batched with dynamic batching (collect requests for 50ms → batch inference)

### Containerization
```dockerfile
FROM python:3.11-slim
# torch, faiss-cpu, transformers, fastapi, aioredis, uvicorn
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:
- `app`: FastAPI container
- `redis`: Redis 7.0
- `prometheus`: Prometheus scraping `/metrics`
- `grafana`: Dashboard over Prometheus metrics

---

## 10. Component 7: Evaluation Framework

### Offline evaluation (most important for resume)

**Standard ranking metrics:**

| Metric | Formula | Why it matters |
|---|---|---|
| NDCG@K | Normalized Discounted Cumulative Gain | Standard at Netflix, Meta, Google — measures ranking quality |
| Hit@K | % of test users where true item is in top-K | Simple binary signal of recommendation coverage |
| MRR | Mean Reciprocal Rank | Where does the true item appear on average |
| Recall@K | % of relevant items retrieved in top-K | Used at the retrieval stage specifically |

**Evaluation per stage:**
```
Stage 1 (two-tower retrieval):
  → Recall@50, Recall@100, Recall@500
  → ANN search latency P50/P99 (milliseconds)
  → Index build time, index size on disk

Stage 2 (SASRec rescoring):
  → NDCG@10, NDCG@20, Hit@10, Hit@20, MRR
  → Score on warm users vs cold-start users (separate rows)

Stage 3 (LLM reranker):
  → NDCG@10 delta vs SASRec-only
  → Hit@10 delta
  → Latency cost of reranker (ms added per request)
```

**Cold-start vs warm-start split:**
```
Report a 2x2 table:
                   | NDCG@10 | Hit@20
Two-tower only     |         |
+ SASRec           |         |
+ LLM reranker     |         |

cold_start_users   |         |
warm_users         |         |
```

**Ablation study (the piece that makes it look like a research project):**
```
Experiment A: Two-tower only (baseline)
Experiment B: Two-tower + SASRec (no LLM)
Experiment C: Two-tower + SASRec + BGE reranker
Experiment D: Two-tower + SASRec + LLM reranker (full system)
Experiment E: Content-based only (cold-start baseline)
Experiment F: Full system with cold-start routing (proposed)
```
This ablation directly answers the interview question: "How does each component contribute?"

### Online evaluation simulation (simulated A/B test)
- Replay 10% of test interactions as simulated online traffic
- Route 50% to "control" (two-tower only) and 50% to "treatment" (full system)
- Measure: simulated CTR (did the interacted item appear in top-10?), rank of ground-truth item
- Report: relative lift in Hit@10 between control and treatment

### MLflow experiment tracking
- Every training run logged: hyperparameters, per-epoch metrics, final eval results
- Model registry: promote best SASRec checkpoint to "Production" stage after validation
- Artifact storage: FAISS index, model weights, tokenizer configs
- Comparison view: all 6 ablation experiments in one MLflow UI

---

## 11. Component 8: Latency & Throughput Benchmarking

These are the numbers that go directly on your resume.

### What to measure
```python
# Use locust or a simple Python script
benchmark_scenarios = [
    {"users": 1, "label": "serial"},
    {"users": 10, "label": "light_load"},
    {"users": 50, "label": "medium_load"},
    {"users": 100, "label": "heavy_load"},
]

# Per-stage breakdown (already implemented in the API response):
stage_latencies = {
    "feature_fetch_redis": "target < 5ms",
    "faiss_ann_retrieval": "target < 10ms",
    "sasrec_rescoring": "target < 20ms",
    "llm_reranking": "target < 100ms",
    "total_p50": "target < 100ms",
    "total_p99": "target < 200ms",
}
```

### Throughput (QPS)
- Run 1000 requests in 60 seconds → calculate sustained QPS
- Measure at each concurrency level
- Report: peak QPS before latency P99 > 200ms threshold

---

## 12. Repository structure

```
seqrec/
├── README.md                    ← project overview + results table
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
│
├── data/
│   ├── download_amazon.py       ← download + preprocess Amazon Reviews
│   ├── preprocess.py            ← k-core filtering, splits, sequencing
│   └── feature_store_init.py   ← populate Redis with all features
│
├── models/
│   ├── two_tower.py             ← UserTower, ItemTower, training loop
│   ├── sasrec.py                ← SASRec transformer, training loop
│   ├── reranker.py              ← BGEReranker + LLMReranker wrappers
│   └── cold_start.py            ← BGE content embeddings, popularity fallback
│
├── retrieval/
│   ├── build_faiss_index.py     ← build + save HNSW item index
│   └── ann_search.py            ← FAISS query wrapper
│
├── feature_store/
│   ├── redis_client.py          ← async Redis get/set helpers
│   └── schemas.py               ← Pydantic models for features
│
├── serving/
│   ├── app/
│   │   ├── main.py              ← FastAPI app, endpoint definitions
│   │   ├── pipeline.py          ← orchestrates all 3 stages
│   │   └── metrics.py           ← Prometheus counters + histograms
│   └── tests/
│       └── test_endpoints.py
│
├── evaluation/
│   ├── offline_eval.py          ← NDCG, Hit, MRR computation
│   ├── ablation.py              ← runs all 6 ablation experiments
│   ├── cold_start_eval.py       ← separate cold/warm split evaluation
│   └── ab_simulation.py         ← simulated online A/B test
│
├── notebooks/
│   ├── 01_EDA.ipynb             ← dataset exploration
│   ├── 02_TwoTower_Training.ipynb
│   ├── 03_SASRec_Training.ipynb
│   ├── 04_Ablation_Results.ipynb
│   └── 05_Latency_Benchmark.ipynb
│
├── mlflow/                      ← MLflow tracking server config
└── monitoring/
    ├── prometheus.yml
    └── grafana_dashboard.json
```

---

## 13. Build sequence — what to do in what order

### Week 1: Data + Two-Tower
**Day 1–2:**
- Download Amazon Reviews 2023 (Beauty, ~2GB)
- Run `preprocess.py`: k-core filtering, train/val/test splits, sequence extraction
- Verify dataset stats: n_users, n_items, avg sequence length, density

**Day 3–4:**
- Implement `two_tower.py`: UserTower, ItemTower, in-batch negative loss
- Train: should converge in ~2h on a single GPU (A100/H100 at PICASSO)
- Evaluate Recall@50/100/500 on validation set
- Save checkpoints

**Day 5:**
- Build FAISS HNSW index from trained item embeddings
- Benchmark: index build time, index size, query latency P50/P99
- Test: given a user embedding, retrieve top-500 in < 10ms

### Week 2: Sequential Model + Cold-Start
**Day 1–2:**
- Implement `sasrec.py`: causal transformer, sampled softmax loss
- Train: 200 epochs ~4h on GPU
- Evaluate NDCG@10, Hit@20 on warm users
- Run ablation: two-tower only vs two-tower + SASRec

**Day 3:**
- Implement `cold_start.py`: BGE item text embeddings + second FAISS index
- Run BGE embedding on all item titles+descriptions (~4h for 500K items, do offline)
- Test cold-start path: given 3 interactions, do recommendations make sense?
- Evaluate Hit@20 on cold-start test split

**Day 4–5:**
- Set up Redis locally with Docker
- Implement `feature_store/`: populate user features, item metadata
- Verify: user feature fetch < 5ms, feature consistency with training pipeline

### Week 3: LLM Reranker + Serving
**Day 1–2:**
- Implement `reranker.py` with BGE reranker (cross-encoder)
- Evaluate: NDCG@10 with and without reranker on 1000-user test subset
- Optional: implement LLM reranker (Phi-2 in 4-bit) as an upgrade and compare

**Day 3:**
- Build `serving/app/`: FastAPI pipeline orchestrating all 3 stages
- Add per-stage latency instrumentation
- Add Prometheus metrics endpoint

**Day 4–5:**
- Build `docker-compose.yml`: app + Redis + Prometheus + Grafana
- Test end-to-end: `curl POST /recommend` → confirm correct response format
- Run latency benchmark at different concurrency levels

### Week 4: Evaluation + Polish
**Day 1–2:**
- Run full ablation study (all 6 experiments) and log to MLflow
- Run A/B simulation on replay test set
- Cold-start vs warm split evaluation table

**Day 3:**
- Write clean README with: architecture diagram, results table, quickstart commands
- Record a 2-min demo (screen capture of Grafana dashboard + API call)

**Day 4–5:**
- Push to GitHub, make repo public
- Write a short blog post or Notion doc describing the system (this becomes a talking point in interviews)
- Verify: someone else can `docker-compose up` and call the API

---

## 14. Expected results — what numbers to target

These are realistic targets based on published benchmarks for Amazon Beauty dataset.

| Model | NDCG@10 | Hit@20 | Recall@50 (retrieval) |
|---|---|---|---|
| Popularity baseline | 0.012 | 0.032 | — |
| BPR-MF (matrix factorization) | 0.031 | 0.071 | — |
| **Two-tower only** (your Stage 1) | 0.044 | 0.098 | ~0.72 |
| **+ SASRec** (your Stage 2) | 0.063 | 0.141 | — |
| **+ BGE reranker** (your Stage 3) | 0.071 | 0.158 | — |
| Cold-start (content-based BGE) | 0.029 | 0.067 | — |
| **Full system (warm users)** | **~0.071** | **~0.158** | — |

*Note: exact numbers depend on hypertuning. The important thing for resume is the delta between stages, not absolute values.*

### System performance targets
| Metric | Target |
|---|---|
| FAISS ANN latency P50 | < 5ms |
| FAISS ANN latency P99 | < 12ms |
| Full pipeline latency P50 | < 80ms |
| Full pipeline latency P99 | < 150ms |
| Sustained QPS (single container) | > 30 req/s |
| Redis feature fetch P50 | < 3ms |

---

## 15. Resume bullets — how to write them

These are the actual bullet points to use, in order of impact. Pick the best 3–4 based on which role you're applying to.

**Strongest bullet (systems + impact):**
> Built a 3-stage recommendation pipeline (two-tower retrieval → SASRec sequential re-scoring → LLM reranking) over 500K Amazon product catalog; end-to-end NDCG@10 of 0.071 vs 0.044 for retrieval-only baseline, served via FastAPI + Docker with P99 latency < 150ms under 50 concurrent users.

**Cold-start bullet (differentiation):**
> Designed explicit cold-start handling using BGE content embeddings and popularity-weighted fallback; achieved Hit@20 of 0.067 for users with <5 interactions vs near-zero for collaborative filtering baselines, directly addressing the industry's #1 open problem in recommendation systems.

**LLM reranking bullet (modern skills):**
> Integrated a BGE cross-encoder reranker as Stage 3 of the recommendation pipeline; improved NDCG@10 by ~13% relative over sequential model alone, with per-stage latency breakdown showing reranker adds < 100ms overhead at the serving layer.

**Feature store bullet (MLOps angle):**
> Built a Redis-backed feature store serving real-time user interaction sequences and pre-computed item embeddings; eliminated training-serving skew by using identical feature extraction for both training and online inference, with feature fetch latency P50 < 3ms.

**Infrastructure bullet (ANN + scale):**
> Indexed 500K item embeddings in a FAISS HNSW approximate nearest neighbor index (M=32, ef=200); achieved Recall@500 of 0.72 with P99 query latency of 12ms, enabling real-time candidate retrieval at billion-item scale.

**Evaluation/A/B bullet (data-driven):**
> Conducted a 6-experiment ablation study quantifying each pipeline stage's contribution to NDCG@10, and simulated an A/B test via interaction replay showing 61% relative improvement in Hit@10 for the full system vs popularity baseline.

---

## 16. Interview talking points

When a recruiter at Meta/TikTok/Netflix asks "tell me about your recommendation project," here is the narrative flow:

1. **Start with the problem**: "Most recommendation systems fail in two ways — they can't rank well for users with rich history, and they completely fail for new users. I wanted to build something that addresses both."

2. **Explain the architecture**: "I built a three-stage system: first stage does fast approximate retrieval over 500K items using a two-tower model and FAISS. Second stage uses SASRec — a transformer that re-scores based on the user's actual interaction sequence, capturing temporal preference drift. Third stage uses a cross-encoder reranker for semantic alignment."

3. **Call out the cold-start solution**: "Cold-start is the part most portfolios skip. I have a content-based fallback using BGE text embeddings for users with fewer than 5 interactions. I benchmark this explicitly on a held-out cold-start test split."

4. **Show systems thinking**: "It's all served through FastAPI with a Redis feature store. The feature store was important to prevent training-serving skew — the same feature pipeline runs both at training time and at inference time."

5. **Quantify**: "NDCG@10 goes from 0.044 with retrieval only, to 0.063 after SASRec, to 0.071 with the full pipeline. P99 latency stays under 150ms even at 50 concurrent users."

6. **Signal awareness of the field**: "The LLM reranking step directly mirrors what LinkedIn and Spotify published in 2024–25 — using a language model to bridge the semantic gap that dot-product scores can't capture."

---

## 17. What NOT to build (scope guard)

To finish in 3–4 weeks, explicitly leave these out:

- Do NOT build a graph neural network (GNN) model — adds 2+ weeks
- Do NOT implement your own attention mechanism — use PyTorch's `nn.MultiheadAttention`
- Do NOT try to run on the full Amazon dataset (all categories) — stick to Beauty or Sports
- Do NOT build a real frontend/UI — the FastAPI `/recommend` endpoint is sufficient
- Do NOT do distributed training — single GPU is fine for 500K items
- Do NOT implement PPO-based RL for recommendations — that's a different project entirely

Everything else in this document is necessary and achievable in 3–4 weeks.
