# SeqRec — Full Upgrade Agent Prompt
## Purpose
This file is a complete, self-contained instruction set for an AI coding agent (Claude Code, Cursor, Codex, etc.) to bring the SeqRec project from its current state to a production-grade recommendation system whose resume bullets will pass a TikTok / Meta / Netflix MLE screen. Read every section before touching any file.

---

## 0. Context: What This Project Is and Where It Stands

### What this project is supposed to be
A three-stage recommendation pipeline:
1. **Two-Tower** dual-encoder → FAISS HNSW ANN index → top-500 candidate retrieval
2. **SASRec** causal transformer → re-scores top-500 → top-50
3. **BGE cross-encoder reranker** → re-scores top-50 → final top-10
4. **Cold-start routing** for users with <5 interactions (BGE content embeddings + popularity fallback)
5. **Redis feature store** for online user/item feature serving
6. **FastAPI serving** layer with per-stage latency breakdown
7. **Offline evaluation** with full ablation study and MLflow tracking

### Current state (what is already done — do NOT rewrite these)
- `evaluation/offline_eval.py` — full ranking metrics (NDCG, Hit, MRR, Recall), sampled eval, popularity-sampled eval. **Complete. Do not touch.**
- `evaluation/ablation.py` — ablation study framework. **Complete. Do not touch.**
- `evaluation/cold_start_eval.py` — cold/warm user split evaluation. **Complete. Do not touch.**
- `evaluation/ab_simulation.py` — A/B replay simulation. **Complete. Do not touch.**
- `evaluation/mlflow_logging.py` — MLflow result logging helpers. **Complete. Do not touch.**
- `feature_store/redis_client.py` — async Redis get/set with typed schemas. **Complete. Do not touch.**
- `feature_store/schemas.py` — Pydantic user/item feature schemas. **Complete. Do not touch.**
- `docker-compose.yml` — app + Redis + Prometheus + Grafana. **Complete. Do not touch.**
- `Dockerfile` — FastAPI container. **Complete. Do not touch.**

### What is broken or missing (the full gap list)
| Gap | Severity | Description |
|---|---|---|
| Dataset | **Critical** | Current dataset (Sports & Outdoors, 319K users) is too large for CPU training. Two-tower trained 3 epochs on CPU; loss barely moved. Must switch to a fast-training benchmark dataset. |
| Two-Tower training | **Critical** | Current Recall@50 = 0.0029 (essentially random). Model never converged. Needs GPU training with proper hyperparameters and early stopping. |
| SASRec training | **Critical** | Only a smoke test exists. No real evaluation numbers. SASRec must be fully trained and evaluated before any pipeline results are valid. |
| Device detection | **Critical** | All training code currently hardcodes `cpu`. Must auto-detect CUDA → MPS → CPU in a single shared utility. |
| End-to-end pipeline | **High** | No script wires all three stages together. Ablation study cannot run because there are no trained artifacts. |
| Unified evaluation script | **High** | No single script runs all 6 ablation experiments and produces a results table. Each evaluator runs in isolation. |
| `data/` directory | **High** | No `download_amazon.py` or `preprocess.py` exists in the repo. Data download and preprocessing must be implemented. |
| `data/feature_store_init.py` | **High** | No script to populate Redis with trained embeddings. Must be created. |
| `models/two_tower.py` | **High** | File exists but training loop has no early stopping, no LR scheduler, no gradient clipping. |
| `models/sasrec.py` | **High** | Smoke-test only. No full training loop with sampled softmax loss, no checkpoint saving. |
| `models/cold_start.py` | **Medium** | Wrapper stub exists but BGE embedding generation and second FAISS index are not wired up. |
| `models/reranker.py` | **Medium** | BGE reranker wrapper exists but is not integrated into the serving pipeline. |
| `retrieval/` | **Medium** | `build_faiss_index.py` and `ann_search.py` exist but are not called from training pipeline. |
| `serving/app/pipeline.py` | **Medium** | Pipeline is stubbed to return 503. Must be wired to real artifacts. |
| `README.md` results table | **Low** | Placeholder only. Should be filled with real numbers after training. |

---

## 1. Dataset Decision — Use Amazon Beauty (McAuley 5-core)

### Why switch from Sports & Outdoors
- Sports & Outdoors after k-core filtering: 319K users, 74K items, 2.6M interactions. Two-tower training on this corpus requires 20+ GPU-hours to converge. On CPU it is infeasible.
- Amazon Beauty 5-core after k-core filtering: ~22K users, ~12K items, ~198K interactions. Full two-tower training < 30 min on GPU. Full SASRec training (200 epochs) < 2 hours on GPU. This is the dataset used in the original SASRec paper, meaning your numbers can be directly compared to published baselines.
- You will NOT lose resume credibility. Beauty 5-core is a standard industrial benchmark cited in SASRec (ICDM 2018), BERT4Rec (RecSys 2019), and dozens of follow-on papers. Reviewers at TikTok/Meta know this dataset.

### Download instructions (implement in `data/download_amazon.py`)
```
URL: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/All_Beauty.csv
Interactions file: All_Beauty.csv (columns: user_id, item_id, rating, timestamp)
Metadata file: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz
```
If the McAuley lab URL changes, fallback to:
```
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 (All_Beauty subset)
```

### Preprocessing spec (implement in `data/preprocess.py`)
1. Load interactions CSV: `user_id, item_id, rating, timestamp`
2. Apply 5-core filtering: keep only users with ≥5 interactions AND items with ≥5 interactions. Repeat until stable.
3. Sort each user's interactions by timestamp ascending → interaction sequence
4. Encode user_id → contiguous int (0-indexed). Encode item_id → contiguous int (1-indexed, 0 reserved for padding).
5. Split: for each user, last interaction = test label, second-to-last = validation label, rest = training sequence
6. Output artifacts to `data/processed/beauty/`:
   - `interactions.parquet` — full filtered interaction table with encoded IDs
   - `train_sequences.parquet` — user_id, sequence (list of item_ids, chronological, excluding last 2)
   - `val_labels.parquet` — user_id, val_item_id
   - `test_labels.parquet` — user_id, test_item_id
   - `item_meta.parquet` — item_id (encoded), asin, title, category, description
   - `user_encoder.json` — {original_user_id: encoded_int}
   - `item_encoder.json` — {original_item_id: encoded_int}
   - `dataset_stats.json` — n_users, n_items, n_interactions, avg_seq_len, sparsity
7. Cold-start split: flag users where total interactions ≤ 5 as cold_start=True in interactions.parquet
8. Log all stats using Python `logging` module at INFO level

### Expected stats after preprocessing
```
n_users:        ~22,000
n_items:        ~12,000
n_interactions: ~198,000
avg_seq_len:    ~9
sparsity:       ~0.075%
```

---

## 2. Device Detection Utility — Shared Across All Training Code

### Create `utils/device.py`
This file must be created and imported by every training script. It must not be duplicated.

```python
"""Shared device detection for all SeqRec training and inference code."""
import torch

def get_device() -> torch.device:
    """
    Return the best available device in priority order:
      1. CUDA (NVIDIA GPU) — if torch.cuda.is_available()
      2. MPS  (Apple Silicon) — if torch.backends.mps.is_available()
      3. CPU  — fallback
    Prints the selected device at INFO level.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] Using: {device}")
    return device
```

**Rules for using this utility:**
- Every training script must call `device = get_device()` at the top of `main()`.
- Every model must be moved to device with `.to(device)` immediately after construction.
- Every tensor must be moved to device with `.to(device)` before any operation.
- Never hardcode `"cpu"`, `"cuda"`, or `"mps"` anywhere in training code.
- For MPS compatibility: use `float32` only (MPS does not support `float64`). Use `torch.float32` explicitly wherever dtype matters.

---

## 3. Two-Tower Model — Fix and Full Training

### File: `models/two_tower.py`
The model architecture is acceptable. The following must be fixed or added:

#### Architecture (keep as-is from PRD)
```
UserTower: Embedding(n_users, 128) → Linear(128, 256) → ReLU → LayerNorm → Linear(256, 128) → L2-normalize
ItemTower: Embedding(n_items, 128) → Linear(128, 256) → ReLU → LayerNorm → Linear(256, 128) → L2-normalize
Loss: In-batch negatives with temperature-scaled dot product (τ=0.07)
```

#### What must be fixed in the training loop
1. **Device**: Replace all hardcoded `"cpu"` with `device = get_device()` from `utils/device.py`.
2. **Early stopping**: Stop training when validation Recall@50 has not improved for 5 consecutive epochs.
3. **LR scheduler**: Add `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)`.
4. **Gradient clipping**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before every optimizer step.
5. **Unique positive batching**: Already implemented. Keep it. This is the `unique items per batch` policy that improved Recall@50.
6. **Checkpoint saving**: Save best checkpoint (by validation Recall@50) to `models/artifacts/beauty_tower/best_checkpoint.pt`. Save: `{'model_state_dict': ..., 'recall_at_50': ..., 'epoch': ..., 'n_users': ..., 'n_items': ...}`.
7. **Embedding export**: After training, export all item embeddings as `numpy` array to `models/artifacts/beauty_tower/item_embeddings.npy`. Shape: `(n_items, 128)`.
8. **Training script**: Create `scripts/train_two_tower.py` that loads `data/processed/beauty/train_sequences.parquet`, trains the model, and saves artifacts. Must be runnable as `python scripts/train_two_tower.py --data-dir data/processed/beauty --output-dir models/artifacts/beauty_tower`.

#### Hyperparameters for Beauty dataset
```python
EMBEDDING_DIM   = 128
HIDDEN_DIM      = 256
BATCH_SIZE      = 1024
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-2
TEMPERATURE     = 0.07
EPOCHS          = 50        # with early stopping
EARLY_STOP_K    = 5         # patience epochs
RECALL_K        = 50        # validation metric
N_VAL_USERS     = 2000      # sample for fast validation
```

#### Expected results after proper training on Beauty 5-core
```
Validation Recall@50: 0.65 – 0.78
FAISS index size: ~6MB (12K items × 128 dims)
Training time: ~20 min on GPU (CUDA or MPS), ~2h on CPU
```

---

## 4. FAISS Index — Wire Into Training Pipeline

### File: `retrieval/build_faiss_index.py`
This file exists. The following must be changed:

1. After `scripts/train_two_tower.py` completes, it must automatically call `build_faiss_index` with the exported `item_embeddings.npy`.
2. Index type: **HNSW** with `M=32, ef_construction=200`. This is already correct in the PRD.
3. Save index to: `models/artifacts/beauty_tower/item_index.faiss`.
4. Save a latency benchmark: query 1000 random user embeddings, measure P50 and P99 latency. Write results to `models/artifacts/beauty_tower/faiss_latency.json`: `{"p50_ms": ..., "p99_ms": ..., "n_items": ..., "dim": 128}`.
5. Target: P99 < 12ms on CPU. On GPU/MPS this will be faster.

### File: `retrieval/ann_search.py`
Add a `retrieve_top_k(user_embedding: np.ndarray, k: int = 500) -> list[int]` function that:
- Loads the saved FAISS index (lazy-load, cache after first load)
- Returns a list of item_ids (encoded ints) sorted by similarity score descending

---

## 5. SASRec Model — Full Training Loop

### File: `models/sasrec.py`
The model architecture is acceptable. The following must be fixed or added:

#### Architecture (keep as-is from PRD)
```
SASRec:
  item_embedding: Embedding(n_items + 1, 64)   # 0 = padding
  positional_embedding: Embedding(max_seq_len=50, 64)
  transformer_layers: 2 ×
    MultiHeadAttention(num_heads=2, embed_dim=64) with causal mask
    FeedForward: Linear(64→256) → GELU → Dropout(0.2) → Linear(256→64)
    LayerNorm + residual connection
  output: dot product of last-position embedding vs candidate item embeddings
```

#### What must be added to the training loop

**Training objective**: Next-item prediction with sampled softmax.
For each (user, sequence) pair:
- Input: sequence of last min(len, 50) item_ids, left-padded with 0 if shorter
- Positive: the item at position t+1 (teacher forcing over all positions)
- Negatives: sample 1 random negative item per position (or use in-batch negatives)
- Loss: Binary cross-entropy: `BCE(dot(seq_emb_t, pos_emb), 1) + BCE(dot(seq_emb_t, neg_emb), 0)`

**Device**: Use `get_device()` from `utils/device.py`. Move all tensors to device.

**Causal mask**: The causal attention mask must be a proper upper-triangular boolean mask so that position `t` cannot attend to positions `> t`. This is not optional — it is a correctness requirement.

**MPS note**: `torch.nn.MultiheadAttention` with `attn_mask` on MPS may require casting the mask to `torch.float32` (not `torch.bool`). Handle this explicitly:
```python
if device.type == "mps":
    causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))
```

**Training hyperparameters for Beauty dataset**:
```python
EMBEDDING_DIM   = 64
MAX_SEQ_LEN     = 50
NUM_HEADS       = 2
NUM_LAYERS      = 2
FFN_DIM         = 256
DROPOUT         = 0.2
BATCH_SIZE      = 256
LEARNING_RATE   = 1e-3
EPOCHS          = 200       # with early stopping
EARLY_STOP_K    = 10        # patience epochs
NEG_SAMPLES     = 1         # per-position in-sequence negatives
EVAL_K_LIST     = [10, 20]  # NDCG and Hit reported at both
N_VAL_USERS     = 2000      # sample for fast validation each epoch
```

**Checkpoint saving**: Save best checkpoint (by validation NDCG@10) to `models/artifacts/beauty_sasrec/best_checkpoint.pt`. Save: `{'model_state_dict': ..., 'ndcg_at_10': ..., 'hit_at_10': ..., 'epoch': ..., 'n_items': ..., 'max_seq_len': 50}`.

**Training script**: Create `scripts/train_sasrec.py` that:
- Loads `data/processed/beauty/train_sequences.parquet` and `val_labels.parquet`
- Trains SASRec with the above hyperparameters
- Evaluates every 10 epochs on a 2000-user validation sample using sampled evaluation (100 popularity-sampled negatives) from `evaluation/offline_eval.py`
- Saves best checkpoint and logs to MLflow
- Must be runnable as `python scripts/train_sasrec.py --data-dir data/processed/beauty --output-dir models/artifacts/beauty_sasrec`

**Expected results after full training on Beauty 5-core**:
```
NDCG@10:  0.055 – 0.075  (sampled, 100 popularity-sampled negatives)
Hit@10:   0.11  – 0.15
Hit@20:   0.17  – 0.22
Training time: ~1.5 – 2.5h on GPU, ~12h on CPU
```
If running on CPU only, reduce EPOCHS to 100 and N_VAL_USERS to 500.

---

## 6. Cold-Start Module — Wire Up Properly

### File: `models/cold_start.py`

This file must implement the following:

#### BGE item embedding generation
```python
def generate_bge_item_embeddings(
    item_meta_path: str,          # path to item_meta.parquet
    output_path: str,             # where to save .npy file
    model_name: str = "BAAI/bge-small-en-v1.5",  # use 'small' for speed
    batch_size: int = 256,
    device: torch.device = None,
) -> None:
    """
    For each item, concatenate title and description (truncate to 128 tokens).
    Run through BGE sentence encoder.
    Save embeddings as numpy array: shape (n_items, 384) for bge-small.
    Also build and save a FAISS flat IP index for cold-start retrieval.
    Save to: models/artifacts/beauty_coldstart/bge_item_embeddings.npy
             models/artifacts/beauty_coldstart/bge_item_index.faiss
    """
```

**Use `BAAI/bge-small-en-v1.5` (384-dim, ~130MB)** — not the large model. On Beauty with 12K items, embedding generation takes ~5 min on GPU. This is fast enough.

#### Cold-start routing logic
```python
def get_cold_start_recommendations(
    interaction_history: list[int],    # encoded item_ids, may be empty
    n_items: int,
    item_popularity: dict[int, int],   # item_id → interaction count
    bge_item_embeddings: np.ndarray,
    bge_faiss_index: faiss.Index,
    k: int = 500,
) -> list[int]:
    """
    IF len(interaction_history) == 0:
        Return top-k items by log(popularity), sampled with recency weight.
    ELIF 1 <= len(interaction_history) < 5:
        Encode each interacted item using pre-computed bge_item_embeddings[item_id].
        Average embeddings → query bge_faiss_index for top-k similar items.
        Return top-k item_ids.
    """
```

---

## 7. End-to-End Pipeline — Wire Everything Together

### File: `scripts/run_full_pipeline.py`

This is the most important new script. It must:

1. Load all artifacts:
   - Two-tower model checkpoint from `models/artifacts/beauty_tower/best_checkpoint.pt`
   - FAISS HNSW index from `models/artifacts/beauty_tower/item_index.faiss`
   - SASRec model checkpoint from `models/artifacts/beauty_sasrec/best_checkpoint.pt`
   - BGE reranker (lazy-load `BAAI/bge-reranker-base`)
   - BGE cold-start index from `models/artifacts/beauty_coldstart/bge_item_index.faiss`
   - Test labels from `data/processed/beauty/test_labels.parquet`
   - Train sequences from `data/processed/beauty/train_sequences.parquet`

2. Run the full 6-experiment ablation study using `evaluation/ablation.py`:

```
Experiment A: Popularity baseline (rank items by global popularity)
Experiment B: Two-tower only (FAISS retrieval → rank by cosine similarity, no reranking)
Experiment C: Two-tower + SASRec (FAISS top-500 → SASRec re-score → top-10)
Experiment D: Two-tower + SASRec + BGE reranker (full warm pipeline)
Experiment E: Cold-start only (BGE content-based, for users with ≤5 interactions)
Experiment F: Full system with routing (warm → Experiment D, cold → Experiment E)
```

3. For each experiment, evaluate using `evaluation/offline_eval.py`:
   - Full validation set: NDCG@10, Hit@10, Hit@20, MRR
   - Sampled (100 popularity-sampled negatives): NDCG@10, Hit@10
   - Cold-start users only: NDCG@10, Hit@10
   - Warm users only: NDCG@10, Hit@10

4. Log all results to MLflow via `evaluation/mlflow_logging.py`. Each experiment = one MLflow run. Tag with `dataset=beauty_5core`.

5. Print a final summary table to stdout in this exact format:
```
==============================================================
  SeqRec Ablation Results — Amazon Beauty 5-core
==============================================================
Experiment                   NDCG@10   Hit@10   Hit@20    MRR
--------------------------------------------------------------
A: Popularity baseline        0.XXX     0.XXX    0.XXX    0.XXX
B: Two-tower only             0.XXX     0.XXX    0.XXX    0.XXX
C: + SASRec                   0.XXX     0.XXX    0.XXX    0.XXX
D: + BGE reranker (full)      0.XXX     0.XXX    0.XXX    0.XXX
E: Cold-start (content-based) 0.XXX     0.XXX    0.XXX    0.XXX
F: Full system w/ routing     0.XXX     0.XXX    0.XXX    0.XXX
--------------------------------------------------------------
Cold-start users (≤5 int)
  D: + BGE reranker            0.XXX     0.XXX
  E: Cold-start only           0.XXX     0.XXX
Warm users (≥10 int)
  D: + BGE reranker            0.XXX     0.XXX
==============================================================
FAISS P99 latency: XX.Xms   SASRec inference P99: XX.Xms
Full pipeline P99: XX.Xms
```

6. Save the table as `evaluation/results/beauty/RESULTS.md`.

---

## 8. Serving Pipeline — Wire Real Artifacts

### File: `serving/app/pipeline.py`

This file currently stubs everything to return 503. Once real artifacts exist, it must:

1. On startup (`lifespan` or `startup_event`), load:
   - Two-tower user tower model
   - FAISS HNSW index (item embeddings)
   - SASRec model
   - BGE reranker (optional, flag: `ENABLE_RERANKER=true`)
   - BGE cold-start index (optional, flag: `ENABLE_COLD_START=true`)
   - Item metadata from `data/processed/beauty/item_meta.parquet`

2. `recommend(user_id, n_items=10, context=None)` must:
   ```
   t0 = now()
   user_features = redis_feature_store.get_user_features(user_id)  → 1-3ms
   IF cold_start:
       candidates = cold_start_module.get_recs(history, k=500)
   ELSE:
       user_emb = two_tower.encode_user(user_id)                   → <1ms
       candidates = faiss_index.retrieve_top_k(user_emb, k=500)   → <12ms
   t_retrieval = now() - t0

   t1 = now()
   scored = sasrec.score_candidates(user_history, candidates)      → <20ms
   top50 = sorted(candidates, key=scored)[:50]
   t_sasrec = now() - t1

   IF ENABLE_RERANKER:
       t2 = now()
       top10 = bge_reranker.rerank(user_history_text, top50)       → <100ms
       t_reranker = now() - t2
   ELSE:
       top10 = top50[:10]
       t_reranker = 0

   return {
       "recommendations": top10,
       "cold_start": cold_start,
       "latency_ms": {
           "feature_fetch": t_feature,
           "retrieval": t_retrieval,
           "rescoring": t_sasrec,
           "reranking": t_reranker,
           "total": now() - t0,
       }
   }
   ```

3. All latency measurements must also be emitted as Prometheus histograms via `serving/app/metrics.py`.

---

## 9. Feature Store Init — Populate Redis With Trained Artifacts

### File: `data/feature_store_init.py`

This script must run after training completes. It:

1. Loads all trained item embeddings from `models/artifacts/beauty_tower/item_embeddings.npy`
2. Loads item metadata from `data/processed/beauty/item_meta.parquet`
3. For each item, writes to Redis:
   ```
   item:{item_id} → {
       title: str,
       category: str,
       description: str (first 200 chars),
       popularity_score: float,
       item_embedding: base64(float32 numpy array),
   }
   ```
4. Loads train sequences from `data/processed/beauty/train_sequences.parquet`
5. For each user, writes to Redis:
   ```
   user:{user_id} → {
       interaction_history: JSON list of last 50 item_ids,
       interaction_count: int,
       cold_start_flag: "1" if count < 5 else "0",
       last_seen_ts: str(max timestamp),
   }
   ```
6. Prints summary: `Written X users, Y items to Redis in Z seconds.`

Must be runnable as: `python data/feature_store_init.py --data-dir data/processed/beauty --artifact-dir models/artifacts`

---

## 10. Unified Run Script — One Command to Do Everything

### File: `scripts/run_all.sh`

Create a shell script that runs the full pipeline in order:

```bash
#!/bin/bash
set -e

echo "=== Step 1: Download and preprocess dataset ==="
python data/download_amazon.py --output-dir data/raw/beauty
python data/preprocess.py --input-dir data/raw/beauty --output-dir data/processed/beauty

echo "=== Step 2: Train Two-Tower ==="
python scripts/train_two_tower.py \
  --data-dir data/processed/beauty \
  --output-dir models/artifacts/beauty_tower

echo "=== Step 3: Build FAISS Index ==="
python retrieval/build_faiss_index.py \
  --embeddings models/artifacts/beauty_tower/item_embeddings.npy \
  --output models/artifacts/beauty_tower/item_index.faiss

echo "=== Step 4: Train SASRec ==="
python scripts/train_sasrec.py \
  --data-dir data/processed/beauty \
  --output-dir models/artifacts/beauty_sasrec

echo "=== Step 5: Generate BGE Cold-Start Embeddings ==="
python models/cold_start.py \
  --meta data/processed/beauty/item_meta.parquet \
  --output-dir models/artifacts/beauty_coldstart

echo "=== Step 6: Run Full Ablation and Evaluation ==="
python scripts/run_full_pipeline.py \
  --data-dir data/processed/beauty \
  --artifact-dir models/artifacts

echo "=== Step 7: Populate Redis Feature Store ==="
python data/feature_store_init.py \
  --data-dir data/processed/beauty \
  --artifact-dir models/artifacts

echo "=== Done. Results in evaluation/results/beauty/RESULTS.md ==="
```

---

## 11. Updated `requirements.txt`

Replace the current requirements.txt with the following (add missing packages):

```
# Core ML
torch>=2.1.0
numpy>=1.24.0
scipy>=1.10.0

# Recommendation / retrieval
faiss-cpu>=1.7.4       # use faiss-gpu if CUDA is available and installed separately

# NLP / embeddings
transformers>=4.38.0
sentence-transformers>=2.6.0
tokenizers>=0.15.0

# Data
pandas>=2.0.0
pyarrow>=14.0.0
requests>=2.31.0

# Feature store
redis>=5.0.0

# Serving
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0

# Experiment tracking
mlflow>=2.10.0

# Monitoring
prometheus-client>=0.19.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0

# Utilities
tqdm>=4.66.0
python-dotenv>=1.0.0
```

---

## 12. Updated `README.md` Results Section

After training, the README must contain this exact table (fill in real numbers after running):

```markdown
## Results — Amazon Beauty 5-core

All evaluations use leave-one-out protocol with 100 popularity-sampled negatives.

| Model | NDCG@10 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| Popularity baseline | — | — | — | — |
| Two-tower (FAISS ANN) | — | — | — | — |
| + SASRec re-scoring | — | — | — | — |
| + BGE cross-encoder reranker | — | — | — | — |
| Cold-start (BGE content, <5 int) | — | — | — | — |
| **Full system w/ routing** | — | — | — | — |

### Reference benchmarks (from published papers on Beauty 5-core)
| Model (published) | NDCG@10 | Hit@10 |
|---|---:|---:|
| BPR-MF | 0.031 | 0.070 |
| GRU4Rec | 0.044 | 0.098 |
| SASRec (original paper) | 0.063 | 0.132 |
| BERT4Rec | 0.069 | 0.145 |

### System performance
| Metric | Value |
|---|---|
| FAISS ANN P50 latency | Xms |
| FAISS ANN P99 latency | Xms |
| SASRec inference P99 | Xms |
| Full pipeline P50 | Xms |
| Full pipeline P99 | Xms |
| Sustained QPS (1 container) | X req/s |
```

---

## 13. Resume Bullets — Fill These In After Running

The following bullets are templates. Fill in real numbers from `evaluation/results/beauty/RESULTS.md`.

**Primary bullet (paste this on resume):**
> Built a 3-stage production recommendation pipeline (Two-Tower retrieval → SASRec sequential re-scoring → BGE cross-encoder reranking) over Amazon Beauty catalog (12K items, 198K interactions); full-system NDCG@10 of [D.RESULT] vs [B.RESULT] for retrieval-only, served via FastAPI + Docker with per-stage latency breakdown and P99 < 150ms.

**Cold-start bullet:**
> Designed explicit cold-start path using BGE-small content embeddings and popularity-weighted fallback for users with <5 interactions; achieved Hit@10 of [E.RESULT] vs near-zero for collaborative filtering, addressing the #1 open challenge in production recommendation systems.

**Infrastructure bullet:**
> Indexed [N_ITEMS] item embeddings in FAISS HNSW (M=32, ef=200); achieved Recall@500 of [RECALL] with P99 ANN latency of [P99]ms; backed by Redis feature store with user interaction sequences serving at P50 < 3ms.

**Ablation bullet:**
> Conducted 6-experiment ablation study (popularity → two-tower → + SASRec → + BGE reranker → cold-start routing) logged to MLflow; each stage's NDCG@10 contribution quantified, with SASRec accounting for [X]% relative gain and reranker for [Y]% additional gain over retrieval baseline.

---

## 14. Execution Order for the Agent

Execute tasks in this exact order. Do not skip steps. Do not proceed to step N+1 until step N passes its acceptance criteria.

### Step 1 — Create `utils/device.py`
- Acceptance: `python -c "from utils.device import get_device; print(get_device())"` prints without error

### Step 2 — Create `data/download_amazon.py`
- Acceptance: Script downloads `All_Beauty.csv` and `meta_All_Beauty.json.gz` to `data/raw/beauty/`
- Note: If downloading is slow, the agent may skip to Step 3 and use a tiny synthetic dataset for smoke testing

### Step 3 — Create `data/preprocess.py`
- Acceptance: Running on full Beauty data produces all 8 output files in `data/processed/beauty/` with the expected stats

### Step 4 — Fix `models/two_tower.py` and create `scripts/train_two_tower.py`
- Acceptance: Smoke test forward pass with synthetic data; loss decreases over 3 epochs on synthetic data; device detection works

### Step 5 — Fix `retrieval/build_faiss_index.py` and `retrieval/ann_search.py`
- Acceptance: Index built from random embeddings (dim=128, 1000 items); top-10 query returns correct shape

### Step 6 — Fix `models/sasrec.py` and create `scripts/train_sasrec.py`
- Acceptance: Forward pass with causal mask; loss decreases on synthetic sequences; NDCG computation works

### Step 7 — Fix `models/cold_start.py`
- Acceptance: Given 0 interactions, returns popularity recommendations; given 3 interactions, returns content-based recommendations; no crash

### Step 8 — Fix `models/reranker.py`
- Acceptance: Given 50 (user_text, item_text) pairs, returns 50 scores in correct order; handles BGE model lazy-load

### Step 9 — Create `scripts/run_full_pipeline.py`
- Acceptance: Runs on synthetic data without crash; prints correctly structured results table

### Step 10 — Fix `serving/app/pipeline.py`
- Acceptance: `POST /recommend` returns a valid JSON response with `latency_ms` breakdown when artifacts exist; returns a helpful 503 with a specific error message when artifacts are missing

### Step 11 — Create `data/feature_store_init.py`
- Acceptance: Script writes user and item features to Redis; test by reading back 3 users and 3 items

### Step 12 — Create `scripts/run_all.sh`
- Acceptance: Script is executable and all paths match actual file locations

### Step 13 — Update `requirements.txt` and `README.md`
- Acceptance: `pip install -r requirements.txt` in a fresh venv installs without conflict errors

### Step 14 — Full GPU training run (human step)
- **This step is performed by the human, not the agent.**
- Run: `bash scripts/run_all.sh` on a machine with CUDA or MPS
- Expected total time: 3–4 hours on GPU (CUDA), 4–5 hours on MPS, 12–16 hours on CPU
- After this step, fill in the results table in `README.md` and the resume bullets

---

## 15. What the Agent Must NOT Do

- Do not change `evaluation/offline_eval.py` — it is correct and tested
- Do not change `evaluation/ablation.py` — it is correct and tested
- Do not change `feature_store/redis_client.py` or `feature_store/schemas.py`
- Do not change `docker-compose.yml` or `Dockerfile`
- Do not add GNN models, distributed training, or PPO/RL
- Do not use `float64` tensors anywhere (breaks MPS)
- Do not hardcode device strings (`"cpu"`, `"cuda"`, `"mps"`)
- Do not download the full Amazon Reviews all-categories dataset
- Do not add a frontend/UI
- Do not replace FAISS with another ANN library
- Do not replace FastAPI with another serving framework
- Do not change the repository structure substantially

---

## 16. Sanity Check — How to Know You Are Done

The project is complete when all of the following are true:

1. `python -c "from utils.device import get_device; print(get_device())"` runs without error
2. `data/processed/beauty/dataset_stats.json` exists and shows n_users ~22K, n_items ~12K
3. `models/artifacts/beauty_tower/best_checkpoint.pt` exists and `validation_recall_at_k` > 0.60
4. `models/artifacts/beauty_tower/item_index.faiss` exists and `faiss_latency.json` shows P99 < 30ms
5. `models/artifacts/beauty_sasrec/best_checkpoint.pt` exists and `ndcg_at_10` > 0.050
6. `evaluation/results/beauty/RESULTS.md` exists with all 6 ablation experiments filled in with real numbers
7. `docker-compose up` starts all 4 services without error
8. `curl -X POST http://localhost:8000/recommend -d '{"user_id": "1", "n_items": 10}'` returns a valid JSON response with a `latency_ms` object
9. MLflow UI at `http://localhost:5000` shows 6 experiment runs with logged metrics
10. `README.md` contains a filled-in results table with real numbers comparable to published SASRec (NDCG@10 within 25% of 0.063)