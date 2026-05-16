# SeqRec Implementation Plan

## Phase 0 — Repo foundation

Goal:
Create the repository skeleton, dependency files, configuration, and minimal smoke tests.

Deliverables:
- Repo folders matching PRD
- requirements.txt
- pyproject.toml or setup config
- README skeleton
- Basic test setup
- Small synthetic sample data generator for tests

Acceptance criteria:
- Repo imports work
- Tests run
- No model training required yet

## Phase 1 — Data preprocessing

Goal:
Implement Amazon Reviews preprocessing pipeline.

Deliverables:
- data/download_amazon.py
- data/preprocess.py
- ID encoding
- k-core filtering
- chronological user sequences
- train/val/test split
- metadata table

Acceptance criteria:
- Works on a tiny synthetic dataset
- Outputs documented artifacts
- Logs dataset stats

## Phase 2 — Two-Tower retrieval

Goal:
Implement and train two-tower retrieval model.

Deliverables:
- models/two_tower.py
- in-batch negative loss
- training loop
- validation Recall@K
- checkpoint saving

Acceptance criteria:
- Forward pass smoke test
- Loss decreases on synthetic data
- Recall@K function tested

## Phase 3 — FAISS retrieval

Goal:
Build ANN retrieval over item embeddings.

Deliverables:
- retrieval/build_faiss_index.py
- retrieval/ann_search.py
- HNSW index save/load
- latency benchmark helper

Acceptance criteria:
- Tiny FAISS index can be built, saved, loaded, queried
- Query returns top-k item IDs

## Phase 4 — SASRec

Goal:
Implement sequential re-scoring model.

Deliverables:
- models/sasrec.py
- causal masking
- sampled negative training
- candidate scoring function

Acceptance criteria:
- Forward pass test
- Causal mask test
- Candidate scoring works on top-k candidates

## Phase 5 — Cold-start

Goal:
Implement content-based fallback path.

Deliverables:
- models/cold_start.py
- BGE item embedding wrapper
- BGE FAISS index
- popularity fallback
- cold/warm routing logic

Acceptance criteria:
- 0-interaction user gets popularity recommendations
- 1–4 interaction user gets content-based recommendations
- warm user goes to normal path

## Phase 6 — Reranker

Goal:
Implement BGE cross-encoder reranker first.

Deliverables:
- models/reranker.py
- BGEReranker wrapper
- rerank top-50 to top-10

Acceptance criteria:
- Reranker accepts user context + candidate item text
- Returns sorted top-n items
- Can be disabled for ablation

## Phase 7 — Redis feature store

Goal:
Implement online feature serving.

Deliverables:
- feature_store/redis_client.py
- feature_store/schemas.py
- data/feature_store_init.py

Acceptance criteria:
- User features can be written/read
- Item metadata can be written/read
- Local Redis works through docker-compose

## Phase 8 — FastAPI serving

Goal:
Serve the full recommendation pipeline.

Deliverables:
- serving/app/main.py
- serving/app/pipeline.py
- serving/app/metrics.py
- POST /recommend
- POST /feedback
- GET /metrics

Acceptance criteria:
- API returns PRD-compatible response
- Per-stage latency included
- Endpoint tests pass

## Phase 9 — Evaluation

Goal:
Implement offline evaluation and ablation experiments.

Deliverables:
- evaluation/offline_eval.py
- evaluation/ablation.py
- evaluation/cold_start_eval.py
- evaluation/ab_simulation.py

Acceptance criteria:
- NDCG@K, Hit@K, MRR, Recall@K tested on toy examples
- Ablation framework can run selected stages

## Phase 10 — Docker, monitoring, polish

Goal:
Make the project demo-ready.

Deliverables:
- Dockerfile
- docker-compose.yml
- Prometheus config
- Grafana dashboard JSON
- README quickstart
- Results table template

Acceptance criteria:
- docker-compose starts services
- API is callable
- README explains setup and results