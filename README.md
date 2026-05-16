# SeqRec

SeqRec is a production-style multi-stage recommendation system based on
`RecSys_Project_PRD.md`.

## Architecture

1. Two-tower retrieval for first-stage candidate generation.
2. FAISS ANN search over item embeddings.
3. SASRec sequential re-scoring over recent user history.
4. BGE/LLM reranking for final semantic ranking.
5. Cold-start handling with content embeddings and popularity fallback.
6. Redis feature store for online user and item features.
7. FastAPI serving with per-stage latency reporting.
8. Offline evaluation, ablation, and benchmarking.

## Project Layout

- `data/` - dataset download helper, preprocessing, feature-store initialization, and synthetic test data.
- `models/` - two-tower, SASRec, reranker, and cold-start logic.
- `retrieval/` - FAISS HNSW index build, save/load, ANN search, and latency helper.
- `feature_store/` - Redis client helpers and Pydantic feature schemas.
- `serving/` - FastAPI application, injectable pipeline, metrics, and endpoint tests.
- `evaluation/` - ranking metrics, ablations, cold/warm split evaluation, and A/B simulation.
- `notebooks/` - planned exploratory and training notebooks.
- `mlflow/` - planned MLflow tracking configuration.
- `monitoring/` - planned Prometheus and Grafana configuration.
- `tests/` - project-level tests and smoke checks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Test

```bash
python -m pytest
```

On this macOS environment, Torch/FAISS tests may require:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest
```

The repo also includes a small `Makefile` for repeatable local commands:

```bash
make test
make eval-sequence
make log-sequence-result
make serve-demo
```

## Local API

```bash
uvicorn serving.app.main:app --host 0.0.0.0 --port 8000
```

The default app returns `503` for recommendations until trained model artifacts
and feature-store clients are injected. Endpoint tests use a fake injected
pipeline.

## Docker

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- Redis: `localhost:6379`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`


## Current Sports & Outdoors Artifacts

The larger Amazon Reviews 2023 Sports & Outdoors category has been downloaded and stream-preprocessed with the PRD k-core thresholds.

Processed stats:
- users: 319,204
- items: 74,538
- filtered interactions: 2,646,845
- train/validation/test interactions: 2,008,437 / 319,204 / 319,204

Current smoke artifacts:
- `models/artifacts/sports_and_outdoors_tower_smoke/` — Two-Tower CPU smoke run on 100k interactions
- `models/artifacts/sports_and_outdoors_tower_500k/` — Two-Tower CPU run on 500k interactions for 3 epochs; Recall@50 on 10k validation users: 0.0029
- `models/artifacts/sports_and_outdoors_tower_500k_unique/` — improved Two-Tower CPU run with unique positive items per batch; Recall@50 on 10k validation users: 0.0039
- `models/artifacts/sports_and_outdoors_tower_1m_unique/` — larger CPU run with unique positive batches; Recall@50 on 20k validation users: 0.00315
- `models/artifacts/sports_and_outdoors_sasrec_smoke/` — SASRec CPU smoke run on 10k examples
- `evaluation/results/sports_and_outdoors/popularity_summary.json` — popularity baseline metrics
- `evaluation/results/sports_and_outdoors/top_item_sequence_5k.json` — sequence-aware co-occurrence retrieval over top 5k items; sampled Hit@10 0.4478, Hit@20 0.5846, NDCG@10 0.2672
- `evaluation/results/sports_and_outdoors/RESULTS.md` — consolidated experiment write-up, protocol notes, and resume-ready summary

These smoke runs validate the artifact path. The current strongest metric is the sequence-aware retrieval run: Hit@10 0.4478 and NDCG@10 0.2672 on a top-5k, 100 popularity-sampled-negative protocol, with same-protocol popularity at Hit@10 0.0808 and NDCG@10 0.0341. Longer neural SASRec training remains the main next quality-improvement step.

## Current Status

Implemented:
- preprocessing with k-core filtering, ID encoding, and train/validation/test artifacts
- two-tower retrieval model, in-batch negative loss, training helpers, and checkpoint saving
- FAISS HNSW retrieval with save/load and latency benchmarking
- SASRec sequential re-scoring with sampled BCE training step
- BGE-style cold-start routing and content retrieval wrappers
- BGE cross-encoder reranker wrapper and mock reranker
- Redis feature-store schemas and async helpers
- FastAPI endpoints and injectable multi-stage serving pipeline
- offline metrics, ablation helpers, cold/warm evaluation, replay A/B simulation, and optional MLflow result logging
- Docker Compose, Prometheus config, and starter Grafana dashboard

Not included in tests:
- large Amazon dataset download
- real model training on full data
- BGE/Phi/Llama model downloads
- production-quality Grafana dashboard design

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
