# Sports & Outdoors Experiment Results

This file summarizes the experiments run so far for the SeqRec project. The strongest result is currently the sequence-aware retrieval experiment; the raw two-tower model is implemented and tested, but its retrieval quality is still weak on this dataset without more training/tuning.

## Dataset

Processed dataset: `data/processed/sports_and_outdoors`

| Split / Stat | Value |
| --- | ---: |
| Raw interactions scanned | 19,595,170 |
| Raw items | 248,035 |
| Filtered interactions | 2,646,845 |
| Users | 319,204 |
| Items | 74,538 |
| Train interactions | 2,008,437 |
| Validation interactions | 319,204 |
| Test interactions | 319,204 |

Strict k-core on `All_Beauty` collapsed to only 3 users and 2 items, so All Beauty is useful only as a smoke-test artifact in this repo, not as a resume-grade benchmark result.

## Main Results

### Full Validation Popularity Baseline

Protocol: full Sports & Outdoors validation split, global popularity recommendations.

| Metric | Score |
| --- | ---: |
| Hit@10 / Recall@10 | 0.0082 |
| NDCG@10 | 0.0041 |
| Hit@20 / Recall@20 | 0.0138 |
| NDCG@20 | 0.0056 |
| Hit@50 / Recall@50 | 0.0266 |
| NDCG@50 | 0.0081 |

### Two-Tower Training Runs

Protocol: model validation Recall@50 during artifact training.

| Run | Train Interactions | Batch Policy | Validation Users | Recall@50 |
| --- | ---: | --- | ---: | ---: |
| 500k duplicate-positive batches | 500,000 | duplicates allowed | 10,000 | 0.0029 |
| 500k unique-positive batches | 500,000 | unique items per batch | 10,000 | 0.0039 |
| 1M unique-positive batches | 1,000,000 | unique items per batch | 20,000 | 0.0032 |

Interpretation: unique-positive in-batch sampling helped, but the raw two-tower model is not yet competitive. It should stay in the project as a first-stage retrieval component and implementation milestone, not as the headline metric.

### Two-Tower Sampled Evaluation

Protocol: 2,000 users, 100 popularity-sampled negatives.

| Method | Hit@10 | NDCG@10 | Hit@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.1400 | 0.0607 | 0.2940 | 0.0991 |
| Two-Tower | 0.0905 | 0.0424 | 0.1705 | 0.0623 |
| Two-Tower + popularity prior | 0.1360 | 0.0592 | 0.2955 | 0.0991 |

Interpretation: the two-tower embeddings alone underperform popularity; blending mostly recovers the popularity baseline.

### Uniform Sampled Evaluation

Protocol: leave-one-out validation with 100 uniformly sampled negatives per user, 10,000 validation users.

| Method | Hit@10 | NDCG@10 | Hit@20 | NDCG@20 | Hit@50 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.4319 | 0.2580 | 0.5744 | 0.2941 | 0.7756 | 0.3341 |
| Two-Tower 500k unique | 0.1029 | 0.0509 | 0.2008 | 0.0754 | 0.4970 | 0.1332 |

Interpretation: uniform sampled negatives make popularity look much stronger than full-catalog evaluation, because many sampled negatives are easy. Use this protocol only when stated clearly.

### Best Current Result: Sequence-Aware Retrieval

Protocol: top 5,000 item candidate universe, 5,000 validation users, 100 popularity-sampled negatives, sequence co-occurrence/SVD item embeddings, user vector from recent history.

| Method | Hit@10 | NDCG@10 | Hit@20 | NDCG@20 | Hit@50 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity baseline, same protocol | 0.0808 | 0.0341 | 0.1894 | 0.0611 | 0.5118 | 0.1244 |
| Sequence-aware retrieval | 0.4478 | 0.2672 | 0.5846 | 0.3018 | 0.8186 | 0.3481 |

Relative to the same-protocol popularity baseline, sequence-aware retrieval improves Hit@10 by about 5.5x and NDCG@10 by about 7.8x.

## Benchmark Context

A public Amazon Beauty sequential-recommendation leaderboard reports SASRec around Hit@10 0.4854 and NDCG@10 0.3219, with stronger methods above that. That is not an apples-to-apples comparison with this Sports & Outdoors experiment, but it gives useful scale: a clearly stated Hit@10 of 0.4478 / NDCG@10 of 0.2672 under a sampled top-5k protocol is resume-presentable if the protocol is disclosed honestly.

Reference: https://beta.hyper.ai/en/sota/tasks/recommendation-systems/benchmark/recommendation-systems-on-amazon-beauty

## Current Takeaway

The project is strongest as a full recommendation-system build with preprocessing, evaluation, two-tower retrieval, FAISS, SASRec, cold-start handling, reranking wrappers, Redis feature store, FastAPI serving, metrics, and Docker scaffolding. The current headline metric should use the sequence-aware retrieval result, not the raw two-tower run.

Suggested resume wording:

> Built an end-to-end sequential recommendation system on Amazon Sports & Outdoors with preprocessing, retrieval, reranking, Redis feature serving, FastAPI inference, and offline evaluation; achieved Hit@10 0.4478 and NDCG@10 0.2672 on a top-5k sampled sequence-retrieval protocol, a 5.5x Hit@10 lift over same-protocol popularity.

## Remaining Work

- Train a real SASRec run beyond smoke-test scale and evaluate it under the same sampled protocol.
- Align two-tower, sequence retrieval, and popularity under one shared evaluator so every comparison uses the same users and negatives.
- Run `make log-sequence-result` after starting MLflow to persist the headline experiment in the tracking UI.
- Optionally test a stronger item-item baseline such as BPR/LightGCN only if it remains inside PRD scope or the PRD is updated.
