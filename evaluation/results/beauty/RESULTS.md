# Ablation Results — Amazon Beauty 5-core
Test users: 50  |  k=10  |  Leave-one-out protocol

| Model | NDCG@10 | Hit@10 | MRR | Users |
|---|---:|---:|---:|---:|
| A: Popularity Baseline | 0.1908 | 0.4600 | 0.1125 | 50 |
| B: Two-Tower (FAISS ANN) | 0.1522 | 0.3800 | 0.0859 | 50 |
| C: Two-Tower + SASRec | 0.1847 | 0.3800 | 0.1261 | 50 |
