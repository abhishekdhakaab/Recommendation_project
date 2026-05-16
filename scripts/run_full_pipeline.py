"""Run 6 ablation experiments on Amazon Beauty and report results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.ablation import AblationExperiment, AblationResult, run_ablation_study
from evaluation.offline_eval import evaluate_ranking_batch
from retrieval.build_faiss_index import load_index
from retrieval.ann_search import search_item_ids
from utils.device import get_device

MAX_SEQ_LEN = 50
TOP_K = 10
FAISS_CANDIDATES = 500


def _pad_sequence(seq: list[int], max_len: int) -> list[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def _load_artifact(path: Path, label: str, required: bool = True):
    if not path.exists():
        if required:
            print(f"[error] Missing required artifact: {path}")
            print(f"  Please run the training pipeline first to generate: {label}")
            sys.exit(1)
        else:
            print(f"[warn] Optional artifact not found (skipping): {path}")
            return None
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ablation pipeline on Beauty dataset.")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--artifact-dir", required=True, help="Model artifacts root directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    device = get_device()
    print(f"Using device: {device}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required") from exc

    # --- Load dataset stats ---
    stats_path = data_dir / "dataset_stats.json"
    if not stats_path.exists():
        print(f"[error] dataset_stats.json not found at {stats_path}")
        sys.exit(1)
    with stats_path.open() as f:
        stats = json.load(f)
    n_items: int = stats["n_items"]

    # --- Load test labels ---
    test_path = _load_artifact(data_dir / "test_labels.parquet", "test_labels.parquet")
    df_test = pd.read_parquet(test_path)
    relevant_by_user: dict[int, list[int]] = {}
    for _, row in df_test.iterrows():
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        relevant_by_user[uid] = [iid]
    test_users = list(relevant_by_user.keys())

    # --- Load train sequences ---
    train_path = _load_artifact(data_dir / "train_sequences.parquet", "train_sequences.parquet")
    df_train_seq = pd.read_parquet(train_path)
    train_sequences: dict[int, list[int]] = {}
    for _, row in df_train_seq.iterrows():
        uid = int(row["user_id"])
        train_sequences[uid] = [int(x) for x in row["sequence"]]

    # --- Build popularity counts ---
    popularity_counts: Counter = Counter()
    for seq in train_sequences.values():
        for iid in seq:
            popularity_counts[iid] += 1
    # Sort items by popularity descending
    popular_items = [iid for iid, _ in popularity_counts.most_common()]

    # --- Load Two-Tower artifacts ---
    tower_ckpt_path = _load_artifact(
        artifact_dir / "beauty_tower" / "best_checkpoint.pt",
        "beauty_tower/best_checkpoint.pt",
    )
    faiss_path = _load_artifact(
        artifact_dir / "beauty_tower" / "item_index.faiss",
        "beauty_tower/item_index.faiss",
    )
    embeddings_path = _load_artifact(
        artifact_dir / "beauty_tower" / "item_embeddings.npy",
        "beauty_tower/item_embeddings.npy",
    )

    from models.two_tower import ItemTower, UserTower

    tower_ckpt = torch.load(tower_ckpt_path, map_location=device)
    tower_n_users: int = tower_ckpt["n_users"]
    tower_n_items: int = tower_ckpt["n_items"]

    user_tower = UserTower(n_users=tower_n_users).to(device)
    item_tower = ItemTower(n_items=tower_n_items + 1).to(device)
    user_tower.load_state_dict(tower_ckpt["model_state_dict"]["user_tower"])
    item_tower.load_state_dict(tower_ckpt["model_state_dict"]["item_tower"])
    user_tower.eval()
    item_tower.eval()

    # Load FAISS index
    faiss_index = load_index(faiss_path)

    # Load item embeddings (shape: n_items, 128)
    item_embeddings = np.load(str(embeddings_path))  # shape (n_items, 128)

    # Helper: encode user history as mean of item embeddings
    def encode_user_history(seq: list[int]) -> np.ndarray:
        if not seq:
            return np.zeros(128, dtype=np.float32)
        embs = item_embeddings[[i - 1 for i in seq if 1 <= i <= tower_n_items]]
        if embs.shape[0] == 0:
            return np.zeros(128, dtype=np.float32)
        mean_emb = embs.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        return mean_emb / max(norm, 1e-12)

    # --- Load SASRec artifacts ---
    sasrec_ckpt_path = _load_artifact(
        artifact_dir / "beauty_sasrec" / "best_checkpoint.pt",
        "beauty_sasrec/best_checkpoint.pt",
    )
    from models.sasrec import SASRec

    sasrec_ckpt = torch.load(sasrec_ckpt_path, map_location=device)
    sasrec_n_items: int = sasrec_ckpt["n_items"]
    sasrec_max_seq_len: int = sasrec_ckpt.get("max_seq_len", MAX_SEQ_LEN)
    sasrec_model = SASRec(
        n_items=sasrec_n_items,
        max_seq_len=sasrec_max_seq_len,
        embedding_dim=64,
        num_heads=2,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.2,
    ).to(device)
    sasrec_model.load_state_dict(sasrec_ckpt["model_state_dict"])
    sasrec_model.eval()

    # --- Load BGE cold-start embeddings (optional) ---
    bge_path_candidate = artifact_dir / "beauty_coldstart" / "bge_item_embeddings.npy"
    bge_path = _load_artifact(bge_path_candidate, "beauty_coldstart/bge_item_embeddings.npy", required=False)
    bge_embeddings: np.ndarray | None = None
    if bge_path is not None:
        bge_embeddings = np.load(str(bge_path))
        print(f"Loaded BGE embeddings: {bge_embeddings.shape}")

    # --- Helper: compute user history proxy embedding (for BGE/content-based) ---
    def bge_user_embedding(seq: list[int]) -> np.ndarray | None:
        if bge_embeddings is None or not seq:
            return None
        valid = [i for i in seq if 0 < i < bge_embeddings.shape[0]]
        if not valid:
            return None
        embs = bge_embeddings[valid]
        mean_emb = embs.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        return mean_emb / max(norm, 1e-12)

    # --- Build item meta for item_meta.parquet ---
    item_meta_path = data_dir / "item_meta.parquet"
    item_train_history: dict[int, set[int]] = {
        uid: set(seq) for uid, seq in train_sequences.items()
    }

    def exclude_seen(user_id: int, candidates: list[int]) -> list[int]:
        seen = item_train_history.get(user_id, set())
        return [i for i in candidates if i not in seen]

    # -------------------------------------------------------------------
    # Experiment A: Popularity baseline
    # -------------------------------------------------------------------
    def popularity_recommend(user_id: int) -> list[int]:
        seen = item_train_history.get(user_id, set())
        result: list[int] = []
        for iid in popular_items:
            if iid not in seen:
                result.append(iid)
            if len(result) >= TOP_K:
                break
        return result

    # -------------------------------------------------------------------
    # Experiment B: Two-tower only (FAISS ANN, top-10 filtered)
    # -------------------------------------------------------------------
    def two_tower_recommend(user_id: int) -> list[int]:
        seq = train_sequences.get(user_id, [])
        user_emb = encode_user_history(seq)
        candidates = search_item_ids(faiss_index, user_emb.reshape(1, -1), top_k=TOP_K + len(seq) + 10)
        if candidates:
            return exclude_seen(user_id, candidates[0])[:TOP_K]
        return []

    # -------------------------------------------------------------------
    # Experiment C: Two-tower + SASRec re-score
    # -------------------------------------------------------------------
    def two_tower_sasrec_recommend(user_id: int) -> list[int]:
        seq = train_sequences.get(user_id, [])
        user_emb = encode_user_history(seq)
        candidates_raw = search_item_ids(
            faiss_index, user_emb.reshape(1, -1), top_k=FAISS_CANDIDATES + len(seq) + 10
        )
        candidates = exclude_seen(user_id, candidates_raw[0] if candidates_raw else [])[:FAISS_CANDIDATES]
        if not candidates:
            return []

        padded = _pad_sequence(seq, sasrec_max_seq_len)
        seq_t = torch.tensor([padded], dtype=torch.long, device=device)
        # Score in chunks to avoid OOM
        chunk_size = 256
        all_scores: list[float] = []
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            cand_t = torch.tensor([chunk], dtype=torch.long, device=device)
            with torch.no_grad():
                scores = sasrec_model.score_candidates(seq_t, cand_t)[0]
            all_scores.extend(scores.cpu().tolist())

        ranked = sorted(zip(all_scores, candidates), reverse=True)
        return [iid for _, iid in ranked[:TOP_K]]

    # -------------------------------------------------------------------
    # Experiment D: Two-tower + SASRec + BGE reranker (skip if no BGE)
    # -------------------------------------------------------------------
    def two_tower_sasrec_bge_recommend(user_id: int) -> list[int]:
        # Use C's top-50, then apply BGE reranking by cosine similarity
        top50 = two_tower_sasrec_recommend_top50(user_id)
        if not top50 or bge_embeddings is None:
            return top50[:TOP_K]

        user_bge = bge_user_embedding(train_sequences.get(user_id, []))
        if user_bge is None:
            return top50[:TOP_K]

        # Score by BGE cosine sim
        valid_candidates = [i for i in top50 if 0 < i < bge_embeddings.shape[0]]
        if not valid_candidates:
            return top50[:TOP_K]
        item_embs = bge_embeddings[valid_candidates]  # (n, dim)
        scores = item_embs @ user_bge
        ranked = sorted(zip(scores.tolist(), valid_candidates), reverse=True)
        return [iid for _, iid in ranked[:TOP_K]]

    def two_tower_sasrec_recommend_top50(user_id: int) -> list[int]:
        """Like C but return top-50 for downstream reranking."""
        seq = train_sequences.get(user_id, [])
        user_emb = encode_user_history(seq)
        candidates_raw = search_item_ids(
            faiss_index, user_emb.reshape(1, -1), top_k=FAISS_CANDIDATES + len(seq) + 10
        )
        candidates = exclude_seen(user_id, candidates_raw[0] if candidates_raw else [])[:FAISS_CANDIDATES]
        if not candidates:
            return []

        padded = _pad_sequence(seq, sasrec_max_seq_len)
        seq_t = torch.tensor([padded], dtype=torch.long, device=device)
        chunk_size = 256
        all_scores: list[float] = []
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            cand_t = torch.tensor([chunk], dtype=torch.long, device=device)
            with torch.no_grad():
                scores = sasrec_model.score_candidates(seq_t, cand_t)[0]
            all_scores.extend(scores.cpu().tolist())

        ranked = sorted(zip(all_scores, candidates), reverse=True)
        return [iid for _, iid in ranked[:50]]

    # -------------------------------------------------------------------
    # Experiment E: Cold-start BGE content-based for all users
    # -------------------------------------------------------------------
    def bge_content_recommend(user_id: int) -> list[int]:
        if bge_embeddings is None:
            return popularity_recommend(user_id)
        seq = train_sequences.get(user_id, [])
        user_bge = bge_user_embedding(seq)
        if user_bge is None:
            return popularity_recommend(user_id)
        seen = item_train_history.get(user_id, set())
        # Score all items
        item_embs = bge_embeddings[1:]  # 1-indexed rows 1..max_item_id
        scores = item_embs @ user_bge
        sorted_indices = np.argsort(-scores)
        result: list[int] = []
        for idx in sorted_indices:
            iid = int(idx) + 1  # 1-indexed
            if iid not in seen and 1 <= iid <= n_items:
                result.append(iid)
            if len(result) >= TOP_K:
                break
        return result

    # -------------------------------------------------------------------
    # Experiment F: Full system routing (cold ≤5 items → E, else → C or D)
    # -------------------------------------------------------------------
    def full_system_recommend(user_id: int) -> list[int]:
        seq = train_sequences.get(user_id, [])
        if len(seq) <= 5:
            return bge_content_recommend(user_id)
        if bge_embeddings is not None:
            return two_tower_sasrec_bge_recommend(user_id)
        return two_tower_sasrec_recommend(user_id)

    # --- Define experiments ---
    experiments = [
        AblationExperiment(name="A: Popularity Baseline", recommend=popularity_recommend),
        AblationExperiment(name="B: Two-Tower (FAISS ANN)", recommend=two_tower_recommend),
        AblationExperiment(name="C: Two-Tower + SASRec", recommend=two_tower_sasrec_recommend),
    ]
    if bge_embeddings is not None:
        experiments.append(
            AblationExperiment(name="D: Two-Tower + SASRec + BGE Reranker", recommend=two_tower_sasrec_bge_recommend)
        )
        experiments.append(
            AblationExperiment(name="E: Cold-Start BGE Content", recommend=bge_content_recommend)
        )
        experiments.append(
            AblationExperiment(name="F: Full System w/ Routing", recommend=full_system_recommend)
        )
    else:
        print("[info] BGE embeddings not found — skipping D, E, F experiments")

    # --- Run ablation study ---
    print(f"\nRunning {len(experiments)} ablation experiments on {len(test_users)} test users ...\n")
    results: list[AblationResult] = run_ablation_study(
        experiments,
        test_users,
        relevant_by_user,
        k=TOP_K,
    )

    # --- Print results table ---
    header = f"{'Model':<45} {'NDCG@10':>8} {'Hit@10':>8} {'MRR':>8} {'Users':>8}"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)
    for r in results:
        m = r.metrics
        print(
            f"{r.name:<45} {m.ndcg:>8.4f} {m.hit:>8.4f} {m.mrr:>8.4f} {m.users:>8d}"
        )
    print(divider)

    # --- Save to RESULTS.md ---
    results_dir = Path("evaluation/results/beauty")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_md = results_dir / "RESULTS.md"

    lines = [
        "# Ablation Results — Amazon Beauty 5-core\n",
        f"Test users: {len(test_users)}  |  k=10  |  Leave-one-out protocol\n\n",
        "| Model | NDCG@10 | Hit@10 | MRR | Users |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.name} | {m.ndcg:.4f} | {m.hit:.4f} | {m.mrr:.4f} | {m.users} |\n"
        )

    with results_md.open("w") as f:
        f.writelines(lines)
    print(f"\nResults saved to {results_md.resolve()}")


if __name__ == "__main__":
    main()
