"""Evaluate sequence-aware retrieval from item co-occurrence embeddings."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.offline_eval import evaluate_ranking_batch
from retrieval.scoring import blend_with_popularity_prior


def evaluate_sequence_retrieval(
    *,
    processed_dir: str | Path,
    output_path: str | Path,
    max_users: int = 5000,
    n_negatives: int = 100,
    embedding_dim: int = 128,
    window_size: int = 20,
    popularity_weight: float = 0.0,
    seed: int = 7,
) -> dict[str, object]:
    """Evaluate mean-recent-item embedding retrieval with SVD item vectors."""

    rng = random.Random(seed)
    processed = Path(processed_dir)
    stats = json.loads((processed / "stats.json").read_text())
    train = _read_jsonl(processed / "train.jsonl")
    validation = _read_jsonl(processed / "validation.jsonl", limit=max_users)
    popularity = Counter(int(row["item_id"]) for row in train)
    histories: dict[int, list[int]] = defaultdict(list)
    for row in sorted(train, key=lambda value: int(value["timestamp"])):
        histories[int(row["user_id"])].append(int(row["item_id"]))

    item_embeddings = _build_svd_item_embeddings(histories.values(), n_items=stats["items"], dim=embedding_dim)
    popular_pool = [item for item, _ in popularity.most_common()]
    recommendations = {}
    relevant = {}
    for row in validation:
        user_id = int(row["user_id"])
        positive = int(row["item_id"])
        history = histories.get(user_id, [])[-window_size:]
        if not history:
            continue
        negatives = _sample_unique_popular_negatives(popular_pool, positive, n_negatives, rng)
        candidates = [positive] + negatives
        user_vector = item_embeddings[history].mean(axis=0)
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm
        model_scores = (item_embeddings[candidates] @ user_vector).tolist()
        scores = blend_with_popularity_prior(model_scores, candidates, popularity, popularity_weight=popularity_weight)
        recommendations[user_id] = [item for _, item in sorted(zip(scores, candidates, strict=True), reverse=True)]
        relevant[user_id] = {positive}

    summary = {
        "protocol": f"{len(relevant)} users, {n_negatives} popularity-sampled negatives",
        "embedding_dim": embedding_dim,
        "window_size": window_size,
        "popularity_weight": popularity_weight,
        "at_10": evaluate_ranking_batch(recommendations, relevant, k=10).__dict__,
        "at_20": evaluate_ranking_batch(recommendations, relevant, k=20).__dict__,
        "at_50": evaluate_ranking_batch(recommendations, relevant, k=50).__dict__,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _build_svd_item_embeddings(histories, *, n_items: int, dim: int) -> np.ndarray:
    matrix = np.zeros((n_items, n_items), dtype=np.float32)
    for history in histories:
        unique = list(dict.fromkeys(history))
        for item in unique:
            matrix[item, item] += 1.0
        for i, item_i in enumerate(unique):
            for item_j in unique[i + 1 :]:
                matrix[item_i, item_j] += 1.0
                matrix[item_j, item_i] += 1.0
    matrix = np.log1p(matrix)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    actual_dim = min(dim, u.shape[1])
    embeddings = u[:, :actual_dim] * np.sqrt(singular_values[:actual_dim])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)


def _sample_unique_popular_negatives(pool: list[int], positive: int, n_negatives: int, rng: random.Random) -> list[int]:
    negatives = []
    seen = {positive}
    while len(negatives) < n_negatives:
        item = pool[min(int(rng.random() ** 2 * len(pool)), len(pool) - 1)]
        if item not in seen:
            negatives.append(item)
            seen.add(item)
    return negatives


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sequence-aware retrieval.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--popularity-weight", type=float, default=0.0)
    args = parser.parse_args()
    summary = evaluate_sequence_retrieval(
        processed_dir=args.processed_dir,
        output_path=args.output,
        max_users=args.max_users,
        n_negatives=args.n_negatives,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        popularity_weight=args.popularity_weight,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
