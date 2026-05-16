"""Scalable top-item sequence retrieval evaluation."""

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


def evaluate_top_item_sequence_retrieval(
    *,
    processed_dir: str | Path,
    output_path: str | Path,
    top_items: int = 10000,
    max_users: int = 5000,
    n_negatives: int = 100,
    embedding_dim: int = 64,
    window_size: int = 20,
    popularity_weight: float = 0.0,
    seed: int = 7,
) -> dict[str, object]:
    rng = random.Random(seed)
    processed = Path(processed_dir)
    train = _read_jsonl(processed / "train.jsonl")
    validation = _read_jsonl(processed / "validation.jsonl")
    popularity = Counter(int(row["item_id"]) for row in train)
    top_item_ids = [item for item, _ in popularity.most_common(top_items)]
    top_set = set(top_item_ids)
    local_id = {item_id: index for index, item_id in enumerate(top_item_ids)}

    histories: dict[int, list[int]] = defaultdict(list)
    for row in sorted(train, key=lambda value: int(value["timestamp"])):
        item_id = int(row["item_id"])
        if item_id in top_set:
            histories[int(row["user_id"])].append(item_id)

    embeddings = _build_embeddings(histories.values(), top_item_ids=top_item_ids, local_id=local_id, dim=embedding_dim)
    popular_pool = top_item_ids
    recommendations = {}
    popularity_recommendations = {}
    relevant = {}
    for row in validation:
        if len(relevant) >= max_users:
            break
        user_id = int(row["user_id"])
        positive = int(row["item_id"])
        history = histories.get(user_id, [])[-window_size:]
        if positive not in top_set or not history:
            continue
        negatives = _sample_negatives(popular_pool, positive, n_negatives, rng)
        candidates = [positive] + negatives
        user_vector = embeddings[[local_id[item] for item in history]].mean(axis=0)
        user_vector = user_vector / max(float(np.linalg.norm(user_vector)), 1e-12)
        model_scores = (embeddings[[local_id[item] for item in candidates]] @ user_vector).tolist()
        scores = blend_with_popularity_prior(model_scores, candidates, popularity, popularity_weight=popularity_weight)
        recommendations[user_id] = [item for _, item in sorted(zip(scores, candidates, strict=True), reverse=True)]
        popularity_scores = [popularity[item] for item in candidates]
        popularity_recommendations[user_id] = [
            item for _, item in sorted(zip(popularity_scores, candidates, strict=True), reverse=True)
        ]
        relevant[user_id] = {positive}

    summary = {
        "protocol": f"top {top_items} items, {len(relevant)} users, {n_negatives} popularity-sampled negatives",
        "embedding_dim": embedding_dim,
        "window_size": window_size,
        "popularity_weight": popularity_weight,
        "at_10": evaluate_ranking_batch(recommendations, relevant, k=10).__dict__,
        "at_20": evaluate_ranking_batch(recommendations, relevant, k=20).__dict__,
        "at_50": evaluate_ranking_batch(recommendations, relevant, k=50).__dict__,
        "popularity_baseline": {
            "at_10": evaluate_ranking_batch(popularity_recommendations, relevant, k=10).__dict__,
            "at_20": evaluate_ranking_batch(popularity_recommendations, relevant, k=20).__dict__,
            "at_50": evaluate_ranking_batch(popularity_recommendations, relevant, k=50).__dict__,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _build_embeddings(histories, *, top_item_ids: list[int], local_id: dict[int, int], dim: int) -> np.ndarray:
    n_items = len(top_item_ids)
    matrix = np.zeros((n_items, n_items), dtype=np.float32)
    for history in histories:
        unique = list(dict.fromkeys(item for item in history if item in local_id))[-50:]
        for item in unique:
            matrix[local_id[item], local_id[item]] += 1.0
        for i, item_i in enumerate(unique):
            row = local_id[item_i]
            for item_j in unique[max(0, i - 10) : i] + unique[i + 1 : i + 11]:
                matrix[row, local_id[item_j]] += 1.0
    matrix = np.log1p(matrix)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    actual_dim = min(dim, u.shape[1])
    embeddings = u[:, :actual_dim] * np.sqrt(singular_values[:actual_dim])
    return (embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _sample_negatives(pool: list[int], positive: int, n_negatives: int, rng: random.Random) -> list[int]:
    negatives = []
    seen = {positive}
    while len(negatives) < n_negatives:
        item = pool[min(int(rng.random() ** 2 * len(pool)), len(pool) - 1)]
        if item not in seen:
            negatives.append(item)
            seen.add(item)
    return negatives


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate top-item sequence retrieval.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-items", type=int, default=10000)
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--popularity-weight", type=float, default=0.0)
    args = parser.parse_args()
    summary = evaluate_top_item_sequence_retrieval(
        processed_dir=args.processed_dir,
        output_path=args.output,
        top_items=args.top_items,
        max_users=args.max_users,
        n_negatives=args.n_negatives,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        popularity_weight=args.popularity_weight,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
