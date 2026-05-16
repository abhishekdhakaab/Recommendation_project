"""Evaluate a trained retrieval artifact with fast sampled metrics."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import random
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from evaluation.offline_eval import evaluate_ranking_batch
from models.two_tower import ItemTower, UserTower
from retrieval.scoring import blend_with_popularity_prior


def evaluate_retrieval_artifact(
    *,
    processed_dir: str | Path,
    artifact_dir: str | Path,
    output_path: str | Path,
    max_users: int = 1000,
    n_negatives: int = 100,
    popularity_weight: float = 0.8,
    seed: int = 7,
) -> dict[str, object]:
    processed = Path(processed_dir)
    artifact = Path(artifact_dir)
    stats = json.loads((processed / "stats.json").read_text(encoding="utf-8"))
    train = _read_jsonl(processed / "train.jsonl")
    validation = _read_jsonl(processed / "validation.jsonl", limit=max_users)
    popularity = Counter(int(row["item_id"]) for row in train)
    rng = random.Random(seed)
    popular_pool = [item for item, _ in popularity.most_common()]

    checkpoint = torch.load(artifact / "two_tower.pt", map_location="cpu", weights_only=False)
    user_tower = UserTower(n_users=stats["users"])
    item_tower = ItemTower(n_items=stats["items"])
    user_tower.load_state_dict(checkpoint["user_tower"])
    item_tower.load_state_dict(checkpoint["item_tower"])
    user_tower.eval()
    item_tower.eval()

    raw_recs = {}
    blended_recs = {}
    pop_recs = {}
    relevant = {}
    with torch.inference_mode():
        for row in validation:
            user_id = int(row["user_id"])
            positive = int(row["item_id"])
            negatives = _sample_unique_popular_negatives(popular_pool, positive, n_negatives, rng)
            candidates = [positive] + negatives
            user = torch.tensor([user_id], dtype=torch.long)
            items = torch.tensor(candidates, dtype=torch.long)
            raw_scores = (user_tower(user) @ item_tower(items).T)[0].tolist()
            blended_scores = blend_with_popularity_prior(raw_scores, candidates, popularity, popularity_weight=popularity_weight)
            pop_scores = [float(popularity[item]) for item in candidates]
            raw_recs[user_id] = _rank(candidates, raw_scores)
            blended_recs[user_id] = _rank(candidates, blended_scores)
            pop_recs[user_id] = _rank(candidates, pop_scores)
            relevant[user_id] = {positive}

    summary = {
        "protocol": f"{len(validation)} users, {n_negatives} popularity-sampled negatives",
        "popularity_weight": popularity_weight,
        "popularity": {
            "at_10": evaluate_ranking_batch(pop_recs, relevant, k=10).__dict__,
            "at_20": evaluate_ranking_batch(pop_recs, relevant, k=20).__dict__,
        },
        "two_tower": {
            "at_10": evaluate_ranking_batch(raw_recs, relevant, k=10).__dict__,
            "at_20": evaluate_ranking_batch(raw_recs, relevant, k=20).__dict__,
        },
        "two_tower_plus_popularity_prior": {
            "at_10": evaluate_ranking_batch(blended_recs, relevant, k=10).__dict__,
            "at_20": evaluate_ranking_batch(blended_recs, relevant, k=20).__dict__,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _sample_unique_popular_negatives(pool: list[int], positive: int, n_negatives: int, rng: random.Random) -> list[int]:
    negatives = []
    seen = {positive}
    while len(negatives) < n_negatives:
        item = pool[min(int(rng.random() ** 2 * len(pool)), len(pool) - 1)]
        if item not in seen:
            negatives.append(item)
            seen.add(item)
    return negatives


def _rank(candidates: list[int], scores: list[float]) -> list[int]:
    return [item for _, item in sorted(zip(scores, candidates, strict=True), reverse=True)]


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
    parser = argparse.ArgumentParser(description="Evaluate retrieval artifact with sampled negatives.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--popularity-weight", type=float, default=0.8)
    args = parser.parse_args()
    summary = evaluate_retrieval_artifact(
        processed_dir=args.processed_dir,
        artifact_dir=args.artifact_dir,
        output_path=args.output,
        max_users=args.max_users,
        n_negatives=args.n_negatives,
        popularity_weight=args.popularity_weight,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
