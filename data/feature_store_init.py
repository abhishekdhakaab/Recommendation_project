"""Utilities to initialize the Redis feature store from processed artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
import sys
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from feature_store.redis_client import RedisFeatureStore, create_redis_feature_store
from feature_store.schemas import ItemFeatures, UserFeatures


async def initialize_feature_store(
    store: RedisFeatureStore,
    *,
    users: Iterable[UserFeatures | dict],
    items: Iterable[ItemFeatures | dict],
) -> dict[str, int]:
    """Write user and item feature records into Redis."""

    user_count = 0
    for user in users:
        await store.set_user_features(user if isinstance(user, UserFeatures) else UserFeatures(**user))
        user_count += 1

    item_count = 0
    for item in items:
        await store.set_item_features(item if isinstance(item, ItemFeatures) else ItemFeatures(**item))
        item_count += 1

    return {"users": user_count, "items": item_count}


def load_jsonl(path: str | Path) -> list[dict]:
    """Load newline-delimited JSON records."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="Initialize SeqRec Redis feature store.")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--data-dir", default=None, help="Processed data directory (parquet files)")
    parser.add_argument("--artifact-dir", default=None, help="Model artifacts root directory")
    # Legacy arguments kept for backward compatibility
    parser.add_argument("--users-jsonl", default=None, help="[Legacy] Path to users JSONL file")
    parser.add_argument("--items-jsonl", default=None, help="[Legacy] Path to items JSONL file")
    args = parser.parse_args()

    store = create_redis_feature_store(args.redis_url)

    if args.data_dir is not None:
        # New parquet-based mode
        await _init_from_parquet(store, args.data_dir, args.artifact_dir)
    elif args.users_jsonl is not None and args.items_jsonl is not None:
        # Legacy JSONL mode
        stats = await initialize_feature_store(
            store,
            users=load_jsonl(args.users_jsonl),
            items=load_jsonl(args.items_jsonl),
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
    else:
        parser.error("Provide either --data-dir (and optionally --artifact-dir) or --users-jsonl + --items-jsonl")


async def _init_from_parquet(
    store: RedisFeatureStore,
    data_dir: str | Path,
    artifact_dir: str | Path | None,
) -> None:
    """Load from parquet files and write to Redis."""

    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and numpy are required for parquet mode") from exc

    data_path = Path(data_dir)
    t0 = time.perf_counter()

    # --- Load data ---
    df_train_seq = pd.read_parquet(data_path / "train_sequences.parquet")
    df_item_meta = pd.read_parquet(data_path / "item_meta.parquet")

    # Load item embeddings if available
    item_embeddings = None
    if artifact_dir is not None:
        emb_path = Path(artifact_dir) / "beauty_tower" / "item_embeddings.npy"
        if emb_path.exists():
            item_embeddings = np.load(str(emb_path))
            print(f"Loaded item embeddings: {item_embeddings.shape}")
        else:
            print(f"[warn] Item embeddings not found at {emb_path}, skipping embedding storage")

    # Build popularity counts from train sequences
    from collections import Counter
    popularity_counts: Counter = Counter()
    train_sequences: dict[int, list[int]] = {}
    for _, row in df_train_seq.iterrows():
        uid = int(row["user_id"])
        seq = [int(x) for x in row["sequence"]]
        train_sequences[uid] = seq
        for iid in seq:
            popularity_counts[iid] += 1

    max_pop = max(popularity_counts.values()) if popularity_counts else 1

    # --- Write items ---
    print(f"Writing {len(df_item_meta)} items to Redis ...")
    item_count = 0
    for _, row in df_item_meta.iterrows():
        item_id = int(row["item_id"])
        title = str(row.get("title", "") or "")
        category = str(row.get("category", "") or "")
        description = str(row.get("description", "") or "")[:200]
        pop_score = float(popularity_counts.get(item_id, 0)) / max(max_pop, 1)

        # Item embedding
        emb_list = None
        if item_embeddings is not None:
            idx = item_id - 1  # 1-indexed → 0-indexed in array
            if 0 <= idx < item_embeddings.shape[0]:
                emb_list = item_embeddings[idx].tolist()

        features = ItemFeatures(
            item_id=str(item_id),
            title=title,
            category=category,
            description=description,
            popularity_score=pop_score,
            item_embedding=emb_list,
        )
        await store.set_item_features(features)
        item_count += 1

    # --- Write users ---
    print(f"Writing {len(train_sequences)} users to Redis ...")
    user_count = 0
    for uid, seq in train_sequences.items():
        last_50 = seq[-50:]
        n_interactions = len(seq)
        cold_start = n_interactions <= 5
        last_seen_ts = 0  # We don't persist timestamps here; use 0 as default

        features = UserFeatures(
            user_id=str(uid),
            interaction_history=last_50,
            interaction_count=n_interactions,
            cold_start_flag=cold_start,
            last_seen_ts=last_seen_ts,
        )
        await store.set_user_features(features)
        user_count += 1

    elapsed = time.perf_counter() - t0
    print(f"Written {user_count} users, {item_count} items to Redis in {elapsed:.1f} seconds.")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
