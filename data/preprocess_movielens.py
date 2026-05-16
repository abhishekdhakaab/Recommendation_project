"""Preprocess MovieLens 1M into SeqRec pipeline format.

Usage:
    python data/preprocess_movielens.py \
        --ratings  data/raw/ml-1m/ml-1m/ratings.dat \
        --movies   data/raw/ml-1m/ml-1m/movies.dat \
        --output-dir data/processed/ml-1m
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.preprocess import preprocess_dataset


def preprocess_movielens(*, ratings_path, movies_path, output_dir,
                         user_min_interactions=5, item_min_interactions=5):
    interactions, item_metadata = [], []

    with open(ratings_path, encoding="latin-1") as f:
        for line in f:
            uid, iid, rating, ts = line.strip().split("::")
            interactions.append({"user_id": uid, "item_id": iid,
                                  "rating": float(rating), "timestamp": int(ts)})

    with open(movies_path, encoding="latin-1") as f:
        for line in f:
            iid, title, genres = line.strip().split("::", 2)
            item_metadata.append({"item_id": iid, "title": title,
                                   "category": genres.split("|")[0],
                                   "description": genres.replace("|", " ")})

    artifacts = preprocess_dataset(
        interactions, item_metadata,
        output_dir=output_dir,
        user_min_interactions=user_min_interactions,
        item_min_interactions=item_min_interactions,
    )
    print(json.dumps(artifacts.stats, indent=2, sort_keys=True))
    return artifacts.stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ratings", required=True)
    p.add_argument("--movies", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--user-min-interactions", type=int, default=5)
    p.add_argument("--item-min-interactions", type=int, default=5)
    args = p.parse_args()
    preprocess_movielens(ratings_path=args.ratings, movies_path=args.movies,
                         output_dir=args.output_dir,
                         user_min_interactions=args.user_min_interactions,
                         item_min_interactions=args.item_min_interactions)
