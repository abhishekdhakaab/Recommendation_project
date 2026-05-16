"""Streaming preprocessing for Amazon Reviews 2023 category JSONL files."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.preprocess import preprocess_dataset

Record = dict[str, Any]


def preprocess_amazon_jsonl_gz(
    *,
    interactions_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    user_min_interactions: int = 5,
    item_min_interactions: int = 10,
    max_interactions: int | None = None,
) -> dict[str, int]:
    """Stream Amazon JSONL.GZ files and write processed artifacts."""

    interactions_file = Path(interactions_path)
    metadata_file = Path(metadata_path)
    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    raw_interactions = 0

    with gzip.open(interactions_file, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            user_id = str(row["user_id"])
            item_id = str(row.get("parent_asin") or row.get("asin"))
            user_counts[user_id] += 1
            item_counts[item_id] += 1
            raw_interactions += 1
            if max_interactions is not None and raw_interactions >= max_interactions:
                break

    eligible_users = {user_id for user_id, count in user_counts.items() if count >= user_min_interactions}
    eligible_items = {item_id for item_id, count in item_counts.items() if count >= item_min_interactions}

    interactions: list[Record] = []
    with gzip.open(interactions_file, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            user_id = str(row["user_id"])
            item_id = str(row.get("parent_asin") or row.get("asin"))
            if user_id in eligible_users and item_id in eligible_items:
                interactions.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "raw_item_id": str(row.get("asin", item_id)),
                        "rating": float(row.get("rating", 0.0)),
                        "timestamp": int(row["timestamp"]),
                        "review_text": row.get("text", ""),
                    }
                )
            if max_interactions is not None and len(interactions) >= max_interactions:
                break

    needed_items = {row["item_id"] for row in interactions}
    item_metadata: list[Record] = []
    seen_items: set[str] = set()
    with gzip.open(metadata_file, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("parent_asin"))
            if item_id not in needed_items or item_id in seen_items:
                continue
            seen_items.add(item_id)
            item_metadata.append(
                {
                    "item_id": item_id,
                    "title": row.get("title", ""),
                    "category": row.get("main_category", "") or _first_string(row.get("categories")),
                    "description": _text_blob(row.get("description")),
                    "price": row.get("price"),
                    "image_url": _first_image_url(row.get("images")),
                }
            )

    artifacts = preprocess_dataset(
        interactions,
        item_metadata,
        output_dir=output_dir,
        user_min_interactions=user_min_interactions,
        item_min_interactions=item_min_interactions,
    )
    stats = dict(artifacts.stats)
    stats["first_pass_raw_interactions"] = raw_interactions
    stats["first_pass_eligible_users"] = len(eligible_users)
    stats["first_pass_eligible_items"] = len(eligible_items)
    with (Path(output_dir) / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, sort_keys=True)
        file.write("\n")
    return stats


def _first_string(value: object) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                return item
    return ""


def _text_blob(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def _first_image_url(value: object) -> str:
    if isinstance(value, list):
        for image in value:
            if isinstance(image, dict):
                for key in ("large", "hi_res", "thumb"):
                    if image.get(key):
                        return str(image[key])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-preprocess Amazon Reviews 2023 JSONL.GZ files.")
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--user-min-interactions", type=int, default=5)
    parser.add_argument("--item-min-interactions", type=int, default=10)
    parser.add_argument("--max-interactions", type=int, default=None)
    args = parser.parse_args()
    stats = preprocess_amazon_jsonl_gz(
        interactions_path=args.interactions,
        metadata_path=args.items,
        output_dir=args.output_dir,
        user_min_interactions=args.user_min_interactions,
        item_min_interactions=args.item_min_interactions,
        max_interactions=args.max_interactions,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
