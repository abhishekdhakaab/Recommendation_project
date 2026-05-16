import csv
import json
from pathlib import Path

from data.preprocess import (
    k_core_filter,
    load_interactions,
    load_item_metadata,
    preprocess_dataset,
    preprocess_files,
)
from data.synthetic import generate_synthetic_dataset


def test_k_core_filter_iteratively_filters_users_and_items() -> None:
    interactions = [
        {"user_id": "u1", "item_id": "i1", "timestamp": 1},
        {"user_id": "u1", "item_id": "i2", "timestamp": 2},
        {"user_id": "u2", "item_id": "i1", "timestamp": 3},
        {"user_id": "u3", "item_id": "i2", "timestamp": 4},
        {"user_id": "u3", "item_id": "i3", "timestamp": 5},
    ]

    filtered = k_core_filter(interactions, user_min_interactions=2, item_min_interactions=2)

    assert filtered == []


def test_preprocess_dataset_sorts_encodes_and_splits_synthetic_data(tmp_path: Path) -> None:
    dataset = generate_synthetic_dataset(seed=42, n_users=6, n_items=12)

    artifacts = preprocess_dataset(
        dataset.interactions,
        dataset.items,
        output_dir=tmp_path,
        user_min_interactions=3,
        item_min_interactions=1,
    )

    assert artifacts.user_mapping
    assert artifacts.item_mapping
    assert sorted(artifacts.user_mapping.values()) == list(range(len(artifacts.user_mapping)))
    assert sorted(artifacts.item_mapping.values()) == list(range(len(artifacts.item_mapping)))
    assert len(artifacts.validation) == len(artifacts.test) == artifacts.stats["users"]
    assert artifacts.stats["raw_interactions"] == len(dataset.interactions)

    for split_name in ("train", "validation", "test"):
        for row in getattr(artifacts, split_name):
            assert isinstance(row["user_id"], int)
            assert isinstance(row["item_id"], int)
            assert "raw_user_id" in row
            assert "raw_item_id" in row

    for user_id in artifacts.user_mapping.values():
        train_rows = [row for row in artifacts.train if row["user_id"] == user_id]
        val_row = next(row for row in artifacts.validation if row["user_id"] == user_id)
        test_row = next(row for row in artifacts.test if row["user_id"] == user_id)
        train_timestamps = [row["timestamp"] for row in train_rows]

        assert train_timestamps == sorted(train_timestamps)
        assert max(train_timestamps, default=-1) < val_row["timestamp"] < test_row["timestamp"]

    for artifact_name in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "item_metadata.jsonl",
        "user_mapping.json",
        "item_mapping.json",
        "stats.json",
    ):
        assert (tmp_path / artifact_name).exists()


def test_preprocess_files_loads_json_inputs_and_writes_outputs(tmp_path: Path) -> None:
    dataset = generate_synthetic_dataset(seed=7, n_users=6, n_items=12)
    interactions_path = tmp_path / "interactions.json"
    items_path = tmp_path / "items.json"
    output_dir = tmp_path / "processed"
    interactions_path.write_text(json.dumps(dataset.interactions), encoding="utf-8")
    items_path.write_text(json.dumps({"items": dataset.items}), encoding="utf-8")

    artifacts = preprocess_files(
        interactions_path=interactions_path,
        item_metadata_path=items_path,
        output_dir=output_dir,
        user_min_interactions=3,
        item_min_interactions=1,
    )

    stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats == artifacts.stats
    assert (output_dir / "train.jsonl").read_text(encoding="utf-8")


def test_loaders_support_csv(tmp_path: Path) -> None:
    interactions_path = tmp_path / "interactions.csv"
    items_path = tmp_path / "items.csv"

    with interactions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "item_id", "timestamp"])
        writer.writeheader()
        writer.writerow({"user_id": "u1", "item_id": "i1", "timestamp": "1"})

    with items_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["item_id", "title"])
        writer.writeheader()
        writer.writerow({"item_id": "i1", "title": "Item 1"})

    assert load_interactions(interactions_path) == [{"user_id": "u1", "item_id": "i1", "timestamp": "1"}]
    assert load_item_metadata(items_path) == [{"item_id": "i1", "title": "Item 1", "category": "", "description": ""}]




def test_loaders_support_amazon_jsonl_gz(tmp_path: Path) -> None:
    import gzip

    interactions_path = tmp_path / "reviews.jsonl.gz"
    items_path = tmp_path / "meta.jsonl.gz"
    with gzip.open(interactions_path, "wt", encoding="utf-8") as file:
        file.write(json.dumps({"user_id": "U1", "parent_asin": "P1", "asin": "A1", "timestamp": 1, "text": "review"}) + "\n")
    with gzip.open(items_path, "wt", encoding="utf-8") as file:
        file.write(json.dumps({"parent_asin": "P1", "title": "Item", "main_category": "All Beauty", "description": ["nice"]}) + "\n")

    interactions = load_interactions(interactions_path)
    items = load_item_metadata(items_path)

    assert interactions[0]["item_id"] == "P1"
    assert interactions[0]["raw_item_id"] == "A1"
    assert interactions[0]["review_text"] == "review"
    assert items[0]["item_id"] == "P1"
    assert items[0]["category"] == "All Beauty"
    assert items[0]["description"] == "nice"
