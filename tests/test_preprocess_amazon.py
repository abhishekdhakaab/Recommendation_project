import gzip
import json
from pathlib import Path

from data.preprocess_amazon import preprocess_amazon_jsonl_gz


def test_preprocess_amazon_jsonl_gz_streams_category_files(tmp_path: Path) -> None:
    reviews = tmp_path / "reviews.jsonl.gz"
    meta = tmp_path / "meta.jsonl.gz"
    output = tmp_path / "processed"
    review_rows = []
    for user in ("u1", "u2"):
        for idx, item in enumerate(("p1", "p2", "p1")):
            review_rows.append({"user_id": user, "parent_asin": item, "asin": item, "rating": 5, "timestamp": idx + (10 if user == "u2" else 0), "text": "ok"})
    with gzip.open(reviews, "wt", encoding="utf-8") as file:
        for row in review_rows:
            file.write(json.dumps(row) + "\n")
    with gzip.open(meta, "wt", encoding="utf-8") as file:
        for item in ("p1", "p2"):
            file.write(json.dumps({"parent_asin": item, "title": item, "main_category": "Sports", "description": ["desc"]}) + "\n")

    stats = preprocess_amazon_jsonl_gz(
        interactions_path=reviews,
        metadata_path=meta,
        output_dir=output,
        user_min_interactions=3,
        item_min_interactions=2,
    )

    assert stats["users"] == 2
    assert stats["items"] == 2
    assert stats["train_interactions"] == 2
    assert (output / "train.jsonl").exists()
