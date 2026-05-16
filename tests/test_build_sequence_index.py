import json

from retrieval.build_sequence_index import build_sequence_index_artifact
from retrieval.sequence_retrieval import load_sequence_retrieval_index


def test_build_sequence_index_artifact(tmp_path) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "artifact"
    processed.mkdir()
    rows = [
        {"user_id": 0, "item_id": 1, "timestamp": 1},
        {"user_id": 0, "item_id": 2, "timestamp": 2},
        {"user_id": 1, "item_id": 2, "timestamp": 1},
        {"user_id": 1, "item_id": 3, "timestamp": 2},
    ]
    _write_jsonl(processed / "train.jsonl", rows)

    summary = build_sequence_index_artifact(
        processed_dir=processed,
        output_dir=output,
        top_items=3,
        embedding_dim=2,
    )

    assert summary["users"] == 2
    assert (output / "sequence_index.npz").exists()
    assert (output / "popularity.jsonl").exists()
    assert (output / "user_histories.jsonl").exists()
    index = load_sequence_retrieval_index(output / "sequence_index.npz")
    assert set(index.item_ids) == {1, 2, 3}


def _write_jsonl(path, rows) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "
")
