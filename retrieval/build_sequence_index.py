"""Build a persisted sequence-aware retrieval artifact from processed data."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval.sequence_retrieval import build_sequence_retrieval_index, save_sequence_retrieval_index


def build_sequence_index_artifact(
    *,
    processed_dir: str | Path,
    output_dir: str | Path,
    top_items: int = 5000,
    embedding_dim: int = 64,
    context_window: int = 10,
) -> dict[str, object]:
    processed = Path(processed_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train = _read_jsonl(processed / "train.jsonl")
    popularity = Counter(int(row["item_id"]) for row in train)
    top_item_ids = [item_id for item_id, _ in popularity.most_common(top_items)]

    histories: dict[int, list[int]] = defaultdict(list)
    for row in sorted(train, key=lambda value: int(value["timestamp"])):
        histories[int(row["user_id"])].append(int(row["item_id"]))

    index = build_sequence_retrieval_index(
        histories.values(),
        item_ids=top_item_ids,
        embedding_dim=embedding_dim,
        context_window=context_window,
    )
    save_sequence_retrieval_index(index, output / "sequence_index.npz")
    _write_jsonl(
        output / "popularity.jsonl",
        ({"item_id": item_id, "count": count} for item_id, count in popularity.most_common()),
    )
    _write_jsonl(
        output / "user_histories.jsonl",
        ({"user_id": user_id, "history": history} for user_id, history in sorted(histories.items())),
    )
    summary = {
        "processed_dir": str(processed),
        "top_items": len(top_item_ids),
        "embedding_dim": embedding_dim,
        "context_window": context_window,
        "users": len(histories),
        "train_interactions": len(train),
        "index_path": str(output / "sequence_index.npz"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "
")
    return summary


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "
")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sequence retrieval artifact.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-items", type=int, default=5000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--context-window", type=int, default=10)
    args = parser.parse_args()
    summary = build_sequence_index_artifact(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        top_items=args.top_items,
        embedding_dim=args.embedding_dim,
        context_window=args.context_window,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
