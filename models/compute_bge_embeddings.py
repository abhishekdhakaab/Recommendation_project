"""Pre-compute BGE text embeddings for all items in a processed dataset.

Produces a .npy array of shape (n_items, bge_dim) aligned to item IDs 1..n_items
(0-indexed in the file, so file row i corresponds to model item ID i+1).

Usage:
    python models/compute_bge_embeddings.py \
        --processed-dir data/processed/sports_and_outdoors \
        --output-path models/artifacts/sports_and_outdoors_bge/bge_item_embeddings.npy \
        --model-path models/artifacts/hf_models/bge-large-en-v1.5

The resulting .npy file can be passed directly to train_sasrec.py via
--text-embeddings-path (the training script prepends the padding row
automatically) or to models/cold_start.py for cold-start retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np


DEFAULT_BGE_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_BATCH_SIZE = 256


def compute_item_embeddings(
    *,
    processed_dir: str | Path,
    output_path: str | Path,
    model_name_or_path: str = DEFAULT_BGE_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_items: int | None = None,
) -> dict[str, object]:
    """Compute and save BGE embeddings for every item in the processed dataset.

    Returns a summary dict with shape, model, and output path.
    """
    from sentence_transformers import SentenceTransformer

    processed_path = Path(processed_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = json.loads((processed_path / "stats.json").read_text(encoding="utf-8"))
    n_items = stats["items"]
    print(f"Dataset: {n_items} items")

    # Load item metadata sorted by item_id (0-indexed in the file).
    metadata_path = processed_path / "item_metadata.jsonl"
    rows: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    # Sort by item_id so the embedding matrix rows are aligned.
    rows.sort(key=lambda r: int(r["item_id"]))
    if max_items is not None:
        rows = rows[:max_items]
    actual_n = len(rows)
    print(f"Embedding {actual_n} items (batch_size={batch_size})")

    # Build text strings: "title . description" (description often empty).
    def _item_text(row: dict) -> str:
        title = str(row.get("title") or "").strip()
        desc = str(row.get("description") or "").strip()
        if desc:
            return f"{title}. {desc}"
        return title

    texts = [_item_text(r) for r in rows]

    print(f"Loading BGE model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path)

    print("Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(str(out_path), embeddings)
    print(f"Saved → {out_path}")

    # Save a summary alongside the .npy file.
    summary = {
        "model": model_name_or_path,
        "n_items": actual_n,
        "embedding_dim": int(embeddings.shape[1]),
        "output_path": str(out_path),
    }
    summary_path = out_path.parent / (out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BGE item text embeddings.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-path", default=DEFAULT_BGE_MODEL,
                        help="HuggingFace model name or local path to bge model directory.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    summary = compute_item_embeddings(
        processed_dir=args.processed_dir,
        output_path=args.output_path,
        model_name_or_path=args.model_path,
        batch_size=args.batch_size,
        max_items=args.max_items,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
