"""Build exact FAISS flat indexes for smoke/evaluation workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from retrieval.build_faiss_index import HNSWItemIndex, save_index


def build_flat_index(item_embeddings: np.ndarray, item_ids: np.ndarray | None = None) -> HNSWItemIndex:
    embeddings = np.asarray(item_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("item_embeddings must be 2D")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))
    ids = np.arange(embeddings.shape[0], dtype=np.int64) if item_ids is None else np.asarray(item_ids, dtype=np.int64)
    return HNSWItemIndex(index=index, item_ids=ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact FAISS flat index from saved item embeddings.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    embeddings = np.load(args.embeddings)
    index = build_flat_index(embeddings)
    save_index(index, args.output)
    summary = {"items": int(embeddings.shape[0]), "dim": int(embeddings.shape[1]), "output": args.output}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
