"""Build and persist FAISS indexes for SeqRec retrieval."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import faiss
import numpy as np


DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 200


@dataclass(frozen=True)
class HNSWItemIndex:
    """FAISS HNSW index plus external item ID mapping."""

    index: faiss.Index
    item_ids: np.ndarray


def build_hnsw_index(
    item_embeddings: np.ndarray,
    item_ids: Sequence[int] | np.ndarray | None = None,
    *,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
) -> HNSWItemIndex:
    """Build an HNSW inner-product index with explicit item IDs."""

    embeddings = _as_float32_matrix(item_embeddings, "item_embeddings")
    _validate_positive("m", m)
    _validate_positive("ef_construction", ef_construction)

    dim = embeddings.shape[1]
    hnsw = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = ef_construction

    ids = _item_ids(item_ids, embeddings.shape[0])
    hnsw.add(embeddings)
    return HNSWItemIndex(index=hnsw, item_ids=ids)


def save_index(index: HNSWItemIndex | faiss.Index, path: str | Path) -> None:
    """Save a FAISS index to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(index, HNSWItemIndex):
        faiss.write_index(index.index, str(output_path))
        np.save(_ids_path(output_path), index.item_ids)
        return

    faiss.write_index(index, str(output_path))


def load_index(path: str | Path) -> HNSWItemIndex:
    """Load a FAISS index from disk."""

    input_path = Path(path)
    index = faiss.read_index(str(input_path))
    ids_path = _ids_path(input_path)
    if ids_path.exists():
        item_ids = np.load(ids_path)
    else:
        item_ids = np.arange(index.ntotal, dtype=np.int64)
    return HNSWItemIndex(index=index, item_ids=np.asarray(item_ids, dtype=np.int64))


def _as_float32_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    return np.ascontiguousarray(matrix)


def _item_ids(item_ids: Sequence[int] | np.ndarray | None, n_items: int) -> np.ndarray:
    if item_ids is None:
        return np.arange(n_items, dtype=np.int64)

    ids = np.asarray(item_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("item_ids must be a 1D array")
    if ids.shape[0] != n_items:
        raise ValueError("item_ids length must match item_embeddings rows")
    if len(set(ids.tolist())) != ids.shape[0]:
        raise ValueError("item_ids must be unique")
    return np.ascontiguousarray(ids)


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _ids_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".ids.npy")
