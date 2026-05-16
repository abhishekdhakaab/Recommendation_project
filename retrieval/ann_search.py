"""FAISS ANN search helpers for SeqRec retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Sequence

import faiss
import numpy as np

from retrieval.build_faiss_index import HNSWItemIndex


@dataclass(frozen=True)
class SearchResult:
    """Single ANN search result."""

    item_id: int
    score: float


@dataclass(frozen=True)
class SearchBenchmark:
    """Latency summary for repeated ANN searches."""

    runs: int
    top_k: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


def search_top_k(
    index: HNSWItemIndex | faiss.Index,
    query_embeddings: np.ndarray,
    *,
    top_k: int,
) -> list[list[SearchResult]]:
    """Query a FAISS index and return top-k item IDs with scores."""

    _validate_top_k(top_k)
    queries = _as_float32_matrix(query_embeddings, "query_embeddings")
    faiss_index = _faiss_index(index)
    item_ids = _item_ids(index)
    if queries.shape[1] != faiss_index.d:
        raise ValueError("query embedding dimension must match index dimension")

    search_rows: list[list[SearchResult]] = []
    for query in queries:
        scores, ids = faiss_index.search(query.reshape(1, -1), top_k)
        search_rows.append(
            [
                SearchResult(item_id=int(item_ids[int(internal_id)]), score=float(score))
                for internal_id, score in zip(ids[0], scores[0], strict=True)
                if int(internal_id) != -1
            ]
        )
    return search_rows


def search_item_ids(
    index: HNSWItemIndex | faiss.Index,
    query_embeddings: np.ndarray,
    *,
    top_k: int,
) -> list[list[int]]:
    """Query a FAISS index and return only top-k item IDs."""

    return [[result.item_id for result in row] for row in search_top_k(index, query_embeddings, top_k=top_k)]


def benchmark_search(
    index: HNSWItemIndex | faiss.Index,
    query_embeddings: np.ndarray,
    *,
    top_k: int,
    runs: int = 20,
) -> SearchBenchmark:
    """Measure repeated FAISS search latency in milliseconds."""

    _validate_top_k(top_k)
    if runs < 1:
        raise ValueError("runs must be at least 1")

    latencies_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        search_top_k(index, query_embeddings, top_k=top_k)
        latencies_ms.append((time.perf_counter() - started) * 1000)

    sorted_latencies = sorted(latencies_ms)
    return SearchBenchmark(
        runs=runs,
        top_k=top_k,
        mean_ms=statistics.fmean(latencies_ms),
        p50_ms=_percentile(sorted_latencies, 50),
        p95_ms=_percentile(sorted_latencies, 95),
    )


def _as_float32_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    return np.ascontiguousarray(matrix)


def _percentile(sorted_values: Sequence[float], percentile: int) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _validate_top_k(top_k: int) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")


def _faiss_index(index: HNSWItemIndex | faiss.Index) -> faiss.Index:
    if isinstance(index, HNSWItemIndex):
        return index.index
    return index


def _item_ids(index: HNSWItemIndex | faiss.Index) -> np.ndarray:
    if isinstance(index, HNSWItemIndex):
        return index.item_ids
    return np.arange(index.ntotal, dtype=np.int64)


# ---------------------------------------------------------------------------
# Lazy cached retrieval helper
# ---------------------------------------------------------------------------

_cached_index: HNSWItemIndex | None = None


def retrieve_top_k(
    user_embedding: np.ndarray,
    *,
    k: int = 500,
    index_path: str | None = None,
) -> list[int]:
    """Load FAISS index (lazy, cached) and return top-k item_ids for user_embedding.

    Parameters
    ----------
    user_embedding:
        1D or 2D float32 array representing the user query.
    k:
        Number of top items to retrieve.
    index_path:
        Optional path to the FAISS index file. Defaults to the standard
        beauty tower artifact location.
    """

    global _cached_index
    if _cached_index is None:
        from retrieval.build_faiss_index import load_index

        path = index_path or "models/artifacts/beauty_tower/item_index.faiss"
        _cached_index = load_index(path)

    results = search_item_ids(
        _cached_index,
        np.atleast_2d(user_embedding).astype(np.float32),
        top_k=k,
    )
    return results[0] if results else []
