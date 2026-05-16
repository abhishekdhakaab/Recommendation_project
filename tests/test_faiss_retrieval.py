import numpy as np
import pytest

from retrieval.ann_search import benchmark_search, search_item_ids, search_top_k
from retrieval.build_faiss_index import (
    DEFAULT_EF_CONSTRUCTION,
    DEFAULT_HNSW_M,
    build_hnsw_index,
    load_index,
    save_index,
)


def test_build_hnsw_index_uses_prd_defaults_and_queries_item_ids() -> None:
    embeddings = _tiny_embeddings()
    item_ids = np.array([100, 101, 102, 103], dtype=np.int64)

    index = build_hnsw_index(embeddings, item_ids)
    results = search_item_ids(index, embeddings[[2]], top_k=2)

    assert index.index.hnsw.nb_neighbors(0) == DEFAULT_HNSW_M * 2
    assert index.index.hnsw.efConstruction == DEFAULT_EF_CONSTRUCTION
    assert results[0][0] == 102
    assert len(results[0]) == 2


def test_save_and_load_index_preserves_search_results(tmp_path) -> None:
    embeddings = _tiny_embeddings()
    item_ids = np.array([10, 11, 12, 13], dtype=np.int64)
    index = build_hnsw_index(embeddings, item_ids)
    index_path = tmp_path / "item_index.faiss"

    save_index(index, index_path)
    loaded = load_index(index_path)

    assert index_path.exists()
    assert index_path.with_suffix(index_path.suffix + ".ids.npy").exists()
    assert search_item_ids(loaded, embeddings[[1]], top_k=1) == [[11]]


def test_search_top_k_returns_scores_and_accepts_single_query_vector() -> None:
    embeddings = _tiny_embeddings()
    index = build_hnsw_index(embeddings, [0, 1, 2, 3])

    results = search_top_k(index, embeddings[0], top_k=3)

    assert len(results) == 1
    assert [result.item_id for result in results[0]][0] == 0
    assert results[0][0].score == pytest.approx(1.0)


def test_benchmark_search_reports_latency_summary() -> None:
    embeddings = _tiny_embeddings()
    index = build_hnsw_index(embeddings)

    benchmark = benchmark_search(index, embeddings[:2], top_k=2, runs=3)

    assert benchmark.runs == 3
    assert benchmark.top_k == 2
    assert benchmark.mean_ms >= 0
    assert benchmark.p50_ms >= 0
    assert benchmark.p95_ms >= 0


def test_faiss_helpers_validate_inputs() -> None:
    embeddings = _tiny_embeddings()
    index = build_hnsw_index(embeddings)

    with pytest.raises(ValueError, match="item_ids length"):
        build_hnsw_index(embeddings, [1, 2])
    with pytest.raises(ValueError, match="query embedding dimension"):
        search_item_ids(index, np.ones((1, 2), dtype=np.float32), top_k=1)
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        search_item_ids(index, embeddings[:1], top_k=0)
    with pytest.raises(ValueError, match="runs must be at least 1"):
        benchmark_search(index, embeddings[:1], top_k=1, runs=0)


def _tiny_embeddings() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.7, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
