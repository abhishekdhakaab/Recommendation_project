import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import types

import numpy as np
import pytest

from models.cold_start import (
    BGEItemEmbeddingProvider,
    ColdStartRouter,
    PopularityItem,
    RecommendationRoute,
    average_item_embeddings,
)
from retrieval.build_faiss_index import build_hnsw_index


def test_zero_interaction_user_gets_popularity_weighted_fallback() -> None:
    router = ColdStartRouter(
        popularity_items=[
            PopularityItem(item_id=3, popularity_score=0.3),
            PopularityItem(item_id=1, popularity_score=0.9),
            PopularityItem(item_id=2, popularity_score=0.5),
        ],
    )

    result = router.route(interaction_history=[], top_k=2)

    assert result.route == RecommendationRoute.POPULARITY_FALLBACK
    assert [recommendation.item_id for recommendation in result.recommendations] == [1, 2]
    assert [recommendation.score for recommendation in result.recommendations] == [0.9, 0.5]


def test_sparse_user_averages_item_embeddings_and_queries_content_index() -> None:
    item_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    content_index = build_hnsw_index(item_embeddings, item_ids=[10, 11, 12, 13])
    router = ColdStartRouter(
        popularity_items=[],
        embedding_provider=FakeEmbeddingProvider({10: item_embeddings[0], 11: item_embeddings[1]}),
        content_index=content_index,
    )

    result = router.route(interaction_history=[10, 11], top_k=2)

    assert result.route == RecommendationRoute.CONTENT_BASED
    assert set(recommendation.item_id for recommendation in result.recommendations) == {10, 11}
    assert all(recommendation.source == RecommendationRoute.CONTENT_BASED for recommendation in result.recommendations)


def test_warm_user_routes_to_warm_path_without_recommendations() -> None:
    router = ColdStartRouter(popularity_items=[])

    result = router.route(interaction_history=[1, 2, 3, 4, 5], top_k=10)

    assert result.route == RecommendationRoute.WARM_PATH
    assert result.recommendations == []


def test_average_item_embeddings_returns_normalized_query() -> None:
    provider = FakeEmbeddingProvider(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([1.0, 1.0], dtype=np.float32),
        }
    )

    query = average_item_embeddings(provider, [1, 2])

    assert query.shape == (1, 2)
    assert np.linalg.norm(query, axis=1)[0] == pytest.approx(1.0)
    assert query[0, 0] > query[0, 1]


def test_sparse_user_requires_content_dependencies() -> None:
    router = ColdStartRouter(popularity_items=[])

    with pytest.raises(ValueError, match="content-based routing requires"):
        router.route(interaction_history=[1], top_k=5)


def test_bge_provider_loads_model_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            calls.append(model_name)

        def encode(self, texts: list[str], normalize_embeddings: bool) -> np.ndarray:
            assert normalize_embeddings is True
            return np.ones((len(texts), 3), dtype=np.float32)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    provider = BGEItemEmbeddingProvider({1: "one", 2: "two"}, model_name="fake-bge")
    assert calls == []

    embeddings = provider.embed_item_ids([1, 2])

    assert calls == ["fake-bge"]
    assert embeddings.shape == (2, 3)


class FakeEmbeddingProvider:
    def __init__(self, embeddings_by_item_id: dict[int, np.ndarray]) -> None:
        self.embeddings_by_item_id = embeddings_by_item_id

    def embed_item_ids(self, item_ids: list[int]) -> np.ndarray:
        return np.asarray([self.embeddings_by_item_id[item_id] for item_id in item_ids], dtype=np.float32)
