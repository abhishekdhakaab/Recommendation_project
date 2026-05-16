import sys
import types

import pytest

from models.reranker import (
    BGEReranker,
    DEFAULT_BGE_RERANKER_MODEL,
    MockReranker,
    RerankCandidate,
    rerank_candidates,
    rerank_top_50_to_top_10,
)


def test_mock_reranker_scores_item_texts() -> None:
    reranker = MockReranker({"great item": 0.9, "okay item": 0.2})

    scores = reranker.score("recent user history", ["okay item", "great item", "unknown"])

    assert scores == [0.2, 0.9, 0.0]


def test_rerank_candidates_sorts_by_score_and_returns_top_n() -> None:
    reranker = MockReranker({"alpha": 0.1, "bravo": 0.8, "charlie": 0.5})
    candidates = [
        RerankCandidate(item_id=1, text="alpha", retrieval_score=0.99),
        RerankCandidate(item_id=2, text="bravo", retrieval_score=0.3),
        RerankCandidate(item_id=3, text="charlie", retrieval_score=0.7),
    ]

    reranked = rerank_candidates(
        reranker,
        user_query_text="user likes bravo",
        candidates=candidates,
        top_n=2,
    )

    assert [candidate.item_id for candidate in reranked] == [2, 3]
    assert [candidate.score for candidate in reranked] == [0.8, 0.5]
    assert reranked[0].retrieval_score == 0.3


def test_rerank_candidates_accepts_dict_candidates() -> None:
    reranker = MockReranker({"item a": 0.4, "item b": 0.6})

    reranked = rerank_candidates(
        reranker,
        user_query_text="query",
        candidates=[
            {"item_id": 10, "text": "item a", "retrieval_score": 1.0},
            {"item_id": 11, "text": "item b", "retrieval_score": 0.5},
        ],
        top_n=10,
    )

    assert [candidate.item_id for candidate in reranked] == [11, 10]


def test_rerank_top_50_to_top_10_caps_input_and_output() -> None:
    candidates = [RerankCandidate(item_id=index, text=f"item {index}") for index in range(60)]
    scores = {f"item {index}": float(index) for index in range(60)}
    reranker = MockReranker(scores)

    reranked = rerank_top_50_to_top_10(
        reranker,
        user_query_text="query",
        candidates=candidates,
    )

    assert len(reranked) == 10
    assert [candidate.item_id for candidate in reranked] == list(range(49, 39, -1))


def test_bge_reranker_loads_cross_encoder_lazily_and_uses_configurable_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    predicted_pairs: list[tuple[str, str]] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            calls.append(model_name)

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            predicted_pairs.extend(pairs)
            return [0.1, 0.9]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = BGEReranker(model_name="fake-cross-encoder")
    assert calls == []

    scores = reranker.score("history text", ["item one", "item two"])

    assert calls == ["fake-cross-encoder"]
    assert predicted_pairs == [("history text", "item one"), ("history text", "item two")]
    assert scores == [0.1, 0.9]


def test_bge_reranker_default_model_matches_prd_choice() -> None:
    assert BGEReranker().model_name == DEFAULT_BGE_RERANKER_MODEL


def test_reranker_validation() -> None:
    with pytest.raises(ValueError, match="top_n must be at least 1"):
        rerank_candidates(MockReranker(), user_query_text="query", candidates=[], top_n=0)

    with pytest.raises(ValueError, match="must not be empty"):
        MockReranker().score("", ["item"])

    with pytest.raises(ValueError, match="different number of scores"):
        rerank_candidates(
            BadReranker(),
            user_query_text="query",
            candidates=[RerankCandidate(item_id=1, text="one")],
        )


class BadReranker:
    def score(self, user_query_text: str, item_texts: list[str]) -> list[float]:
        return []
