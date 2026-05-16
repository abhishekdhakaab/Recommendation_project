import math

import pytest

from evaluation.offline_eval import (
    RankingMetrics,
    evaluate_ranking_batch,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_ndcg_at_k_with_exact_binary_relevance() -> None:
    recommended = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    expected_dcg = (1 / math.log2(3)) + (1 / math.log2(5))
    expected_idcg = 1 + (1 / math.log2(3))

    assert ndcg_at_k(recommended, relevant, k=4) == pytest.approx(expected_dcg / expected_idcg)


def test_hit_at_k_with_hit_and_miss() -> None:
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert hit_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0


def test_mrr_at_k_uses_first_relevant_rank() -> None:
    assert mrr_at_k(["a", "b", "c", "d"], {"b", "d"}, k=4) == pytest.approx(1 / 2)
    assert mrr_at_k(["a", "b", "c", "d"], {"d"}, k=3) == 0.0


def test_recall_at_k_counts_relevant_items_in_top_k() -> None:
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d", "e"}, k=4) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d", "e"}, k=2) == pytest.approx(1 / 3)


def test_empty_relevance_returns_zero_for_single_user_metrics() -> None:
    recommended = ["a", "b"]

    assert ndcg_at_k(recommended, set(), k=2) == 0.0
    assert hit_at_k(recommended, set(), k=2) == 0.0
    assert mrr_at_k(recommended, set(), k=2) == 0.0
    assert recall_at_k(recommended, set(), k=2) == 0.0


def test_evaluate_ranking_batch_averages_over_users() -> None:
    recommendations = {
        "u1": ["a", "b", "c"],
        "u2": ["d", "e", "f"],
        "u3": ["g", "h", "i"],
    }
    relevant = {
        "u1": {"a"},
        "u2": {"f"},
        "u3": {"x"},
    }

    metrics = evaluate_ranking_batch(recommendations, relevant, k=3)

    assert metrics == RankingMetrics(
        ndcg=pytest.approx((1.0 + (1 / math.log2(4)) + 0.0) / 3),
        hit=pytest.approx(2 / 3),
        mrr=pytest.approx((1.0 + (1 / 3) + 0.0) / 3),
        recall=pytest.approx(2 / 3),
        users=3,
    )


def test_evaluate_ranking_batch_counts_missing_recommendations_as_zero() -> None:
    metrics = evaluate_ranking_batch({"u1": ["a"]}, {"u1": {"a"}, "u2": {"b"}}, k=1)

    assert metrics.ndcg == pytest.approx(0.5)
    assert metrics.hit == pytest.approx(0.5)
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.users == 2


def test_metrics_validate_k() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        ndcg_at_k(["a"], {"a"}, k=0)


def test_evaluate_sampled_ranking_batch() -> None:
    from evaluation.offline_eval import evaluate_sampled_ranking_batch

    def score_items(user_id: str, candidates: list[str]) -> list[float]:
        return [1.0 if item_id == "i1" else 0.0 for item_id in candidates]

    metrics = evaluate_sampled_ranking_batch(
        score_items,
        {"u1": {"i1"}},
        all_item_ids=["i1", "i2", "i3", "i4"],
        k=1,
        n_negatives=2,
    )

    assert metrics.hit == 1.0
    assert metrics.ndcg == 1.0


def test_evaluate_popularity_sampled_ranking_batch() -> None:
    from evaluation.offline_eval import evaluate_popularity_sampled_ranking_batch

    def score_items(user_id: str, candidates: list[str]) -> list[float]:
        return [1.0 if item_id == "i1" else 0.0 for item_id in candidates]

    metrics = evaluate_popularity_sampled_ranking_batch(
        score_items,
        {"u1": {"i1"}},
        popularity_counts={"i1": 10, "i2": 5, "i3": 4, "i4": 3},
        k=1,
        n_negatives=2,
    )

    assert metrics.hit == 1.0
