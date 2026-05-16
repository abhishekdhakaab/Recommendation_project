"""Offline ranking metrics for SeqRec evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Iterable, Mapping, Sequence


ItemId = Hashable


@dataclass(frozen=True)
class RankingMetrics:
    """Averaged ranking metrics over a batch of users."""

    ndcg: float
    hit: float
    mrr: float
    recall: float
    users: int


def ndcg_at_k(recommended: Sequence[ItemId], relevant: Iterable[ItemId], k: int) -> float:
    """Compute binary NDCG@K for a single user."""

    _validate_k(k)
    relevant_items = set(relevant)
    if not relevant_items:
        return 0.0

    dcg = 0.0
    for rank, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant_items:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def hit_at_k(recommended: Sequence[ItemId], relevant: Iterable[ItemId], k: int) -> float:
    """Return 1.0 when any relevant item appears in the top K."""

    _validate_k(k)
    relevant_items = set(relevant)
    if not relevant_items:
        return 0.0
    return float(any(item_id in relevant_items for item_id in recommended[:k]))


def mrr_at_k(recommended: Sequence[ItemId], relevant: Iterable[ItemId], k: int) -> float:
    """Compute reciprocal rank of the first relevant item in the top K."""

    _validate_k(k)
    relevant_items = set(relevant)
    if not relevant_items:
        return 0.0

    for rank, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant_items:
            return 1.0 / rank
    return 0.0


def recall_at_k(recommended: Sequence[ItemId], relevant: Iterable[ItemId], k: int) -> float:
    """Compute Recall@K for a single user."""

    _validate_k(k)
    relevant_items = set(relevant)
    if not relevant_items:
        return 0.0
    hits = len(set(recommended[:k]) & relevant_items)
    return hits / len(relevant_items)


def evaluate_ranking_batch(
    recommendations_by_user: Mapping[ItemId, Sequence[ItemId]],
    relevant_by_user: Mapping[ItemId, Iterable[ItemId]],
    *,
    k: int,
) -> RankingMetrics:
    """Average ranking metrics over users with relevance labels."""

    _validate_k(k)
    if not relevant_by_user:
        return RankingMetrics(ndcg=0.0, hit=0.0, mrr=0.0, recall=0.0, users=0)

    ndcg_total = 0.0
    hit_total = 0.0
    mrr_total = 0.0
    recall_total = 0.0

    for user_id, relevant in relevant_by_user.items():
        relevant_items = set(relevant)
        recommended = recommendations_by_user.get(user_id, [])
        ndcg_total += ndcg_at_k(recommended, relevant_items, k)
        hit_total += hit_at_k(recommended, relevant_items, k)
        mrr_total += mrr_at_k(recommended, relevant_items, k)
        recall_total += recall_at_k(recommended, relevant_items, k)

    user_count = len(relevant_by_user)
    return RankingMetrics(
        ndcg=ndcg_total / user_count,
        hit=hit_total / user_count,
        mrr=mrr_total / user_count,
        recall=recall_total / user_count,
        users=user_count,
    )


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def sample_negative_items(
    *,
    all_item_ids: Sequence[ItemId],
    positive_items: Iterable[ItemId],
    n_negatives: int,
    seed: int = 7,
) -> list[ItemId]:
    """Sample negative item IDs not present in positive_items."""

    import random

    _validate_k(n_negatives)
    positive_set = set(positive_items)
    pool = [item_id for item_id in all_item_ids if item_id not in positive_set]
    if len(pool) < n_negatives:
        raise ValueError("not enough negative items to sample")
    rng = random.Random(seed)
    return rng.sample(pool, n_negatives)


def evaluate_sampled_ranking_batch(
    score_items: callable,
    relevant_by_user: Mapping[ItemId, Iterable[ItemId]],
    *,
    all_item_ids: Sequence[ItemId],
    k: int,
    n_negatives: int = 100,
    seed: int = 7,
) -> RankingMetrics:
    """Evaluate users by ranking each positive against sampled negatives.

    `score_items(user_id, candidate_item_ids)` must return scores aligned with
    candidate_item_ids, where larger scores rank higher.
    """

    _validate_k(k)
    recommendations_by_user = {}
    normalized_relevant = {}
    for offset, (user_id, relevant_items_iter) in enumerate(relevant_by_user.items()):
        relevant_items = list(relevant_items_iter)
        if not relevant_items:
            continue
        positive = relevant_items[0]
        negatives = sample_negative_items(
            all_item_ids=all_item_ids,
            positive_items={positive},
            n_negatives=n_negatives,
            seed=seed + offset,
        )
        candidates = [positive] + negatives
        scores = score_items(user_id, candidates)
        if len(scores) != len(candidates):
            raise ValueError("score_items must return one score per candidate")
        ranked = [item_id for _, item_id in sorted(zip(scores, candidates, strict=True), reverse=True)]
        recommendations_by_user[user_id] = ranked
        normalized_relevant[user_id] = {positive}
    return evaluate_ranking_batch(recommendations_by_user, normalized_relevant, k=k)


def sample_popularity_negative_items(
    *,
    popularity_counts: Mapping[ItemId, int],
    positive_items: Iterable[ItemId],
    n_negatives: int,
    seed: int = 7,
) -> list[ItemId]:
    """Sample negative item IDs according to item popularity."""

    import random

    _validate_k(n_negatives)
    positive_set = set(positive_items)
    pool = [item_id for item_id, count in popularity_counts.items() if item_id not in positive_set and count > 0]
    if len(pool) < n_negatives:
        raise ValueError("not enough negative items to sample")
    weights = [float(popularity_counts[item_id]) for item_id in pool]
    rng = random.Random(seed)
    sampled: list[ItemId] = []
    sampled_set: set[ItemId] = set()
    while len(sampled) < n_negatives:
        item_id = rng.choices(pool, weights=weights, k=1)[0]
        if item_id not in sampled_set:
            sampled.append(item_id)
            sampled_set.add(item_id)
    return sampled


def evaluate_popularity_sampled_ranking_batch(
    score_items: callable,
    relevant_by_user: Mapping[ItemId, Iterable[ItemId]],
    *,
    popularity_counts: Mapping[ItemId, int],
    k: int,
    n_negatives: int = 100,
    seed: int = 7,
) -> RankingMetrics:
    """Evaluate ranking with popularity-sampled negatives."""

    _validate_k(k)
    recommendations_by_user = {}
    normalized_relevant = {}
    for offset, (user_id, relevant_items_iter) in enumerate(relevant_by_user.items()):
        relevant_items = list(relevant_items_iter)
        if not relevant_items:
            continue
        positive = relevant_items[0]
        negatives = sample_popularity_negative_items(
            popularity_counts=popularity_counts,
            positive_items={positive},
            n_negatives=n_negatives,
            seed=seed + offset,
        )
        candidates = [positive] + negatives
        scores = score_items(user_id, candidates)
        if len(scores) != len(candidates):
            raise ValueError("score_items must return one score per candidate")
        ranked = [item_id for _, item_id in sorted(zip(scores, candidates, strict=True), reverse=True)]
        recommendations_by_user[user_id] = ranked
        normalized_relevant[user_id] = {positive}
    return evaluate_ranking_batch(recommendations_by_user, normalized_relevant, k=k)
