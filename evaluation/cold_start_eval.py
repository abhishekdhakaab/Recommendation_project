"""Cold-start versus warm-start evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence

from evaluation.offline_eval import RankingMetrics, evaluate_ranking_batch


@dataclass(frozen=True)
class ColdWarmMetrics:
    """Metrics split by cold and warm users."""

    cold_start: RankingMetrics
    warm_start: RankingMetrics


def split_users_by_interaction_count(
    interaction_counts: Mapping[Hashable, int],
    *,
    cold_max_interactions: int = 3,
    warm_min_interactions: int = 10,
) -> tuple[set[Hashable], set[Hashable]]:
    """Return cold and warm user ID sets using PRD thresholds."""

    cold_users = {user_id for user_id, count in interaction_counts.items() if count <= cold_max_interactions}
    warm_users = {user_id for user_id, count in interaction_counts.items() if count >= warm_min_interactions}
    return cold_users, warm_users


def evaluate_cold_warm_split(
    recommendations_by_user: Mapping[Hashable, Sequence[Hashable]],
    relevant_by_user: Mapping[Hashable, Iterable[Hashable]],
    interaction_counts: Mapping[Hashable, int],
    *,
    k: int,
) -> ColdWarmMetrics:
    """Evaluate ranking metrics separately for cold-start and warm users."""

    cold_users, warm_users = split_users_by_interaction_count(interaction_counts)
    return ColdWarmMetrics(
        cold_start=evaluate_ranking_batch(
            _select_users(recommendations_by_user, cold_users),
            _select_users(relevant_by_user, cold_users),
            k=k,
        ),
        warm_start=evaluate_ranking_batch(
            _select_users(recommendations_by_user, warm_users),
            _select_users(relevant_by_user, warm_users),
            k=k,
        ),
    )


def _select_users(mapping: Mapping[Hashable, object], users: set[Hashable]) -> dict[Hashable, object]:
    return {user_id: value for user_id, value in mapping.items() if user_id in users}
