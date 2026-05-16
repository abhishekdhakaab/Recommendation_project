"""Ablation-study helpers for SeqRec evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Mapping, Sequence

from evaluation.offline_eval import RankingMetrics, evaluate_ranking_batch


@dataclass(frozen=True)
class AblationExperiment:
    """One ablation experiment definition."""

    name: str
    recommend: Callable[[Hashable], Sequence[Hashable]]


@dataclass(frozen=True)
class AblationResult:
    """Metrics for one ablation experiment."""

    name: str
    metrics: RankingMetrics


def run_ablation_study(
    experiments: Sequence[AblationExperiment],
    users: Sequence[Hashable],
    relevant_by_user: Mapping[Hashable, Iterable[Hashable]],
    *,
    k: int,
) -> list[AblationResult]:
    """Run configured recommenders and evaluate each with ranking metrics."""

    results = []
    for experiment in experiments:
        recommendations = {user_id: list(experiment.recommend(user_id)) for user_id in users}
        results.append(
            AblationResult(
                name=experiment.name,
                metrics=evaluate_ranking_batch(recommendations, relevant_by_user, k=k),
            )
        )
    return results
