"""Simple interaction-replay A/B simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Sequence


@dataclass(frozen=True)
class ReplayEvent:
    """One replayed online interaction."""

    user_id: Hashable
    item_id: Hashable


@dataclass(frozen=True)
class ABSimulationResult:
    """Hit-rate summary for control and treatment recommenders."""

    control_hit_rate: float
    treatment_hit_rate: float
    relative_lift: float
    events: int


def simulate_ab_test(
    events: Sequence[ReplayEvent],
    control_recommender: Callable[[Hashable], Sequence[Hashable]],
    treatment_recommender: Callable[[Hashable], Sequence[Hashable]],
    *,
    k: int = 10,
) -> ABSimulationResult:
    """Replay interactions and compare Hit@K between two recommenders."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if not events:
        return ABSimulationResult(control_hit_rate=0.0, treatment_hit_rate=0.0, relative_lift=0.0, events=0)

    control_hits = 0
    treatment_hits = 0
    for event in events:
        control_hits += int(event.item_id in set(control_recommender(event.user_id)[:k]))
        treatment_hits += int(event.item_id in set(treatment_recommender(event.user_id)[:k]))

    control_rate = control_hits / len(events)
    treatment_rate = treatment_hits / len(events)
    relative_lift = 0.0 if control_rate == 0 else (treatment_rate - control_rate) / control_rate
    return ABSimulationResult(
        control_hit_rate=control_rate,
        treatment_hit_rate=treatment_rate,
        relative_lift=relative_lift,
        events=len(events),
    )
