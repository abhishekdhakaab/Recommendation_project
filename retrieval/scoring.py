"""Retrieval score calibration helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def blend_with_popularity_prior(
    model_scores: Sequence[float],
    item_ids: Sequence[int],
    popularity_counts: Mapping[int, int],
    *,
    popularity_weight: float,
) -> list[float]:
    """Blend model scores with a log-popularity prior.

    The prior is intentionally simple and stable: log1p(interaction_count). This
    mirrors a common retrieval calibration trick without replacing the retrieval
    model itself.
    """

    if len(model_scores) != len(item_ids):
        raise ValueError("model_scores and item_ids must have the same length")
    return [
        float(score) + popularity_weight * math.log1p(float(popularity_counts.get(int(item_id), 0)))
        for score, item_id in zip(model_scores, item_ids, strict=True)
    ]
