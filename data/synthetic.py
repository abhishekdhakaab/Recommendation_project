"""Tiny deterministic datasets for SeqRec tests and smoke checks."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for generated recommendation test data."""

    interactions: list[dict[str, int | float | str]]
    items: list[dict[str, int | str]]


def generate_synthetic_dataset(
    *,
    seed: int = 7,
    n_users: int = 6,
    n_items: int = 12,
    start_timestamp: int = 1_700_000_000,
) -> SyntheticDataset:
    """Generate a tiny deterministic recommendation dataset.

    The generated data intentionally contains both cold-start users with at
    most three interactions and warm users with at least five interactions.
    It is small enough for tests while still preserving user-item-timestamp
    structure for future preprocessing smoke checks.
    """

    if n_users < 2:
        raise ValueError("n_users must be at least 2 to include cold and warm users")
    if n_items < 5:
        raise ValueError("n_items must be at least 5")

    rng = random.Random(seed)
    categories = ["Beauty", "Sports", "Home"]

    items = [
        {
            "item_id": item_id,
            "raw_item_id": f"I{item_id:04d}",
            "title": f"Synthetic Item {item_id}",
            "category": categories[item_id % len(categories)],
            "description": f"Small synthetic catalog item {item_id}.",
        }
        for item_id in range(n_items)
    ]

    interactions: list[dict[str, int | float | str]] = []
    for user_id in range(n_users):
        interaction_count = _interaction_count_for_user(user_id)
        timestamp = start_timestamp + user_id * 10_000
        used_items = rng.sample(range(n_items), k=min(interaction_count, n_items))

        for position, item_id in enumerate(used_items):
            interactions.append(
                {
                    "user_id": user_id,
                    "raw_user_id": f"U{user_id:04d}",
                    "item_id": item_id,
                    "raw_item_id": f"I{item_id:04d}",
                    "rating": float(rng.randint(3, 5)),
                    "timestamp": timestamp + position * 60,
                }
            )

    interactions.sort(key=lambda row: (int(row["user_id"]), int(row["timestamp"])))
    return SyntheticDataset(interactions=interactions, items=items)


def _interaction_count_for_user(user_id: int) -> int:
    if user_id == 0:
        return 0
    if user_id == 1:
        return 2
    if user_id == 2:
        return 3
    return 5 + (user_id % 3)
