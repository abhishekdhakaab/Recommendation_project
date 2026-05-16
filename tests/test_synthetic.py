from collections import Counter, defaultdict

import pytest

from data.synthetic import generate_synthetic_dataset


def test_generate_synthetic_dataset_is_deterministic() -> None:
    first = generate_synthetic_dataset(seed=42)
    second = generate_synthetic_dataset(seed=42)

    assert first == second


def test_generate_synthetic_dataset_has_expected_fields() -> None:
    dataset = generate_synthetic_dataset(seed=42)

    assert dataset.interactions
    assert dataset.items
    assert {
        "user_id",
        "raw_user_id",
        "item_id",
        "raw_item_id",
        "rating",
        "timestamp",
    } <= dataset.interactions[0].keys()
    assert {"item_id", "raw_item_id", "title", "category", "description"} <= dataset.items[0].keys()


def test_generate_synthetic_dataset_is_chronological_per_user() -> None:
    dataset = generate_synthetic_dataset(seed=42)
    timestamps_by_user: dict[int, list[int]] = defaultdict(list)

    for row in dataset.interactions:
        timestamps_by_user[int(row["user_id"])].append(int(row["timestamp"]))

    for timestamps in timestamps_by_user.values():
        assert timestamps == sorted(timestamps)


def test_generate_synthetic_dataset_has_cold_and_warm_users() -> None:
    dataset = generate_synthetic_dataset(seed=42)
    interaction_counts = Counter(int(row["user_id"]) for row in dataset.interactions)

    assert any(count <= 3 for count in interaction_counts.values())
    assert any(count >= 5 for count in interaction_counts.values())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_users": 1}, "n_users must be at least 2"),
        ({"n_items": 4}, "n_items must be at least 5"),
    ],
)
def test_generate_synthetic_dataset_validates_size(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_synthetic_dataset(**kwargs)
