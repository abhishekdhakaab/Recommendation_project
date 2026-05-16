import pytest

from evaluation.ab_simulation import ReplayEvent, simulate_ab_test
from evaluation.ablation import AblationExperiment, run_ablation_study
from evaluation.cold_start_eval import evaluate_cold_warm_split, split_users_by_interaction_count


def test_cold_warm_split_and_metrics() -> None:
    cold_users, warm_users = split_users_by_interaction_count({"u1": 2, "u2": 5, "u3": 12})

    assert cold_users == {"u1"}
    assert warm_users == {"u3"}

    metrics = evaluate_cold_warm_split(
        {"u1": ["i1"], "u3": ["i9"]},
        {"u1": {"i1"}, "u3": {"i8"}},
        {"u1": 2, "u3": 12},
        k=1,
    )

    assert metrics.cold_start.hit == 1.0
    assert metrics.warm_start.hit == 0.0


def test_run_ablation_study() -> None:
    results = run_ablation_study(
        [
            AblationExperiment(name="control", recommend=lambda user_id: ["i1"]),
            AblationExperiment(name="treatment", recommend=lambda user_id: ["i2"]),
        ],
        users=["u1"],
        relevant_by_user={"u1": {"i2"}},
        k=1,
    )

    assert [result.name for result in results] == ["control", "treatment"]
    assert results[0].metrics.hit == 0.0
    assert results[1].metrics.hit == 1.0


def test_simulate_ab_test() -> None:
    result = simulate_ab_test(
        [ReplayEvent(user_id="u1", item_id="i2"), ReplayEvent(user_id="u2", item_id="i3")],
        control_recommender=lambda user_id: ["i1"],
        treatment_recommender=lambda user_id: ["i2", "i3"],
        k=2,
    )

    assert result.control_hit_rate == 0.0
    assert result.treatment_hit_rate == 1.0
    assert result.relative_lift == 0.0
    assert result.events == 2


def test_simulate_ab_test_validates_k() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        simulate_ab_test([], lambda user_id: [], lambda user_id: [], k=0)
