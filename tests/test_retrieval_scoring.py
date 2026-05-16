import pytest

from retrieval.scoring import blend_with_popularity_prior


def test_blend_with_popularity_prior_boosts_popular_items() -> None:
    scores = blend_with_popularity_prior([0.0, 0.0], [1, 2], {1: 0, 2: 9}, popularity_weight=1.0)

    assert scores[1] > scores[0]


def test_blend_with_popularity_prior_validates_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        blend_with_popularity_prior([1.0], [1, 2], {}, popularity_weight=1.0)
