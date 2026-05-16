import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
from torch import nn

from data.preprocess import preprocess_dataset
from data.synthetic import generate_synthetic_dataset
from models.two_tower import (
    DEFAULT_EMBEDDING_DIM,
    ItemTower,
    UserTower,
    in_batch_negative_loss,
    train_two_tower_step,
    train_two_tower_epoch,
    validation_recall_at_k,
)


def test_user_and_item_towers_return_normalized_128_dim_embeddings() -> None:
    user_tower = UserTower(n_users=4)
    item_tower = ItemTower(n_items=6)

    user_embeddings = user_tower(torch.tensor([0, 1, 2]))
    item_embeddings = item_tower(torch.tensor([0, 1, 2, 3]))

    assert user_embeddings.shape == (3, DEFAULT_EMBEDDING_DIM)
    assert item_embeddings.shape == (4, DEFAULT_EMBEDDING_DIM)
    assert torch.allclose(torch.linalg.norm(user_embeddings, dim=1), torch.ones(3), atol=1e-6)
    assert torch.allclose(torch.linalg.norm(item_embeddings, dim=1), torch.ones(4), atol=1e-6)


def test_in_batch_negative_loss_matches_cross_entropy() -> None:
    user_embeddings = torch.eye(3)
    item_embeddings = torch.eye(3)

    loss = in_batch_negative_loss(user_embeddings, item_embeddings, temperature=0.07)
    expected_logits = user_embeddings @ item_embeddings.T / 0.07
    expected_loss = nn.functional.cross_entropy(expected_logits, torch.arange(3))

    assert loss.item() == pytest.approx(expected_loss.item())


def test_train_two_tower_step_updates_parameters_on_synthetic_batch() -> None:
    dataset = generate_synthetic_dataset(seed=11, n_users=6, n_items=12)
    artifacts = preprocess_dataset(
        dataset.interactions,
        dataset.items,
        user_min_interactions=3,
        item_min_interactions=1,
    )
    batch = artifacts.train[:4]
    user_ids = torch.tensor([row["user_id"] for row in batch], dtype=torch.long)
    item_ids = torch.tensor([row["item_id"] for row in batch], dtype=torch.long)
    user_tower = UserTower(n_users=artifacts.stats["users"])
    item_tower = ItemTower(n_items=artifacts.stats["items"])
    optimizer = torch.optim.AdamW(
        list(user_tower.parameters()) + list(item_tower.parameters()),
        lr=1e-3,
    )
    before = user_tower.embedding.weight.detach().clone()

    result = train_two_tower_step(user_tower, item_tower, optimizer, user_ids, item_ids)

    assert result.loss > 0
    assert not torch.equal(before, user_tower.embedding.weight.detach())


def test_train_two_tower_epoch_smoke() -> None:
    user_tower = UserTower(n_users=3)
    item_tower = ItemTower(n_items=4)
    optimizer = torch.optim.AdamW(list(user_tower.parameters()) + list(item_tower.parameters()), lr=1e-3)

    result = train_two_tower_epoch(
        user_tower,
        item_tower,
        optimizer,
        user_ids=[0, 1, 2, 0],
        positive_item_ids=[1, 2, 3, 0],
        batch_size=2,
    )

    assert result.batches == 2
    assert result.loss > 0


def test_validation_recall_at_k_uses_existing_recall_metric() -> None:
    user_tower = _LookupTower(
        {
            0: [1.0, 0.0, 0.0],
            1: [0.0, 1.0, 0.0],
        }
    )
    item_tower = _LookupTower(
        {
            0: [0.0, 0.0, 1.0],
            1: [1.0, 0.0, 0.0],
            2: [0.0, 1.0, 0.0],
        }
    )

    recall = validation_recall_at_k(
        user_tower,
        item_tower,
        user_ids=[0, 1],
        relevant_item_ids=[1, 2],
        all_item_ids=[0, 1, 2],
        k=1,
    )

    assert recall == pytest.approx(1.0)


def test_two_tower_validation_checks_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        validation_recall_at_k(
            UserTower(n_users=1),
            ItemTower(n_items=1),
            user_ids=[0, 0],
            relevant_item_ids=[0],
            all_item_ids=[0],
            k=1,
        )


class _LookupTower(nn.Module):
    def __init__(self, vectors: dict[int, list[float]]) -> None:
        super().__init__()
        ordered_vectors = [vectors[index] for index in sorted(vectors)]
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(ordered_vectors, dtype=torch.float32),
            freeze=True,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.embedding(ids), p=2, dim=-1)
