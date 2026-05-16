"""Two-tower retrieval model for SeqRec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from evaluation.offline_eval import recall_at_k


DEFAULT_EMBEDDING_DIM = 128
DEFAULT_HIDDEN_DIM = 256
DEFAULT_TEMPERATURE = 0.07


@dataclass(frozen=True)
class TrainingStepResult:
    """Metrics returned by one retrieval training step."""

    loss: float


@dataclass(frozen=True)
class EpochResult:
    """Aggregated metrics returned by one retrieval training epoch."""

    loss: float
    batches: int


class UserTower(nn.Module):
    """User encoder from the PRD two-tower architecture."""

    def __init__(
        self,
        n_users: int,
        *,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        _validate_count("n_users", n_users)
        self.embedding = nn.Embedding(n_users, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, user_ids: Tensor) -> Tensor:
        embeddings = self.mlp(self.embedding(user_ids))
        return F.normalize(embeddings, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item encoder from the PRD two-tower architecture."""

    def __init__(
        self,
        n_items: int,
        *,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        _validate_count("n_items", n_items)
        self.embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, item_ids: Tensor) -> Tensor:
        embeddings = self.mlp(self.embedding(item_ids))
        return F.normalize(embeddings, p=2, dim=-1)


def in_batch_negative_loss(
    user_embeddings: Tensor,
    positive_item_embeddings: Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tensor:
    """Compute cross-entropy loss with in-batch negatives."""

    _validate_temperature(temperature)
    if user_embeddings.ndim != 2 or positive_item_embeddings.ndim != 2:
        raise ValueError("user_embeddings and positive_item_embeddings must be 2D tensors")
    if user_embeddings.shape != positive_item_embeddings.shape:
        raise ValueError("user_embeddings and positive_item_embeddings must have the same shape")

    logits = user_embeddings @ positive_item_embeddings.T
    logits = logits / temperature
    labels = torch.arange(user_embeddings.shape[0], device=user_embeddings.device)
    return F.cross_entropy(logits, labels)


def train_two_tower_step(
    user_tower: UserTower,
    item_tower: ItemTower,
    optimizer: torch.optim.Optimizer,
    user_ids: Tensor,
    positive_item_ids: Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> TrainingStepResult:
    """Run one optimizer step for two-tower retrieval training."""

    user_tower.train()
    item_tower.train()
    optimizer.zero_grad(set_to_none=True)

    user_embeddings = user_tower(user_ids)
    item_embeddings = item_tower(positive_item_ids)
    loss = in_batch_negative_loss(user_embeddings, item_embeddings, temperature=temperature)
    loss.backward()
    optimizer.step()

    return TrainingStepResult(loss=float(loss.detach().cpu().item()))


def train_two_tower_epoch(
    user_tower: UserTower,
    item_tower: ItemTower,
    optimizer: torch.optim.Optimizer,
    user_ids: Sequence[int] | Tensor,
    positive_item_ids: Sequence[int] | Tensor,
    *,
    batch_size: int = 1024,
    temperature: float = DEFAULT_TEMPERATURE,
    device: torch.device | str | None = None,
) -> EpochResult:
    """Train one epoch over positive user-item interactions."""

    _validate_count("batch_size", batch_size)
    target_device = torch.device(device) if device is not None else _module_device(user_tower)
    users = _to_long_tensor(user_ids, target_device)
    items = _to_long_tensor(positive_item_ids, target_device)
    if users.shape[0] != items.shape[0]:
        raise ValueError("user_ids and positive_item_ids must have the same length")
    if users.numel() == 0:
        return EpochResult(loss=0.0, batches=0)

    total_loss = 0.0
    batches = 0
    for start in range(0, users.shape[0], batch_size):
        result = train_two_tower_step(
            user_tower,
            item_tower,
            optimizer,
            users[start : start + batch_size],
            items[start : start + batch_size],
            temperature=temperature,
        )
        total_loss += result.loss
        batches += 1
    return EpochResult(loss=total_loss / batches, batches=batches)


@torch.inference_mode()
def encode_all_items(
    item_tower: ItemTower,
    item_ids: Sequence[int] | Tensor,
    *,
    batch_size: int = 4096,
    device: torch.device | str | None = None,
) -> Tensor:
    """Encode item IDs into retrieval embeddings for FAISS indexing."""

    _validate_count("batch_size", batch_size)
    target_device = torch.device(device) if device is not None else _module_device(item_tower)
    ids = _to_long_tensor(item_ids, target_device)
    item_tower.eval()
    return torch.cat([item_tower(ids[start : start + batch_size]) for start in range(0, ids.shape[0], batch_size)])


def save_two_tower_checkpoint(
    path: str | Path,
    user_tower: UserTower,
    item_tower: ItemTower,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save two-tower weights and optional run metadata."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "user_tower": user_tower.state_dict(),
            "item_tower": item_tower.state_dict(),
            "metadata": metadata or {},
        },
        output_path,
    )


@torch.inference_mode()
def validation_recall_at_k(
    user_tower: nn.Module,
    item_tower: nn.Module,
    user_ids: Sequence[int] | Tensor,
    relevant_item_ids: Sequence[int],
    all_item_ids: Sequence[int] | Tensor,
    *,
    k: int,
    device: torch.device | str | None = None,
) -> float:
    """Evaluate mean Recall@K by scoring every candidate item for each user."""

    if len(user_ids) != len(relevant_item_ids):
        raise ValueError("user_ids and relevant_item_ids must have the same length")
    if not relevant_item_ids:
        return 0.0

    target_device = torch.device(device) if device is not None else _module_device(user_tower)
    user_tensor = _to_long_tensor(user_ids, target_device)
    item_tensor = _to_long_tensor(all_item_ids, target_device)

    if item_tensor.numel() == 0:
        return 0.0

    was_user_training = user_tower.training
    was_item_training = item_tower.training
    user_tower.eval()
    item_tower.eval()
    try:
        user_embeddings = user_tower(user_tensor)
        item_embeddings = item_tower(item_tensor)
        scores = user_embeddings @ item_embeddings.T
        top_k = min(k, item_tensor.numel())
        top_indices = torch.topk(scores, k=top_k, dim=1).indices.cpu().tolist()
        candidate_ids = item_tensor.cpu().tolist()

        total_recall = 0.0
        for row_indices, relevant_item_id in zip(top_indices, relevant_item_ids, strict=True):
            recommended = [candidate_ids[index] for index in row_indices]
            total_recall += recall_at_k(recommended, {int(relevant_item_id)}, k)
        return total_recall / len(relevant_item_ids)
    finally:
        user_tower.train(was_user_training)
        item_tower.train(was_item_training)


def _to_long_tensor(values: Sequence[int] | Tensor, device: torch.device) -> Tensor:
    if isinstance(values, Tensor):
        return values.to(device=device, dtype=torch.long)
    return torch.tensor(list(values), dtype=torch.long, device=device)


def _module_device(module: nn.Module) -> torch.device:
    first_parameter = next(module.parameters(), None)
    if first_parameter is None:
        return torch.device("cpu")
    return first_parameter.device


def _validate_count(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _validate_temperature(value: float) -> None:
    if value <= 0:
        raise ValueError("temperature must be positive")
