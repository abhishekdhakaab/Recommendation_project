"""Sequence-aware retrieval based on item co-occurrence embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SequenceRetrievalIndex:
    """Top-item co-occurrence embedding index."""

    item_ids: list[int]
    embeddings: np.ndarray
    local_id: dict[int, int]


def build_sequence_retrieval_index(
    histories: Iterable[Sequence[int]],
    *,
    item_ids: Sequence[int],
    embedding_dim: int = 64,
    context_window: int = 10,
) -> SequenceRetrievalIndex:
    """Build normalized SVD item embeddings from co-occurrence histories."""

    if embedding_dim < 1:
        raise ValueError("embedding_dim must be at least 1")
    if context_window < 1:
        raise ValueError("context_window must be at least 1")
    ids = [int(item_id) for item_id in item_ids]
    local_id = {item_id: index for index, item_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=np.float32)
    for history in histories:
        filtered = [int(item_id) for item_id in history if int(item_id) in local_id]
        for index, item_id in enumerate(filtered):
            row = local_id[item_id]
            matrix[row, row] += 1.0
            left = max(0, index - context_window)
            right = min(len(filtered), index + context_window + 1)
            for neighbor in filtered[left:index] + filtered[index + 1 : right]:
                matrix[row, local_id[neighbor]] += 1.0
    matrix = np.log1p(matrix)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    actual_dim = min(embedding_dim, u.shape[1])
    embeddings = u[:, :actual_dim] * np.sqrt(singular_values[:actual_dim])
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return SequenceRetrievalIndex(item_ids=ids, embeddings=embeddings.astype(np.float32), local_id=local_id)


def save_sequence_retrieval_index(index: SequenceRetrievalIndex, path: str | Path) -> None:
    """Persist a sequence retrieval index as a compact NumPy artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        item_ids=np.asarray(index.item_ids, dtype=np.int64),
        embeddings=index.embeddings.astype(np.float32),
    )


def load_sequence_retrieval_index(path: str | Path) -> SequenceRetrievalIndex:
    """Load a sequence retrieval index saved by save_sequence_retrieval_index."""

    artifact = np.load(Path(path))
    item_ids = [int(item_id) for item_id in artifact["item_ids"].tolist()]
    embeddings = artifact["embeddings"].astype(np.float32)
    return SequenceRetrievalIndex(
        item_ids=item_ids,
        embeddings=embeddings,
        local_id={item_id: index for index, item_id in enumerate(item_ids)},
    )


def recommend_from_history(
    index: SequenceRetrievalIndex,
    history: Sequence[int],
    *,
    candidates: Sequence[int] | None = None,
    top_k: int = 50,
) -> list[int]:
    """Rank candidates by similarity to the mean recent-history item vector."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    local_history = [index.local_id[int(item_id)] for item_id in history if int(item_id) in index.local_id]
    if not local_history:
        return []
    candidate_ids = [int(item_id) for item_id in (candidates or index.item_ids) if int(item_id) in index.local_id]
    user_vector = index.embeddings[local_history].mean(axis=0)
    user_vector = user_vector / max(float(np.linalg.norm(user_vector)), 1e-12)
    candidate_local_ids = [index.local_id[item_id] for item_id in candidate_ids]
    scores = index.embeddings[candidate_local_ids] @ user_vector
    return [item_id for _, item_id in sorted(zip(scores.tolist(), candidate_ids, strict=True), reverse=True)[:top_k]]
