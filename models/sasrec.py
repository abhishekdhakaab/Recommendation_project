"""SASRec sequential re-scoring model for SeqRec."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_MAX_SEQ_LEN = 50
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_NUM_HEADS = 2
DEFAULT_NUM_LAYERS = 2
DEFAULT_FEEDFORWARD_DIM = 256
DEFAULT_DROPOUT = 0.1
DEFAULT_NEGATIVES_PER_POSITIVE = 100


class SASRec(nn.Module):
    """Transformer-based sequential recommendation model.

    Item ID ``0`` is reserved for padding. Real item IDs are expected to be in
    the inclusive range ``1..n_items``.

    Optional text enrichment: pass ``text_embedding_dim`` to the constructor,
    then call ``set_text_embeddings(tensor)`` with a (n_items+1, text_dim) tensor
    (row 0 = zero-vector for the padding token). The model learns a linear
    projection from text space into embedding space and adds it to the ID
    embedding before the transformer layers — exactly the approach used by
    Pinterest, Spotify, and LinkedIn to bridge semantic and behavioral signals.
    """

    def __init__(
        self,
        n_items: int,
        *,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        feedforward_dim: int = DEFAULT_FEEDFORWARD_DIM,
        dropout: float = DEFAULT_DROPOUT,
        text_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        _validate_positive("n_items", n_items)
        _validate_positive("max_seq_len", max_seq_len)
        _validate_positive("embedding_dim", embedding_dim)
        _validate_positive("num_heads", num_heads)
        _validate_positive("num_layers", num_layers)
        _validate_positive("feedforward_dim", feedforward_dim)
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")
        if text_embedding_dim is not None and text_embedding_dim < 1:
            raise ValueError("text_embedding_dim must be at least 1")

        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Optional text projection: text_dim → embedding_dim (no bias so the
        # learned ID embedding retains full control over the origin point).
        self.text_proj: nn.Linear | None = (
            nn.Linear(text_embedding_dim, embedding_dim, bias=False)
            if text_embedding_dim is not None
            else None
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def set_text_embeddings(self, text_embeddings: Tensor) -> None:
        """Register pre-computed text embeddings for item text enrichment.

        ``text_embeddings`` must have shape ``(n_items + 1, text_embedding_dim)``
        where row 0 is a zero-vector for the padding token. The tensor is
        registered as a module buffer so it moves automatically with ``.to()``.
        """
        if self.text_proj is None:
            raise ValueError(
                "text_embedding_dim was not set at construction time — "
                "rebuild the model with text_embedding_dim set to use text enrichment."
            )
        expected_rows = self.n_items + 1
        if text_embeddings.shape[0] != expected_rows:
            raise ValueError(
                f"text_embeddings must have {expected_rows} rows (n_items+1); "
                f"got {text_embeddings.shape[0]}"
            )
        if text_embeddings.shape[1] != self.text_embedding_dim:
            raise ValueError(
                f"text_embeddings must have {self.text_embedding_dim} columns; "
                f"got {text_embeddings.shape[1]}"
            )
        self.register_buffer("text_emb", text_embeddings.float())

    def forward(self, item_sequences: Tensor) -> Tensor:
        """Encode item histories into contextual sequence embeddings."""

        sequences = self._validate_item_sequences(item_sequences)
        batch_size, seq_len = sequences.shape
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, seq_len)
        padding_mask = sequences.eq(0)

        hidden = self.item_embedding(sequences) + self.positional_embedding(positions)

        # Text enrichment: add projected BGE embedding to the ID embedding.
        if self.text_proj is not None and hasattr(self, "text_emb") and self.text_emb is not None:
            text_feats = self.text_proj(self.text_emb[sequences])
            hidden = hidden + text_feats

        hidden = self.transformer(
            hidden,
            mask=build_causal_mask(seq_len, device=sequences.device),
            src_key_padding_mask=padding_mask,
        )
        hidden = self.output_norm(hidden)
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def sequence_embedding(self, item_sequences: Tensor) -> Tensor:
        """Return the contextual embedding at the last sequence position.

        Training and evaluation use left-padded sequences [0,0,...,i1,i2,iN],
        so the last position always holds the most-recent item's representation.
        """
        encoded = self.forward(item_sequences)
        return encoded[:, -1, :]

    def score_candidates(self, item_sequences: Tensor, candidate_item_ids: Tensor) -> Tensor:
        """Score candidate items against each user's sequence embedding."""

        sequences = self._validate_item_sequences(item_sequences)
        candidates = self._validate_candidate_item_ids(candidate_item_ids, batch_size=sequences.shape[0])
        sequence_embeddings = self.sequence_embedding(sequences)
        candidate_embeddings = self.item_embedding(candidates)
        return torch.einsum("bd,bkd->bk", sequence_embeddings, candidate_embeddings)

    def _validate_item_sequences(self, item_sequences: Tensor) -> Tensor:
        if item_sequences.ndim != 2:
            raise ValueError("item_sequences must be a 2D tensor")
        if item_sequences.shape[1] > self.max_seq_len:
            raise ValueError("item_sequences length exceeds max_seq_len")
        sequences = item_sequences.to(dtype=torch.long)
        if torch.any(sequences < 0) or torch.any(sequences > self.n_items):
            raise ValueError("item_sequences must contain IDs in the range 0..n_items")
        return sequences

    def _validate_candidate_item_ids(self, candidate_item_ids: Tensor, *, batch_size: int) -> Tensor:
        if candidate_item_ids.ndim == 1:
            candidates = candidate_item_ids.unsqueeze(0).expand(batch_size, -1)
        elif candidate_item_ids.ndim == 2:
            candidates = candidate_item_ids
        else:
            raise ValueError("candidate_item_ids must be a 1D or 2D tensor")

        candidates = candidates.to(dtype=torch.long)
        if candidates.shape[0] != batch_size:
            raise ValueError("candidate_item_ids batch size must match item_sequences")
        if candidates.shape[1] == 0:
            raise ValueError("candidate_item_ids must include at least one candidate")
        if torch.any(candidates < 0) or torch.any(candidates > self.n_items):
            raise ValueError("candidate_item_ids must contain IDs in the range 0..n_items")
        return candidates


def score_candidates(model: SASRec, item_sequences: Tensor, candidate_item_ids: Tensor) -> Tensor:
    """Convenience wrapper for SASRec candidate scoring."""

    return model.score_candidates(item_sequences, candidate_item_ids)


def sampled_bce_loss(
    model: SASRec,
    item_sequences: Tensor,
    positive_item_ids: Tensor,
    negative_item_ids: Tensor,
) -> Tensor:
    """Compute sampled BCE loss for SASRec next-item prediction."""

    positives = positive_item_ids.to(dtype=torch.long)
    negatives = negative_item_ids.to(dtype=torch.long)
    if positives.ndim != 1:
        raise ValueError("positive_item_ids must be a 1D tensor")
    if negatives.ndim != 2:
        raise ValueError("negative_item_ids must be a 2D tensor")
    if negatives.shape[0] != positives.shape[0]:
        raise ValueError("negative_item_ids batch size must match positive_item_ids")

    positive_scores = model.score_candidates(item_sequences, positives.unsqueeze(1)).squeeze(1)
    negative_scores = model.score_candidates(item_sequences, negatives)
    logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
    labels = torch.zeros_like(logits)
    labels[:, 0] = 1.0
    return F.binary_cross_entropy_with_logits(logits, labels)


def train_sasrec_step(
    model: SASRec,
    optimizer: torch.optim.Optimizer,
    item_sequences: Tensor,
    positive_item_ids: Tensor,
    negative_item_ids: Tensor,
) -> float:
    """Run one SASRec optimizer step with sampled negatives."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = sampled_bce_loss(model, item_sequences, positive_item_ids, negative_item_ids)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def build_causal_mask(seq_len: int, *, device: torch.device | str | None = None) -> Tensor:
    """Return a boolean causal mask where True blocks future attention."""

    _validate_positive("seq_len", seq_len)
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
