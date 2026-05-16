"""Reranker wrappers for SeqRec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-large"
DEFAULT_RERANK_INPUT_K = 50
DEFAULT_RERANK_OUTPUT_K = 10


@dataclass(frozen=True)
class RerankCandidate:
    """Candidate item text passed to the reranker."""

    item_id: int
    text: str
    retrieval_score: float | None = None


@dataclass(frozen=True)
class RerankedCandidate:
    """Candidate item after reranking."""

    item_id: int
    text: str
    score: float
    retrieval_score: float | None = None


class TextReranker(Protocol):
    """Scores item texts against a user query text."""

    def score(self, user_query_text: str, item_texts: Sequence[str]) -> list[float]:
        """Return one relevance score per item text."""


class BGEReranker:
    """Lazy BGE cross-encoder reranker wrapper.

    The underlying model is imported and loaded only on first use, which keeps
    unit tests and lightweight imports from downloading large model weights.
    """

    def __init__(self, *, model_name: str = DEFAULT_BGE_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def score(self, user_query_text: str, item_texts: Sequence[str]) -> list[float]:
        if not user_query_text:
            raise ValueError("user_query_text must not be empty")
        if not item_texts:
            return []

        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

        pairs = [(user_query_text, item_text) for item_text in item_texts]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]


class MockReranker:
    """Small deterministic reranker for tests and smoke checks."""

    def __init__(self, scores_by_text: dict[str, float] | None = None) -> None:
        self.scores_by_text = scores_by_text or {}

    def score(self, user_query_text: str, item_texts: Sequence[str]) -> list[float]:
        if not user_query_text:
            raise ValueError("user_query_text must not be empty")
        return [float(self.scores_by_text.get(item_text, 0.0)) for item_text in item_texts]


def rerank_candidates(
    reranker: TextReranker,
    *,
    user_query_text: str,
    candidates: Sequence[RerankCandidate | dict[str, int | float | str | None]],
    top_n: int = DEFAULT_RERANK_OUTPUT_K,
) -> list[RerankedCandidate]:
    """Rerank candidates and return the top-N by relevance score."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    normalized_candidates = [_normalize_candidate(candidate) for candidate in candidates]
    if not normalized_candidates:
        return []

    item_texts = [candidate.text for candidate in normalized_candidates]
    scores = reranker.score(user_query_text, item_texts)
    if len(scores) != len(normalized_candidates):
        raise ValueError("reranker returned a different number of scores than candidates")

    reranked = [
        RerankedCandidate(
            item_id=candidate.item_id,
            text=candidate.text,
            score=float(score),
            retrieval_score=candidate.retrieval_score,
        )
        for candidate, score in zip(normalized_candidates, scores, strict=True)
    ]
    return sorted(reranked, key=lambda candidate: (-candidate.score, candidate.item_id))[:top_n]


def rerank_top_50_to_top_10(
    reranker: TextReranker,
    *,
    user_query_text: str,
    candidates: Sequence[RerankCandidate | dict[str, int | float | str | None]],
) -> list[RerankedCandidate]:
    """PRD helper: rerank up to top-50 candidates down to top-10."""

    return rerank_candidates(
        reranker,
        user_query_text=user_query_text,
        candidates=candidates[:DEFAULT_RERANK_INPUT_K],
        top_n=DEFAULT_RERANK_OUTPUT_K,
    )


def _normalize_candidate(candidate: RerankCandidate | dict[str, int | float | str | None]) -> RerankCandidate:
    if isinstance(candidate, RerankCandidate):
        return candidate
    return RerankCandidate(
        item_id=int(candidate["item_id"]),
        text=str(candidate["text"]),
        retrieval_score=(
            None if candidate.get("retrieval_score") is None else float(candidate["retrieval_score"])
        ),
    )
