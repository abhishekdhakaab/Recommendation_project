"""Dependency-injected recommendation pipeline interfaces for serving."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import time
from typing import Awaitable, Callable, Protocol, Sequence


@dataclass(frozen=True)
class Recommendation:
    """Recommendation returned by the online pipeline."""

    item_id: str
    title: str
    score: float
    stage: str


@dataclass(frozen=True)
class Candidate:
    """Intermediate candidate passed between serving stages."""

    item_id: str
    title: str
    text: str
    score: float
    stage: str


@dataclass(frozen=True)
class PipelineResult:
    """Full recommendation result with per-stage latency breakdown."""

    user_id: str
    cold_start: bool
    recommendations: list[Recommendation]
    latency_ms: dict[str, float]


class RecommendationPipeline(Protocol):
    """Protocol implemented by real and fake serving pipelines."""

    async def recommend(self, *, user_id: str, n_items: int, context: str | None = None) -> PipelineResult:
        """Return recommendations for a user."""

    async def record_feedback(self, *, user_id: str, item_id: str, interaction: str) -> None:
        """Record simulated feedback for online evaluation."""


class UnconfiguredRecommendationPipeline:
    """Placeholder used until trained models and stores are wired in."""

    async def recommend(self, *, user_id: str, n_items: int, context: str | None = None) -> PipelineResult:
        raise RuntimeError("Recommendation pipeline is not configured")

    async def record_feedback(self, *, user_id: str, item_id: str, interaction: str) -> None:
        raise RuntimeError("Recommendation pipeline is not configured")


FeatureFetcher = Callable[[str], object | Awaitable[object]]
Retriever = Callable[[str, int, object], Sequence[Candidate] | Awaitable[Sequence[Candidate]]]
Rescorer = Callable[[object, Sequence[Candidate], int], Sequence[Candidate] | Awaitable[Sequence[Candidate]]]
Reranker = Callable[[object, Sequence[Candidate], int], Sequence[Candidate] | Awaitable[Sequence[Candidate]]]
FeedbackSink = Callable[[str, str, str], None | Awaitable[None]]


class MultiStageRecommendationPipeline:
    """Injectable orchestration of feature fetch, retrieval, rescoring, and reranking."""

    def __init__(
        self,
        *,
        feature_fetcher: FeatureFetcher,
        retriever: Retriever,
        rescorer: Rescorer,
        reranker: Reranker,
        feedback_sink: FeedbackSink | None = None,
        retrieval_k: int = 500,
        rescore_k: int = 50,
    ) -> None:
        if retrieval_k < 1 or rescore_k < 1:
            raise ValueError("retrieval_k and rescore_k must be at least 1")
        self.feature_fetcher = feature_fetcher
        self.retriever = retriever
        self.rescorer = rescorer
        self.reranker = reranker
        self.feedback_sink = feedback_sink
        self.retrieval_k = retrieval_k
        self.rescore_k = rescore_k

    async def recommend(self, *, user_id: str, n_items: int, context: str | None = None) -> PipelineResult:
        started = time.perf_counter()
        latency_ms: dict[str, float] = {}

        features, latency_ms["feature_fetch"] = await _timed(self.feature_fetcher(user_id))
        retrieved, latency_ms["retrieval"] = await _timed(self.retriever(user_id, self.retrieval_k, features))
        rescored, latency_ms["rescoring"] = await _timed(self.rescorer(features, list(retrieved), self.rescore_k))
        reranked, latency_ms["reranking"] = await _timed(self.reranker(features, list(rescored), n_items))
        latency_ms["total"] = (time.perf_counter() - started) * 1000

        cold_start = bool(getattr(features, "cold_start_flag", False))
        return PipelineResult(
            user_id=user_id,
            cold_start=cold_start,
            recommendations=[
                Recommendation(
                    item_id=candidate.item_id,
                    title=candidate.title,
                    score=candidate.score,
                    stage=candidate.stage,
                )
                for candidate in list(reranked)[:n_items]
            ],
            latency_ms=latency_ms,
        )

    async def record_feedback(self, *, user_id: str, item_id: str, interaction: str) -> None:
        if self.feedback_sink is None:
            return
        await _maybe_await(self.feedback_sink(user_id, item_id, interaction))


async def _timed(value: object | Awaitable[object]) -> tuple[object, float]:
    started = time.perf_counter()
    result = await _maybe_await(value)
    return result, (time.perf_counter() - started) * 1000


async def _maybe_await(value: object | Awaitable[object]) -> object:
    if inspect.isawaitable(value):
        return await value
    return value
