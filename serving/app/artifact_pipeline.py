"""Artifact-backed serving pipeline for local SeqRec demos."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time

from retrieval.sequence_retrieval import SequenceRetrievalIndex, load_sequence_retrieval_index, recommend_from_history
from serving.app.pipeline import PipelineResult, Recommendation


@dataclass(frozen=True)
class ArtifactPipelineConfig:
    processed_dir: Path
    artifact_dir: Path
    retrieval_k: int = 500


class SequenceArtifactRecommendationPipeline:
    """Serve recommendations from a persisted sequence retrieval artifact."""

    def __init__(self, config: ArtifactPipelineConfig) -> None:
        if config.retrieval_k < 1:
            raise ValueError("retrieval_k must be at least 1")
        self.config = config
        self.index = load_sequence_retrieval_index(config.artifact_dir / "sequence_index.npz")
        self.items_by_id = _load_items(config.processed_dir / "item_metadata.jsonl")
        self.histories = _load_histories(config.artifact_dir / "user_histories.jsonl")
        self.popularity = _load_popularity(config.artifact_dir / "popularity.jsonl")
        self.popular_items = [item_id for item_id, _ in self.popularity.most_common()]
        self.feedback_events: list[dict[str, str]] = []

    async def recommend(self, *, user_id: str, n_items: int, context: str | None = None) -> PipelineResult:
        started = time.perf_counter()
        latency_ms: dict[str, float] = {}

        feature_started = time.perf_counter()
        history = self.histories.get(_parse_user_id(user_id), [])
        seen = set(history)
        cold_start = len(history) < 5
        latency_ms["feature_fetch"] = (time.perf_counter() - feature_started) * 1000

        retrieval_started = time.perf_counter()
        recommended_ids = self._retrieve(history=history, seen=seen, top_k=max(n_items, self.config.retrieval_k))
        latency_ms["retrieval"] = (time.perf_counter() - retrieval_started) * 1000

        rescore_started = time.perf_counter()
        recommendations = [self._recommendation(item_id, rank=index) for index, item_id in enumerate(recommended_ids[:n_items])]
        latency_ms["rescoring"] = (time.perf_counter() - rescore_started) * 1000
        latency_ms["reranking"] = 0.0
        latency_ms["total"] = (time.perf_counter() - started) * 1000

        return PipelineResult(
            user_id=user_id,
            cold_start=cold_start,
            recommendations=recommendations,
            latency_ms=latency_ms,
        )

    async def record_feedback(self, *, user_id: str, item_id: str, interaction: str) -> None:
        self.feedback_events.append({"user_id": user_id, "item_id": item_id, "interaction": interaction})

    def _retrieve(self, *, history: list[int], seen: set[int], top_k: int) -> list[int]:
        sequence_ids = recommend_from_history(self.index, history[-50:], candidates=self.index.item_ids, top_k=top_k + len(seen))
        ranked = [item_id for item_id in sequence_ids if item_id not in seen]
        if len(ranked) >= top_k:
            return ranked[:top_k]
        fallback = [item_id for item_id in self.popular_items if item_id not in seen and item_id not in ranked]
        return (ranked + fallback)[:top_k]

    def _recommendation(self, item_id: int, *, rank: int) -> Recommendation:
        row = self.items_by_id.get(item_id, {})
        title = str(row.get("title") or f"Item {item_id}")
        popularity_score = float(self.popularity.get(item_id, 0))
        score = popularity_score + 1.0 / (rank + 1)
        return Recommendation(
            item_id=str(item_id),
            title=title,
            score=score,
            stage="sequence_retrieval" if item_id in self.index.local_id else "popularity_fallback",
        )


def create_sequence_artifact_pipeline(
    *,
    processed_dir: str | Path = "data/processed/sports_and_outdoors",
    artifact_dir: str | Path = "models/artifacts/sports_and_outdoors_sequence",
    retrieval_k: int = 500,
) -> SequenceArtifactRecommendationPipeline:
    return SequenceArtifactRecommendationPipeline(
        ArtifactPipelineConfig(processed_dir=Path(processed_dir), artifact_dir=Path(artifact_dir), retrieval_k=retrieval_k)
    )


def _parse_user_id(user_id: str) -> int:
    try:
        return int(user_id)
    except ValueError:
        return -1


def _load_items(path: Path) -> dict[int, dict]:
    rows = _read_jsonl(path)
    return {int(row["item_id"]): row for row in rows}


def _load_histories(path: Path) -> dict[int, list[int]]:
    rows = _read_jsonl(path)
    return {int(row["user_id"]): [int(item_id) for item_id in row["history"]] for row in rows}


def _load_popularity(path: Path) -> Counter[int]:
    rows = _read_jsonl(path)
    return Counter({int(row["item_id"]): int(row["count"]) for row in rows})


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
