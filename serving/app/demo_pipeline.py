"""Artifact-backed demo pipeline for the local All_Beauty smoke dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Sequence

from serving.app.pipeline import Candidate, MultiStageRecommendationPipeline


def create_all_beauty_demo_pipeline(
    *,
    processed_dir: str | Path = "data/processed/all_beauty",
) -> MultiStageRecommendationPipeline:
    """Create a deterministic demo pipeline from processed local artifacts.

    This is intentionally lightweight: it uses the processed item metadata and
    histories to demonstrate the API contract without loading trained Torch or
    BGE models at web-server startup.
    """

    processed_path = Path(processed_dir)
    train = _read_jsonl(processed_path / "train.jsonl")
    validation = _read_jsonl(processed_path / "validation.jsonl")
    test = _read_jsonl(processed_path / "test.jsonl")
    items = _read_jsonl(processed_path / "item_metadata.jsonl")
    items_by_id = {int(row["item_id"]): row for row in items}
    all_interactions = train + validation + test
    histories: dict[str, list[int]] = defaultdict(list)
    popularity = Counter()
    for row in sorted(all_interactions, key=lambda value: int(value["timestamp"])):
        user_id = str(row["user_id"])
        item_id = int(row["item_id"])
        histories[user_id].append(item_id)
        popularity[item_id] += 1
    ranked_popular = [item_id for item_id, _ in popularity.most_common()] or sorted(items_by_id)

    class Features:
        def __init__(self, user_id: str) -> None:
            self.user_id = user_id
            self.interaction_history = histories.get(user_id, [])[-50:]
            self.interaction_count = len(histories.get(user_id, []))
            self.cold_start_flag = self.interaction_count < 5

    def fetch(user_id: str) -> Features:
        return Features(user_id)

    def retrieve(user_id: str, top_k: int, features: Features) -> list[Candidate]:
        seen = set(features.interaction_history)
        ranked = [item_id for item_id in ranked_popular if item_id not in seen] + [item_id for item_id in ranked_popular if item_id in seen]
        return [_candidate(item_id, items_by_id, score=float(popularity[item_id]), stage="retrieval") for item_id in ranked[:top_k]]

    def rescore(features: Features, candidates: Sequence[Candidate], top_k: int) -> list[Candidate]:
        history_set = set(features.interaction_history[-5:])
        rescored = [
            Candidate(
                item_id=candidate.item_id,
                title=candidate.title,
                text=candidate.text,
                score=candidate.score + (0.25 if int(candidate.item_id) in history_set else 0.0),
                stage="sasrec",
            )
            for candidate in candidates
        ]
        return sorted(rescored, key=lambda candidate: (-candidate.score, candidate.item_id))[:top_k]

    def rerank(features: Features, candidates: Sequence[Candidate], top_n: int) -> list[Candidate]:
        reranked = [
            Candidate(
                item_id=candidate.item_id,
                title=candidate.title,
                text=candidate.text,
                score=candidate.score,
                stage="llm_reranker",
            )
            for candidate in candidates
        ]
        return reranked[:top_n]

    return MultiStageRecommendationPipeline(
        feature_fetcher=fetch,
        retriever=retrieve,
        rescorer=rescore,
        reranker=rerank,
        retrieval_k=500,
        rescore_k=50,
    )


def _candidate(item_id: int, items_by_id: dict[int, dict], *, score: float, stage: str) -> Candidate:
    row = items_by_id[item_id]
    title = str(row.get("title", f"Item {item_id}"))
    text = " - ".join(part for part in [title, row.get("category", ""), row.get("description", "")] if part)
    return Candidate(item_id=str(item_id), title=title, text=text, score=score, stage=stage)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
