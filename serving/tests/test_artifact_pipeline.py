import asyncio
import json

from retrieval.build_sequence_index import build_sequence_index_artifact
from serving.app.artifact_pipeline import create_sequence_artifact_pipeline


def test_sequence_artifact_pipeline_returns_real_artifact_recommendations(tmp_path) -> None:
    processed = tmp_path / "processed"
    artifact = tmp_path / "artifact"
    processed.mkdir()
    _write_jsonl(
        processed / "train.jsonl",
        [
            {"user_id": 0, "item_id": 1, "timestamp": 1},
            {"user_id": 0, "item_id": 2, "timestamp": 2},
            {"user_id": 1, "item_id": 1, "timestamp": 1},
            {"user_id": 1, "item_id": 3, "timestamp": 2},
            {"user_id": 2, "item_id": 4, "timestamp": 1},
        ],
    )
    _write_jsonl(
        processed / "item_metadata.jsonl",
        [
            {"item_id": 1, "title": "Trail Shoes"},
            {"item_id": 2, "title": "Water Bottle"},
            {"item_id": 3, "title": "Running Socks"},
            {"item_id": 4, "title": "Yoga Mat"},
        ],
    )
    build_sequence_index_artifact(processed_dir=processed, output_dir=artifact, top_items=4, embedding_dim=2)
    pipeline = create_sequence_artifact_pipeline(processed_dir=processed, artifact_dir=artifact, retrieval_k=3)

    async def run() -> None:
        result = await pipeline.recommend(user_id="0", n_items=2)
        assert result.user_id == "0"
        assert len(result.recommendations) == 2
        assert result.recommendations[0].stage == "sequence_retrieval"
        assert result.recommendations[0].title
        assert set(result.latency_ms) == {"feature_fetch", "retrieval", "rescoring", "reranking", "total"}
        await pipeline.record_feedback(user_id="0", item_id=result.recommendations[0].item_id, interaction="click")
        assert pipeline.feedback_events

    asyncio.run(run())


def _write_jsonl(path, rows) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "
")
