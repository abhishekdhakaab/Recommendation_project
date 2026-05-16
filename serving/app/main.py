"""FastAPI serving entrypoint for SeqRec."""

from __future__ import annotations

import os
import time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from serving.app.metrics import ServingMetrics
from serving.app.pipeline import (
    PipelineResult,
    RecommendationPipeline,
    UnconfiguredRecommendationPipeline,
)

if os.getenv("SEQREC_DEMO_PIPELINE") == "1":
    from serving.app.demo_pipeline import create_all_beauty_demo_pipeline

if os.getenv("SEQREC_ARTIFACT_PIPELINE") == "1":
    from serving.app.artifact_pipeline import create_sequence_artifact_pipeline


LATENCY_KEYS = ("feature_fetch", "retrieval", "rescoring", "reranking", "total")


class RecommendRequest(BaseModel):
    user_id: str
    n_items: int = Field(default=10, ge=1)
    context: str | None = None


class RecommendationResponseItem(BaseModel):
    item_id: str
    title: str
    score: float
    stage: str


class RecommendResponse(BaseModel):
    user_id: str
    cold_start: bool
    recommendations: list[RecommendationResponseItem]
    latency_ms: dict[str, float]


class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    interaction: str


class FeedbackResponse(BaseModel):
    status: str


app = FastAPI(title="SeqRec Serving API")
if os.getenv("SEQREC_ARTIFACT_PIPELINE") == "1":
    app.state.pipeline = create_sequence_artifact_pipeline(
        processed_dir=os.getenv("SEQREC_PROCESSED_DIR", "data/processed/sports_and_outdoors"),
        artifact_dir=os.getenv("SEQREC_ARTIFACT_DIR", "models/artifacts/sports_and_outdoors_sequence"),
    )
elif os.getenv("SEQREC_DEMO_PIPELINE") == "1":
    app.state.pipeline = create_all_beauty_demo_pipeline()
else:
    app.state.pipeline = UnconfiguredRecommendationPipeline()
app.state.metrics = ServingMetrics()


def get_pipeline() -> RecommendationPipeline:
    return app.state.pipeline


def get_metrics() -> ServingMetrics:
    return app.state.metrics


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(
    request: RecommendRequest,
    pipeline: Annotated[RecommendationPipeline, Depends(get_pipeline)],
    metrics: Annotated[ServingMetrics, Depends(get_metrics)],
) -> RecommendResponse:
    started = time.perf_counter()
    try:
        result = await pipeline.recommend(
            user_id=request.user_id,
            n_items=request.n_items,
            context=request.context,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    latency_ms = _normalize_latency(result.latency_ms, started)
    metrics.record_recommendation(latency_ms)
    return RecommendResponse(
        user_id=result.user_id,
        cold_start=result.cold_start,
        recommendations=[
            RecommendationResponseItem(
                item_id=recommendation.item_id,
                title=recommendation.title,
                score=recommendation.score,
                stage=recommendation.stage,
            )
            for recommendation in result.recommendations
        ],
        latency_ms=latency_ms,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    request: FeedbackRequest,
    pipeline: Annotated[RecommendationPipeline, Depends(get_pipeline)],
    metrics: Annotated[ServingMetrics, Depends(get_metrics)],
) -> FeedbackResponse:
    try:
        await pipeline.record_feedback(
            user_id=request.user_id,
            item_id=request.item_id,
            interaction=request.interaction,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    metrics.record_feedback()
    return FeedbackResponse(status="ok")


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(metrics: Annotated[ServingMetrics, Depends(get_metrics)]) -> str:
    return metrics.render_prometheus()


def _normalize_latency(latency_ms: dict[str, float], started: float) -> dict[str, float]:
    normalized = {key: float(latency_ms.get(key, 0.0)) for key in LATENCY_KEYS}
    if normalized["total"] <= 0:
        normalized["total"] = (time.perf_counter() - started) * 1000
    return normalized
