from fastapi.testclient import TestClient

from serving.app.main import app
from serving.app.metrics import ServingMetrics
from serving.app.pipeline import PipelineResult, Recommendation


def test_recommend_endpoint_returns_prd_response_schema() -> None:
    fake_pipeline = FakePipeline()
    client = _client(fake_pipeline)

    response = client.post(
        "/recommend",
        json={"user_id": "U12345", "n_items": 2, "context": "homepage"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "U12345",
        "cold_start": False,
        "recommendations": [
            {"item_id": "I1", "title": "Item 1", "score": 0.9, "stage": "llm_reranker"},
            {"item_id": "I2", "title": "Item 2", "score": 0.8, "stage": "llm_reranker"},
        ],
        "latency_ms": {
            "feature_fetch": 1.0,
            "retrieval": 2.0,
            "rescoring": 3.0,
            "reranking": 4.0,
            "total": 10.0,
        },
    }
    assert fake_pipeline.recommend_calls == [{"user_id": "U12345", "n_items": 2, "context": "homepage"}]


def test_feedback_endpoint_records_feedback() -> None:
    fake_pipeline = FakePipeline()
    client = _client(fake_pipeline)

    response = client.post(
        "/feedback",
        json={"user_id": "U12345", "item_id": "I9", "interaction": "click"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert fake_pipeline.feedback_calls == [{"user_id": "U12345", "item_id": "I9", "interaction": "click"}]


def test_metrics_endpoint_returns_prometheus_text() -> None:
    client = _client(FakePipeline())
    client.post("/recommend", json={"user_id": "U1", "n_items": 1})
    client.post("/feedback", json={"user_id": "U1", "item_id": "I1", "interaction": "click"})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "seqrec_recommend_requests_total 1" in response.text
    assert "seqrec_feedback_requests_total 1" in response.text
    assert 'seqrec_stage_latency_ms_sum{stage="retrieval"} 2.0' in response.text


def test_unconfigured_pipeline_returns_503() -> None:
    app.dependency_overrides.clear()
    app.state.metrics = ServingMetrics()
    client = TestClient(app)

    response = client.post("/recommend", json={"user_id": "U1", "n_items": 1})

    assert response.status_code == 503


def _client(fake_pipeline: "FakePipeline") -> TestClient:
    app.state.metrics = ServingMetrics()
    app.dependency_overrides.clear()
    from serving.app.main import get_pipeline

    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    return TestClient(app)


class FakePipeline:
    def __init__(self) -> None:
        self.recommend_calls: list[dict[str, object]] = []
        self.feedback_calls: list[dict[str, str]] = []

    async def recommend(self, *, user_id: str, n_items: int, context: str | None = None) -> PipelineResult:
        self.recommend_calls.append({"user_id": user_id, "n_items": n_items, "context": context})
        recommendations = [
            Recommendation(item_id="I1", title="Item 1", score=0.9, stage="llm_reranker"),
            Recommendation(item_id="I2", title="Item 2", score=0.8, stage="llm_reranker"),
        ]
        return PipelineResult(
            user_id=user_id,
            cold_start=False,
            recommendations=recommendations[:n_items],
            latency_ms={
                "feature_fetch": 1.0,
                "retrieval": 2.0,
                "rescoring": 3.0,
                "reranking": 4.0,
                "total": 10.0,
            },
        )

    async def record_feedback(self, *, user_id: str, item_id: str, interaction: str) -> None:
        self.feedback_calls.append({"user_id": user_id, "item_id": item_id, "interaction": interaction})
