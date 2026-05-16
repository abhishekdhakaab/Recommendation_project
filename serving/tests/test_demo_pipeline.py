import asyncio

from serving.app.demo_pipeline import create_all_beauty_demo_pipeline


def test_all_beauty_demo_pipeline_returns_prd_shaped_result() -> None:
    async def run() -> None:
        pipeline = create_all_beauty_demo_pipeline()
        result = await pipeline.recommend(user_id="0", n_items=2, context="homepage")

        assert result.user_id == "0"
        assert result.recommendations
        assert result.recommendations[0].stage == "llm_reranker"
        assert set(result.latency_ms) == {"feature_fetch", "retrieval", "rescoring", "reranking", "total"}

    asyncio.run(run())
