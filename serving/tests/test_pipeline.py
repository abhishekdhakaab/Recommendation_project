import asyncio

from serving.app.pipeline import Candidate, MultiStageRecommendationPipeline


def test_multi_stage_pipeline_orchestrates_injected_stages() -> None:
    async def run() -> None:
        calls: list[str] = []

        class Features:
            cold_start_flag = False

        def fetch(user_id: str) -> Features:
            calls.append(f"fetch:{user_id}")
            return Features()

        def retrieve(user_id: str, top_k: int, features: Features) -> list[Candidate]:
            calls.append(f"retrieve:{top_k}")
            return [Candidate(item_id="I1", title="One", text="One text", score=0.3, stage="retrieval")]

        def rescore(features: Features, candidates: list[Candidate], top_k: int) -> list[Candidate]:
            calls.append(f"rescore:{top_k}")
            return [Candidate(item_id="I1", title="One", text="One text", score=0.7, stage="sasrec")]

        def rerank(features: Features, candidates: list[Candidate], top_n: int) -> list[Candidate]:
            calls.append(f"rerank:{top_n}")
            return [Candidate(item_id="I1", title="One", text="One text", score=0.9, stage="llm_reranker")]

        pipeline = MultiStageRecommendationPipeline(
            feature_fetcher=fetch,
            retriever=retrieve,
            rescorer=rescore,
            reranker=rerank,
            retrieval_k=500,
            rescore_k=50,
        )

        result = await pipeline.recommend(user_id="U1", n_items=10)

        assert calls == ["fetch:U1", "retrieve:500", "rescore:50", "rerank:10"]
        assert result.recommendations[0].stage == "llm_reranker"
        assert set(result.latency_ms) == {"feature_fetch", "retrieval", "rescoring", "reranking", "total"}

    asyncio.run(run())
