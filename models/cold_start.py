"""Cold-start routing and content-based retrieval for SeqRec."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from retrieval.ann_search import SearchResult, search_top_k
from retrieval.build_faiss_index import HNSWItemIndex


DEFAULT_WARM_THRESHOLD = 5
DEFAULT_BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"


class RecommendationRoute(str, Enum):
    """Routing decision for a recommendation request."""

    POPULARITY_FALLBACK = "popularity_fallback"
    CONTENT_BASED = "content_based"
    WARM_PATH = "warm_path"


@dataclass(frozen=True)
class ColdStartRecommendation:
    """Recommendation produced by the cold-start layer."""

    item_id: int
    score: float
    source: RecommendationRoute


@dataclass(frozen=True)
class ColdStartResult:
    """Cold-start routing result."""

    route: RecommendationRoute
    recommendations: list[ColdStartRecommendation]


@dataclass(frozen=True)
class PopularityItem:
    """Popularity-weighted fallback candidate."""

    item_id: int
    popularity_score: float


class ItemEmbeddingProvider(Protocol):
    """Provides BGE-space item embeddings without forcing model loading in tests."""

    def embed_item_ids(self, item_ids: Sequence[int]) -> np.ndarray:
        """Return one embedding per item ID."""


class BGEItemEmbeddingProvider:
    """Lazy BGE wrapper for item text embeddings.

    This class intentionally loads `sentence_transformers` only when embeddings
    are requested, so tests and lightweight imports never download or initialize
    a large model.
    """

    def __init__(
        self,
        item_text_by_id: dict[int, str],
        *,
        model_name: str = DEFAULT_BGE_MODEL_NAME,
    ) -> None:
        self.item_text_by_id = dict(item_text_by_id)
        self.model_name = model_name
        self._model = None

    def embed_item_ids(self, item_ids: Sequence[int]) -> np.ndarray:
        texts = []
        for item_id in item_ids:
            if item_id not in self.item_text_by_id:
                raise KeyError(f"Missing item text for item_id={item_id}")
            texts.append(self.item_text_by_id[item_id])

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)


class ColdStartRouter:
    """Route zero, sparse, and warm users to the correct retrieval path."""

    def __init__(
        self,
        *,
        popularity_items: Sequence[PopularityItem | dict[str, float | int]],
        embedding_provider: ItemEmbeddingProvider | None = None,
        content_index: HNSWItemIndex | None = None,
        warm_threshold: int = DEFAULT_WARM_THRESHOLD,
    ) -> None:
        if warm_threshold < 1:
            raise ValueError("warm_threshold must be at least 1")
        self.popularity_items = _normalize_popularity_items(popularity_items)
        self.embedding_provider = embedding_provider
        self.content_index = content_index
        self.warm_threshold = warm_threshold

    def route(
        self,
        *,
        interaction_history: Sequence[int],
        top_k: int,
    ) -> ColdStartResult:
        """Return fallback/content recommendations or route to warm path."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        interaction_count = len(interaction_history)
        if interaction_count == 0:
            return ColdStartResult(
                route=RecommendationRoute.POPULARITY_FALLBACK,
                recommendations=[
                    ColdStartRecommendation(
                        item_id=item.item_id,
                        score=item.popularity_score,
                        source=RecommendationRoute.POPULARITY_FALLBACK,
                    )
                    for item in self.popularity_items[:top_k]
                ],
            )

        if interaction_count < self.warm_threshold:
            if self.embedding_provider is None or self.content_index is None:
                raise ValueError("content-based routing requires embedding_provider and content_index")
            query_embedding = average_item_embeddings(self.embedding_provider, interaction_history)
            results = search_top_k(self.content_index, query_embedding, top_k=top_k)[0]
            return ColdStartResult(
                route=RecommendationRoute.CONTENT_BASED,
                recommendations=[
                    ColdStartRecommendation(
                        item_id=result.item_id,
                        score=result.score,
                        source=RecommendationRoute.CONTENT_BASED,
                    )
                    for result in results
                ],
            )

        return ColdStartResult(route=RecommendationRoute.WARM_PATH, recommendations=[])


def average_item_embeddings(
    embedding_provider: ItemEmbeddingProvider,
    item_ids: Sequence[int],
) -> np.ndarray:
    """Average BGE-space item embeddings for sparse user histories."""

    if not item_ids:
        raise ValueError("item_ids must not be empty")
    embeddings = np.asarray(embedding_provider.embed_item_ids(item_ids), dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embedding_provider must return a 2D array")
    if embeddings.shape[0] != len(item_ids):
        raise ValueError("embedding count must match item_ids length")

    query = embeddings.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(query, axis=1, keepdims=True)
    return np.divide(query, np.maximum(norm, 1e-12)).astype(np.float32)


def generate_bge_item_embeddings(
    item_meta_path: str,
    output_path: str,
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
    device=None,
) -> None:
    """Load item_meta.parquet, encode with BGE, and save embeddings + FAISS index.

    Shape saved: (max_item_id + 1, embed_dim) with row 0 = zeros (padding).
    Also builds and saves a FAISS flat IP index alongside the numpy file.
    """

    import faiss  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    from retrieval.build_faiss_index import save_index, HNSWItemIndex

    output_path_ = Path(output_path)
    output_path_.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading item metadata from {item_meta_path} ...")
    df = pd.read_parquet(item_meta_path)
    df = df.sort_values("item_id").reset_index(drop=True)

    texts: list[str] = []
    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        description = str(row.get("description", "") or "")
        text = (title + " " + description)[:512]
        texts.append(text)

    print(f"Encoding {len(texts)} items with {model_name} ...")
    encode_kwargs = {}
    if device is not None:
        encode_kwargs["device"] = str(device)

    st_model = SentenceTransformer(model_name)
    embeddings = st_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        **encode_kwargs,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embed_dim = embeddings.shape[1]

    # item_ids are 1-indexed; build array of shape (max_item_id + 1, embed_dim)
    item_ids_list = df["item_id"].tolist()
    max_item_id = max(item_ids_list)
    full_embeddings = np.zeros((max_item_id + 1, embed_dim), dtype=np.float32)
    for i, item_id in enumerate(item_ids_list):
        full_embeddings[int(item_id)] = embeddings[i]

    print(f"Saving embeddings shape {full_embeddings.shape} → {output_path_}")
    np.save(str(output_path_), full_embeddings)

    # Build FAISS flat IP index for items 1..max_item_id
    print("Building FAISS flat IP index ...")
    item_embs_1indexed = full_embeddings[1:]  # shape (max_item_id, embed_dim)
    faiss_idx = faiss.IndexFlatIP(embed_dim)
    faiss_idx.add(item_embs_1indexed)

    # Wrap with item_ids for save_index
    item_ids_arr = np.arange(1, max_item_id + 1, dtype=np.int64)
    hnsw_idx = HNSWItemIndex(index=faiss_idx, item_ids=item_ids_arr)
    faiss_path = output_path_.with_suffix(".faiss")
    save_index(hnsw_idx, faiss_path)
    print(f"Saved FAISS index → {faiss_path}")
    print("BGE embedding generation complete.")


def _normalize_popularity_items(
    popularity_items: Sequence[PopularityItem | dict[str, float | int]],
) -> list[PopularityItem]:
    normalized = []
    for item in popularity_items:
        if isinstance(item, PopularityItem):
            normalized.append(item)
        else:
            normalized.append(
                PopularityItem(
                    item_id=int(item["item_id"]),
                    popularity_score=float(item["popularity_score"]),
                )
            )
    return sorted(normalized, key=lambda item: (-item.popularity_score, item.item_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BGE item embeddings for cold-start.")
    parser.add_argument("--meta", required=True, help="Path to item_meta.parquet")
    parser.add_argument("--output-dir", required=True, help="Output directory for embeddings")
    parser.add_argument("--model-name", default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bge_item_embeddings.npy"

    generate_bge_item_embeddings(
        args.meta,
        str(output_path),
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
