import numpy as np

from retrieval.ann_search import search_item_ids
from retrieval.build_flat_index import build_flat_index


def test_build_flat_index_returns_expected_neighbor() -> None:
    embeddings = np.eye(3, dtype=np.float32)
    index = build_flat_index(embeddings, np.array([10, 11, 12]))

    assert search_item_ids(index, embeddings[[1]], top_k=1) == [[11]]
