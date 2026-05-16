from retrieval.sequence_retrieval import build_sequence_retrieval_index, load_sequence_retrieval_index, recommend_from_history, save_sequence_retrieval_index


def test_sequence_retrieval_recommends_cooccurring_items() -> None:
    index = build_sequence_retrieval_index(
        [[1, 2, 3], [1, 2], [4, 5]],
        item_ids=[1, 2, 3, 4, 5],
        embedding_dim=3,
    )

    recs = recommend_from_history(index, [1], candidates=[2, 4, 5], top_k=1)

    assert recs == [2]


def test_sequence_retrieval_returns_empty_for_unknown_history() -> None:
    index = build_sequence_retrieval_index([[1, 2]], item_ids=[1, 2], embedding_dim=2)

    assert recommend_from_history(index, [99]) == []


def test_sequence_retrieval_index_round_trip(tmp_path) -> None:
    index = build_sequence_retrieval_index([[1, 2], [2, 3]], item_ids=[1, 2, 3], embedding_dim=2)
    path = tmp_path / "index.npz"

    save_sequence_retrieval_index(index, path)
    loaded = load_sequence_retrieval_index(path)

    assert loaded.item_ids == index.item_ids
    assert loaded.local_id == index.local_id
    assert recommend_from_history(loaded, [1], candidates=[2, 3], top_k=1) == [2]
