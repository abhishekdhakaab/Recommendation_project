import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch

from models.sasrec import (
    DEFAULT_MAX_SEQ_LEN,
    SASRec,
    build_causal_mask,
    sampled_bce_loss,
    score_candidates,
    train_sasrec_step,
)


def test_sasrec_forward_returns_expected_shape() -> None:
    model = SASRec(n_items=20, max_seq_len=6, dropout=0.0)
    sequences = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0],
            [4, 5, 6, 7, 8, 9],
        ],
        dtype=torch.long,
    )

    encoded = model(sequences)

    assert encoded.shape == (2, 6, 64)


def test_sasrec_uses_padding_token_and_zeroes_padded_outputs() -> None:
    model = SASRec(n_items=10, max_seq_len=5, dropout=0.0)
    sequences = torch.tensor([[1, 2, 0, 0, 0]], dtype=torch.long)

    encoded = model(sequences)

    assert torch.allclose(model.item_embedding.weight[0], torch.zeros(model.embedding_dim))
    assert torch.allclose(encoded[0, 2:], torch.zeros(3, model.embedding_dim), atol=1e-6)


def test_build_causal_mask_blocks_future_positions() -> None:
    mask = build_causal_mask(4)

    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected)


def test_sasrec_causal_mask_prevents_future_items_from_changing_prefix_outputs() -> None:
    torch.manual_seed(0)
    model = SASRec(n_items=20, max_seq_len=5, dropout=0.0)
    model.eval()
    shared_prefix_a = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    shared_prefix_b = torch.tensor([[1, 2, 3, 9, 10]], dtype=torch.long)

    with torch.inference_mode():
        encoded_a = model(shared_prefix_a)
        encoded_b = model(shared_prefix_b)

    assert torch.allclose(encoded_a[:, :3], encoded_b[:, :3], atol=1e-5)
    assert not torch.allclose(encoded_a[:, 3:], encoded_b[:, 3:], atol=1e-5)


def test_score_candidates_returns_one_score_per_candidate() -> None:
    model = SASRec(n_items=30, max_seq_len=5, dropout=0.0)
    sequences = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]], dtype=torch.long)
    candidates = torch.tensor([[3, 8, 9], [7, 10, 11]], dtype=torch.long)

    scores = model.score_candidates(sequences, candidates)
    wrapped_scores = score_candidates(model, sequences, candidates)

    assert scores.shape == (2, 3)
    assert torch.allclose(scores, wrapped_scores)
    assert torch.isfinite(scores).all()


def test_score_candidates_accepts_shared_candidate_list() -> None:
    model = SASRec(n_items=30, max_seq_len=5, dropout=0.0)
    sequences = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]], dtype=torch.long)
    candidates = torch.tensor([3, 8, 9], dtype=torch.long)

    scores = model.score_candidates(sequences, candidates)

    assert scores.shape == (2, 3)


def test_sasrec_sampled_loss_and_train_step_smoke() -> None:
    model = SASRec(n_items=30, max_seq_len=5, dropout=0.0)
    sequences = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]], dtype=torch.long)
    positives = torch.tensor([4, 8], dtype=torch.long)
    negatives = torch.tensor([[9, 10], [11, 12]], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = sampled_bce_loss(model, sequences, positives, negatives)
    step_loss = train_sasrec_step(model, optimizer, sequences, positives, negatives)

    assert loss.item() > 0
    assert step_loss > 0


def test_sasrec_defaults_match_prd_shape_choices() -> None:
    model = SASRec(n_items=50)

    assert model.max_seq_len == DEFAULT_MAX_SEQ_LEN
    assert model.item_embedding.num_embeddings == 51
    assert len(model.transformer.layers) == 2


def test_sasrec_text_enrichment_changes_output() -> None:
    torch.manual_seed(0)
    n_items = 20
    text_dim = 32
    model = SASRec(n_items=n_items, max_seq_len=5, dropout=0.0, text_embedding_dim=text_dim)
    model.eval()

    # (n_items+1) rows, row 0 is padding
    text_emb = torch.randn(n_items + 1, text_dim)
    text_emb[0] = 0.0
    model.set_text_embeddings(text_emb)

    sequences = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    with torch.inference_mode():
        out_with_text = model(sequences)

    # Remove text embeddings and re-run — output should differ.
    model_no_text = SASRec(n_items=n_items, max_seq_len=5, dropout=0.0)
    model_no_text.load_state_dict(model.state_dict(), strict=False)
    model_no_text.eval()
    with torch.inference_mode():
        out_no_text = model_no_text(sequences)

    assert not torch.allclose(out_with_text, out_no_text, atol=1e-4), (
        "Text enrichment should change the model output"
    )


def test_sasrec_set_text_embeddings_validates_shape() -> None:
    model = SASRec(n_items=10, max_seq_len=5, text_embedding_dim=16)

    with pytest.raises(ValueError, match="n_items\\+1"):
        model.set_text_embeddings(torch.zeros(5, 16))  # wrong n_rows

    with pytest.raises(ValueError, match="16 columns"):
        model.set_text_embeddings(torch.zeros(11, 8))  # wrong dim

    with pytest.raises(ValueError, match="text_embedding_dim was not set"):
        SASRec(n_items=10, max_seq_len=5).set_text_embeddings(torch.zeros(11, 16))


def test_sasrec_validates_sequence_and_candidate_inputs() -> None:
    model = SASRec(n_items=5, max_seq_len=3)

    with pytest.raises(ValueError, match="length exceeds max_seq_len"):
        model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    with pytest.raises(ValueError, match="range 0..n_items"):
        model(torch.tensor([[1, 6, 0]], dtype=torch.long))
    with pytest.raises(ValueError, match="batch size"):
        model.score_candidates(
            torch.tensor([[1, 2, 0]], dtype=torch.long),
            torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        )
