from pathlib import Path

from data.preprocess import preprocess_dataset
from data.synthetic import generate_synthetic_dataset
from evaluation.evaluate_sequence_retrieval import evaluate_sequence_retrieval


def test_evaluate_sequence_retrieval_writes_summary(tmp_path: Path) -> None:
    dataset = generate_synthetic_dataset(seed=23, n_users=6, n_items=12)
    processed = tmp_path / "processed"
    output = tmp_path / "summary.json"
    preprocess_dataset(dataset.interactions, dataset.items, output_dir=processed, user_min_interactions=3, item_min_interactions=1)

    summary = evaluate_sequence_retrieval(processed_dir=processed, output_path=output, max_users=2, n_negatives=3, embedding_dim=4)

    assert output.exists()
    assert "at_10" in summary
