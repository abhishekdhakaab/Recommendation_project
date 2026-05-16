import json
from pathlib import Path

from data.preprocess import preprocess_dataset
from data.synthetic import generate_synthetic_dataset
from evaluation.evaluate_retrieval_artifact import evaluate_retrieval_artifact
from models.train_two_tower import train_from_processed


def test_evaluate_retrieval_artifact_writes_summary(tmp_path: Path) -> None:
    dataset = generate_synthetic_dataset(seed=19, n_users=6, n_items=12)
    processed = tmp_path / "processed"
    artifact = tmp_path / "artifact"
    output = tmp_path / "summary.json"
    preprocess_dataset(dataset.interactions, dataset.items, output_dir=processed, user_min_interactions=3, item_min_interactions=1)
    train_from_processed(processed_dir=processed, output_dir=artifact, epochs=1, batch_size=4, max_train_interactions=8, max_validation_users=3, device="cpu", skip_index=True)

    summary = evaluate_retrieval_artifact(processed_dir=processed, artifact_dir=artifact, output_path=output, max_users=2, n_negatives=3)

    assert output.exists()
    assert json.loads(output.read_text())["protocol"] == summary["protocol"]
    assert "two_tower_plus_popularity_prior" in summary
