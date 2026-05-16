import json
from pathlib import Path

from data.synthetic import generate_synthetic_dataset
from data.preprocess import preprocess_dataset
from models.train_two_tower import train_from_processed


def test_train_from_processed_writes_artifacts(tmp_path: Path) -> None:
    dataset = generate_synthetic_dataset(seed=13, n_users=6, n_items=12)
    processed = tmp_path / "processed"
    output = tmp_path / "artifacts"
    preprocess_dataset(dataset.interactions, dataset.items, output_dir=processed, user_min_interactions=3, item_min_interactions=1)

    summary = train_from_processed(
        processed_dir=processed,
        output_dir=output,
        epochs=1,
        batch_size=4,
        max_train_interactions=8,
        max_validation_users=3,
        device="cpu",
    )

    assert summary["epochs"] == 1
    assert summary["unique_items_per_batch"] is True
    assert (output / "two_tower.pt").exists()
    assert (output / "item_embeddings.npy").exists()
    assert (output / "item_index.faiss").exists()
    assert json.loads((output / "training_summary.json").read_text())["train_interactions"] == 8
