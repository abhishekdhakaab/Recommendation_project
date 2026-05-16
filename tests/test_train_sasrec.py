import json
from pathlib import Path

from data.preprocess import preprocess_dataset
from data.synthetic import generate_synthetic_dataset
from models.train_sasrec import train_from_processed


def test_train_sasrec_from_processed_writes_artifacts(tmp_path: Path) -> None:
    dataset = generate_synthetic_dataset(seed=17, n_users=6, n_items=12)
    processed = tmp_path / "processed"
    output = tmp_path / "artifacts"
    preprocess_dataset(dataset.interactions, dataset.items, output_dir=processed, user_min_interactions=3, item_min_interactions=1)

    summary = train_from_processed(
        processed_dir=processed,
        output_dir=output,
        epochs=1,
        batch_size=2,
        max_train_examples=3,
        max_validation_users=3,
        negatives_per_positive=2,
        max_seq_len=5,
        device="cpu",
    )

    assert summary["epochs"] == 1
    assert (output / "sasrec.pt").exists()
    assert json.loads((output / "sasrec_summary.json").read_text())["train_examples"] <= 3
