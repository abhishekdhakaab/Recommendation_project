"""Train Two-Tower retrieval from processed SeqRec artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from models.two_tower import (
    ItemTower,
    UserTower,
    encode_all_items,
    save_two_tower_checkpoint,
    train_two_tower_epoch,
    train_two_tower_step,
    validation_recall_at_k,
)
from retrieval.build_faiss_index import build_hnsw_index, save_index


def train_from_processed(
    *,
    processed_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    max_train_interactions: int | None = None,
    max_validation_users: int = 5000,
    patience: int = 5,
    seed: int = 7,
    device: str | None = None,
    skip_index: bool = False,
    unique_items_per_batch: bool = True,
) -> dict[str, object]:
    """Train a two-tower model and build a FAISS index from processed artifacts.

    Trains for up to ``epochs`` epochs with early stopping on validation
    Recall@50 (patience = ``patience`` epochs without improvement).
    """

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_validation_users < 1:
        raise ValueError("max_validation_users must be at least 1")

    random.seed(seed)
    torch.manual_seed(seed)
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stats = json.loads((processed_path / "stats.json").read_text(encoding="utf-8"))
    train_rows = _read_jsonl(processed_path / "train.jsonl", limit=max_train_interactions)
    validation_rows = _read_jsonl(processed_path / "validation.jsonl", limit=max_validation_users)
    target_device = torch.device(device or _default_device())

    print(f"Device: {target_device}")
    print(f"Training interactions: {len(train_rows):,} | Validation users: {len(validation_rows):,}")

    user_tower = UserTower(n_users=stats["users"]).to(target_device)
    item_tower = ItemTower(n_items=stats["items"]).to(target_device)
    optimizer = torch.optim.AdamW(
        list(user_tower.parameters()) + list(item_tower.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    all_item_ids = list(range(stats["items"]))
    recall_k = min(50, stats["items"])
    val_user_ids = [int(row["user_id"]) for row in validation_rows]
    val_item_ids = [int(row["item_id"]) for row in validation_rows]

    user_ids = [int(row["user_id"]) for row in train_rows]
    item_ids = [int(row["item_id"]) for row in train_rows]
    pairs = list(zip(user_ids, item_ids, strict=True))

    losses: list[float] = []
    best_recall = 0.0
    best_epoch = 0
    patience_count = 0
    best_user_state: dict | None = None
    best_item_state: dict | None = None

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        if unique_items_per_batch:
            epoch_loss = _train_unique_item_epoch(
                user_tower, item_tower, optimizer, pairs,
                batch_size=batch_size, device=target_device,
            )
        else:
            shuffled_users = [u for u, _ in pairs]
            shuffled_items = [i for _, i in pairs]
            result = train_two_tower_epoch(
                user_tower, item_tower, optimizer,
                shuffled_users, shuffled_items,
                batch_size=batch_size, device=target_device,
            )
            epoch_loss = result.loss
        losses.append(epoch_loss)

        validation_recall = validation_recall_at_k(
            user_tower, item_tower, val_user_ids, val_item_ids,
            all_item_ids, k=recall_k, device=target_device,
        )
        print(f"Epoch {epoch:3d}/{epochs} | loss={epoch_loss:.4f} | Recall@{recall_k}={validation_recall:.4f}")

        if validation_recall > best_recall + 1e-5:
            best_recall = validation_recall
            best_epoch = epoch
            patience_count = 0
            best_user_state = {k: v.cpu().clone() for k, v in user_tower.state_dict().items()}
            best_item_state = {k: v.cpu().clone() for k, v in item_tower.state_dict().items()}
            print(f"  ✓ New best Recall@{recall_k}={best_recall:.4f}")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Restore best weights.
    if best_user_state is not None:
        user_tower.load_state_dict({k: v.to(target_device) for k, v in best_user_state.items()})
        item_tower.load_state_dict({k: v.to(target_device) for k, v in best_item_state.items()})

    item_embeddings = encode_all_items(item_tower, all_item_ids, device=target_device).cpu().numpy().astype("float32")
    np.save(output_path / "item_embeddings.npy", item_embeddings)
    save_two_tower_checkpoint(
        output_path / "two_tower.pt",
        user_tower.cpu(),
        item_tower.cpu(),
        metadata={
            "stats": stats,
            "losses": losses,
            "validation_recall_at_k": best_recall,
            "recall_k": recall_k,
            "best_epoch": best_epoch,
        },
    )
    if not skip_index:
        index = build_hnsw_index(item_embeddings, item_ids=all_item_ids)
        save_index(index, output_path / "item_index.faiss")

    summary = {
        "epochs": len(losses),
        "best_epoch": best_epoch if best_epoch > 0 else len(losses),
        "batch_size": batch_size,
        "device": str(target_device),
        "train_interactions": len(train_rows),
        "validation_users": len(validation_rows),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "recall_k": recall_k,
        "validation_recall_at_k": best_recall,
        "output_dir": str(output_path),
        "index_built": not skip_index,
        "unique_items_per_batch": unique_items_per_batch,
    }
    (output_path / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _train_unique_item_epoch(
    user_tower: UserTower,
    item_tower: ItemTower,
    optimizer: torch.optim.Optimizer,
    pairs: list[tuple[int, int]],
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    total_loss = 0.0
    batches = 0
    current_users: list[int] = []
    current_items: list[int] = []
    seen_items: set[int] = set()

    def flush() -> None:
        nonlocal total_loss, batches, current_users, current_items, seen_items
        if not current_users:
            return
        result = train_two_tower_step(
            user_tower,
            item_tower,
            optimizer,
            torch.tensor(current_users, dtype=torch.long, device=device),
            torch.tensor(current_items, dtype=torch.long, device=device),
        )
        total_loss += result.loss
        batches += 1
        current_users = []
        current_items = []
        seen_items = set()

    for user_id, item_id in pairs:
        if item_id in seen_items or len(current_users) >= batch_size:
            flush()
        current_users.append(user_id)
        current_items.append(item_id)
        seen_items.add(item_id)
    flush()
    return total_loss / batches if batches else 0.0


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SeqRec Two-Tower retrieval from processed artifacts.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-train-interactions", type=int, default=None)
    parser.add_argument("--max-validation-users", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--allow-duplicate-items-per-batch", action="store_true")
    args = parser.parse_args()
    summary = train_from_processed(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_train_interactions=args.max_train_interactions,
        max_validation_users=args.max_validation_users,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        skip_index=args.skip_index,
        unique_items_per_batch=not args.allow_duplicate_items_per_batch,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
