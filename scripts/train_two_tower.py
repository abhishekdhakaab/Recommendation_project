"""Train the two-tower retrieval model on Amazon Beauty data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.two_tower import (
    ItemTower,
    UserTower,
    encode_all_items,
    in_batch_negative_loss,
    validation_recall_at_k,
)
from retrieval.build_faiss_index import build_hnsw_index, save_index
from retrieval.ann_search import search_item_ids
from utils.device import get_device

# --- Hyperparameters ---
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
TEMPERATURE = 0.07
EPOCHS = 50
EARLY_STOP_K = 5
RECALL_K = 50
N_VAL_USERS = 2000


def _train_unique_item_epoch(
    user_tower: UserTower,
    item_tower: ItemTower,
    optimizer: torch.optim.Optimizer,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    *,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> float:
    """Train one epoch using unique positive items per batch for better negatives."""

    user_tower.train()
    item_tower.train()

    # Shuffle
    perm = torch.randperm(user_ids.shape[0])
    user_ids = user_ids[perm]
    item_ids = item_ids[perm]

    total_loss = 0.0
    n_batches = 0

    for start in range(0, user_ids.shape[0], batch_size):
        u_batch = user_ids[start : start + batch_size].to(device)
        i_batch = item_ids[start : start + batch_size].to(device)

        # Deduplicate items within this batch to improve in-batch negatives
        unique_items, inverse_idx = torch.unique(i_batch, return_inverse=True)
        # Map user indices to unique item rows
        u_dedup: list[int] = []
        i_dedup: list[int] = []
        item_to_row: dict[int, int] = {}
        for j, (u, i) in enumerate(zip(u_batch.tolist(), inverse_idx.tolist())):
            item_val = int(unique_items[i].item())
            if item_val not in item_to_row:
                item_to_row[item_val] = len(i_dedup)
                i_dedup.append(item_val)
                u_dedup.append(u)
            else:
                # User already has a unique item pair; just add user
                u_dedup.append(u)
                i_dedup.append(item_val)

        u_t = torch.tensor(u_dedup, dtype=torch.long, device=device)
        i_t = torch.tensor(i_dedup, dtype=torch.long, device=device)

        if u_t.shape[0] < 2:
            continue

        optimizer.zero_grad(set_to_none=True)
        user_emb = user_tower(u_t)
        item_emb = item_tower(i_t)
        loss = in_batch_negative_loss(user_emb, item_emb, temperature=temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(user_tower.parameters()) + list(item_tower.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _latency_benchmark(index, n_queries: int = 1000, dim: int = 128) -> dict:
    """Measure FAISS query latency (P50, P99) with random float32 vectors."""

    latencies_ms: list[float] = []
    rng = np.random.default_rng(42)

    for _ in range(n_queries):
        vec = rng.standard_normal((1, dim)).astype(np.float32)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / np.maximum(norm, 1e-12)

        t0 = time.perf_counter()
        search_item_ids(index, vec, top_k=10)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_ms.sort()
    p50 = np.percentile(latencies_ms, 50)
    p99 = np.percentile(latencies_ms, 99)
    return {"p50_ms": float(p50), "p99_ms": float(p99)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Two-Tower retrieval model.")
    parser.add_argument("--data-dir", required=True, help="Processed data directory (contains parquet files)")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints and artifacts")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--early-stop", type=int, default=EARLY_STOP_K)
    parser.add_argument("--n-val-users", type=int, default=N_VAL_USERS)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # --- Load data ---
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required") from exc

    with (data_dir / "dataset_stats.json").open() as f:
        stats = json.load(f)
    n_users: int = stats["n_users"]
    n_items: int = stats["n_items"]
    print(f"Dataset: {n_users} users, {n_items} items")

    # Load train sequences → expand to (user_id, item_id) pairs
    df_train = pd.read_parquet(data_dir / "train_sequences.parquet")
    train_users: list[int] = []
    train_items: list[int] = []
    for _, row in df_train.iterrows():
        uid = int(row["user_id"])
        for iid in row["sequence"]:
            train_users.append(uid)
            train_items.append(int(iid))
    train_user_t = torch.tensor(train_users, dtype=torch.long)
    train_item_t = torch.tensor(train_items, dtype=torch.long)
    print(f"Training pairs: {len(train_users)}")

    # Load validation labels
    df_val = pd.read_parquet(data_dir / "val_labels.parquet")
    n_val = min(args.n_val_users, len(df_val))
    df_val = df_val.iloc[:n_val]
    val_user_ids = df_val["user_id"].tolist()
    val_item_ids = df_val["item_id"].tolist()
    all_item_ids = list(range(1, n_items + 1))

    # --- Build models ---
    user_tower = UserTower(n_users=n_users, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(device)
    item_tower = ItemTower(n_items=n_items + 1, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(device)

    optimizer = torch.optim.AdamW(
        list(user_tower.parameters()) + list(item_tower.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_recall = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        avg_loss = _train_unique_item_epoch(
            user_tower,
            item_tower,
            optimizer,
            train_user_t,
            train_item_t,
            batch_size=args.batch_size,
            temperature=args.temperature,
            device=device,
        )

        # Validate
        recall = validation_recall_at_k(
            user_tower,
            item_tower,
            val_user_ids,
            val_item_ids,
            all_item_ids,
            k=RECALL_K,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
            f"Recall@{RECALL_K}={recall:.4f} | lr={current_lr:.2e}"
        )

        scheduler.step(recall)

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            no_improve = 0
            # Save best checkpoint
            torch.save(
                {
                    "model_state_dict": {
                        "user_tower": user_tower.state_dict(),
                        "item_tower": item_tower.state_dict(),
                    },
                    "recall_at_50": best_recall,
                    "epoch": best_epoch,
                    "n_users": n_users,
                    "n_items": n_items,
                },
                output_dir / "best_checkpoint.pt",
            )
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop} epochs)")
                break

    print(f"\nBest Recall@{RECALL_K}: {best_recall:.4f} at epoch {best_epoch}")

    # --- Load best checkpoint for export ---
    ckpt = torch.load(output_dir / "best_checkpoint.pt", map_location=device)
    item_tower.load_state_dict(ckpt["model_state_dict"]["item_tower"])
    item_tower.eval()

    # --- Export item embeddings ---
    print("Encoding all items ...")
    item_embeddings = encode_all_items(
        item_tower, list(range(1, n_items + 1)), device=device
    ).cpu().numpy().astype(np.float32)

    embeddings_path = output_dir / "item_embeddings.npy"
    np.save(str(embeddings_path), item_embeddings)
    print(f"Saved item embeddings: {item_embeddings.shape} → {embeddings_path}")

    # --- Build FAISS index ---
    print("Building FAISS HNSW index ...")
    faiss_index = build_hnsw_index(
        item_embeddings,
        item_ids=list(range(1, n_items + 1)),
    )
    index_path = output_dir / "item_index.faiss"
    save_index(faiss_index, index_path)
    print(f"Saved FAISS index → {index_path}")

    # --- Latency benchmark ---
    print("Running FAISS latency benchmark (1000 queries) ...")
    lat = _latency_benchmark(faiss_index, n_queries=1000, dim=EMBEDDING_DIM)
    lat["n_items"] = n_items
    lat["dim"] = EMBEDDING_DIM
    latency_path = output_dir / "faiss_latency.json"
    with latency_path.open("w") as f:
        json.dump(lat, f, indent=2)
    print(f"FAISS latency: P50={lat['p50_ms']:.2f}ms  P99={lat['p99_ms']:.2f}ms")
    print(f"Saved latency → {latency_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
