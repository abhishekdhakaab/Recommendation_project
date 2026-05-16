"""Train the SASRec sequential recommendation model on Amazon Beauty data."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.sasrec import SASRec, sampled_bce_loss, train_sasrec_step
from evaluation.offline_eval import evaluate_popularity_sampled_ranking_batch
from utils.device import get_device

# --- Hyperparameters ---
EMBEDDING_DIM = 64
MAX_SEQ_LEN = 50
NUM_HEADS = 2
NUM_LAYERS = 2
FFN_DIM = 256
DROPOUT = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 200
EARLY_STOP_K = 10
NEG_SAMPLES = 1
N_VAL_USERS = 2000


def _pad_sequence(seq: list[int], max_len: int) -> list[int]:
    """Left-pad a sequence with zeros to max_len."""
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def _build_sliding_windows(
    sequences: dict[int, list[int]],
    max_seq_len: int,
) -> tuple[list[list[int]], list[int]]:
    """Build sliding-window (input_seq, target) pairs from user sequences."""

    inputs: list[list[int]] = []
    targets: list[int] = []

    for uid, seq in sequences.items():
        for t in range(len(seq) - 1):
            # Input: seq[0..t], padded to max_seq_len
            input_seq = _pad_sequence(seq[: t + 1], max_seq_len)
            target = seq[t + 1]
            inputs.append(input_seq)
            targets.append(target)

    return inputs, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SASRec sequential recommendation model.")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--early-stop", type=int, default=EARLY_STOP_K)
    parser.add_argument("--n-val-users", type=int, default=N_VAL_USERS)
    args = parser.parse_args()

    # MPS causal mask fix
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
    n_items: int = stats["n_items"]
    print(f"Dataset: n_items={n_items}")

    # Load train sequences
    df_train = pd.read_parquet(data_dir / "train_sequences.parquet")
    train_sequences: dict[int, list[int]] = {}
    for _, row in df_train.iterrows():
        uid = int(row["user_id"])
        seq = [int(x) for x in row["sequence"]]
        train_sequences[uid] = seq

    # Compute item popularity from training sequences
    popularity_counts: Counter = Counter()
    for seq in train_sequences.values():
        for iid in seq:
            popularity_counts[iid] += 1
    # Ensure all items 1..n_items are present (with at least 0)
    for iid in range(1, n_items + 1):
        if iid not in popularity_counts:
            popularity_counts[iid] = 0

    # Build sliding window training examples
    print("Building sliding-window training examples ...")
    all_inputs, all_targets = _build_sliding_windows(train_sequences, MAX_SEQ_LEN)
    print(f"Training examples: {len(all_inputs)}")

    # Convert to tensors
    inputs_t = torch.tensor(all_inputs, dtype=torch.long)
    targets_t = torch.tensor(all_targets, dtype=torch.long)
    n_train = inputs_t.shape[0]

    # Load validation labels
    df_val = pd.read_parquet(data_dir / "val_labels.parquet")
    n_val = min(args.n_val_users, len(df_val))
    df_val = df_val.iloc[:n_val]
    # Build relevant_by_user for validation
    relevant_by_user: dict[int, list[int]] = {}
    for _, row in df_val.iterrows():
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        relevant_by_user[uid] = [iid]

    # --- Build model ---
    model = SASRec(
        n_items=n_items,
        max_seq_len=MAX_SEQ_LEN,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        feedforward_dim=FFN_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ndcg = 0.0
    best_hit = 0.0
    best_epoch = 0
    no_improve = 0

    rng = random.Random(42)

    print(f"Training SASRec for up to {args.epochs} epochs ...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle indices
        perm = torch.randperm(n_train)
        inputs_shuf = inputs_t[perm]
        targets_shuf = targets_t[perm]

        for start in range(0, n_train, args.batch_size):
            seq_batch = inputs_shuf[start : start + args.batch_size].to(device)
            pos_batch = targets_shuf[start : start + args.batch_size].to(device)
            batch_sz = seq_batch.shape[0]

            # Sample negatives uniformly (not in the positive)
            neg_list: list[list[int]] = []
            for pid in pos_batch.tolist():
                negs: list[int] = []
                while len(negs) < NEG_SAMPLES:
                    candidate = rng.randint(1, n_items)
                    if candidate != pid:
                        negs.append(candidate)
                neg_list.append(negs)
            neg_batch = torch.tensor(neg_list, dtype=torch.long, device=device)

            loss = train_sasrec_step(model, optimizer, seq_batch, pos_batch, neg_batch)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()

            def score_items_fn(user_id: int, candidate_ids: list[int]) -> list[float]:
                seq = train_sequences.get(user_id, [])[-MAX_SEQ_LEN:]
                padded = _pad_sequence(seq, MAX_SEQ_LEN)
                seq_t = torch.tensor([padded], dtype=torch.long, device=device)
                cand_t = torch.tensor([candidate_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    scores = model.score_candidates(seq_t, cand_t)[0]
                return scores.cpu().tolist()

            try:
                metrics = evaluate_popularity_sampled_ranking_batch(
                    score_items_fn,
                    relevant_by_user,
                    popularity_counts=popularity_counts,
                    k=10,
                    n_negatives=100,
                )
                ndcg = metrics.ndcg
                hit = metrics.hit
            except Exception as exc:
                print(f"  [warn] Evaluation failed at epoch {epoch}: {exc}")
                ndcg = 0.0
                hit = 0.0

            print(
                f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
                f"NDCG@10={ndcg:.4f} | Hit@10={hit:.4f}"
            )

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_hit = hit
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "ndcg_at_10": best_ndcg,
                        "hit_at_10": best_hit,
                        "epoch": best_epoch,
                        "n_items": n_items,
                        "max_seq_len": MAX_SEQ_LEN,
                    },
                    output_dir / "best_checkpoint.pt",
                )
            else:
                no_improve += 1
                if no_improve >= args.early_stop:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no NDCG improvement for {args.early_stop} evals)"
                    )
                    break
        else:
            # Print loss-only update for non-eval epochs
            if epoch % 10 == 1 or epoch == 1:
                print(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f}")

    # Always save a checkpoint (fallback to final model if no eval improvement recorded).
    ckpt_path = output_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "ndcg_at_10": best_ndcg,
                "hit_at_10": best_hit,
                "epoch": best_epoch or args.epochs,
                "n_items": n_items,
                "max_seq_len": MAX_SEQ_LEN,
            },
            ckpt_path,
        )
    print(f"\nBest NDCG@10: {best_ndcg:.4f}  Hit@10: {best_hit:.4f}  (epoch {best_epoch})")
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
