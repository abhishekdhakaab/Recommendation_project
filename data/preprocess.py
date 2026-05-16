"""Offline preprocessing utilities for SeqRec interaction data."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
from dataclasses import dataclass
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Iterable


Record = dict[str, Any]


@dataclass(frozen=True)
class ProcessedArtifacts:
    """In-memory representation of processed recommendation artifacts."""

    train: list[Record]
    validation: list[Record]
    test: list[Record]
    item_metadata: list[Record]
    user_mapping: dict[str, int]
    item_mapping: dict[str, int]
    stats: dict[str, int]


def load_interactions(path: str | Path) -> list[Record]:
    """Load interaction records from CSV, JSON, JSONL, or Parquet."""

    records = [_normalize_interaction(row) for row in _load_records(path)]
    _require_fields(records, {"user_id", "item_id", "timestamp"}, "interactions")
    return records


def load_item_metadata(path: str | Path) -> list[Record]:
    """Load item metadata records from CSV, JSON, JSONL, or Parquet."""

    records = [_normalize_item(row) for row in _load_records(path)]
    _require_fields(records, {"item_id"}, "item metadata")
    return records


def preprocess_dataset(
    interactions: Iterable[Record],
    item_metadata: Iterable[Record],
    *,
    output_dir: str | Path | None = None,
    user_min_interactions: int = 5,
    item_min_interactions: int = 10,
) -> ProcessedArtifacts:
    """Filter, sort, encode, split, and optionally write processed artifacts."""

    _validate_threshold("user_min_interactions", user_min_interactions)
    _validate_threshold("item_min_interactions", item_min_interactions)

    raw_interactions = [dict(row) for row in interactions]
    raw_items = [dict(row) for row in item_metadata]
    filtered = k_core_filter(
        raw_interactions,
        user_min_interactions=user_min_interactions,
        item_min_interactions=item_min_interactions,
    )
    sorted_filtered = sort_interactions_chronologically(filtered)
    encoded_interactions, encoded_items, user_mapping, item_mapping = encode_ids(
        sorted_filtered,
        raw_items,
    )
    train, validation, test, dropped_short_sequence_users = split_train_validation_test(encoded_interactions)

    artifacts = ProcessedArtifacts(
        train=train,
        validation=validation,
        test=test,
        item_metadata=encoded_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        stats={
            "raw_interactions": len(raw_interactions),
            "filtered_interactions": len(filtered),
            "train_interactions": len(train),
            "validation_interactions": len(validation),
            "test_interactions": len(test),
            "raw_items": len(raw_items),
            "processed_items": len(encoded_items),
            "users": len(user_mapping),
            "items": len(item_mapping),
            "dropped_short_sequence_users": dropped_short_sequence_users,
        },
    )

    if output_dir is not None:
        write_processed_artifacts(artifacts, output_dir)

    return artifacts


def k_core_filter(
    interactions: Iterable[Record],
    *,
    user_min_interactions: int = 5,
    item_min_interactions: int = 10,
) -> list[Record]:
    """Iteratively apply user/item k-core filtering."""

    _validate_threshold("user_min_interactions", user_min_interactions)
    _validate_threshold("item_min_interactions", item_min_interactions)

    current = [dict(row) for row in interactions]
    while True:
        user_counts = Counter(_raw_id(row, "user_id") for row in current)
        item_counts = Counter(_raw_id(row, "item_id") for row in current)
        next_rows = [
            row
            for row in current
            if user_counts[_raw_id(row, "user_id")] >= user_min_interactions
            and item_counts[_raw_id(row, "item_id")] >= item_min_interactions
        ]
        if len(next_rows) == len(current):
            return next_rows
        current = next_rows


def sort_interactions_chronologically(interactions: Iterable[Record]) -> list[Record]:
    """Sort interactions by user and timestamp."""

    return sorted(
        (dict(row) for row in interactions),
        key=lambda row: (_raw_id(row, "user_id"), _timestamp(row)),
    )


def encode_ids(
    interactions: Iterable[Record],
    item_metadata: Iterable[Record],
) -> tuple[list[Record], list[Record], dict[str, int], dict[str, int]]:
    """Encode raw user and item IDs to contiguous integer IDs."""

    interaction_rows = [dict(row) for row in interactions]
    item_rows = [dict(row) for row in item_metadata]
    user_mapping = {
        raw_user_id: idx
        for idx, raw_user_id in enumerate(sorted({_raw_id(row, "user_id") for row in interaction_rows}))
    }
    item_mapping = {
        raw_item_id: idx
        for idx, raw_item_id in enumerate(sorted({_raw_id(row, "item_id") for row in interaction_rows}))
    }

    encoded_interactions = []
    for row in interaction_rows:
        raw_user_id = _raw_id(row, "user_id")
        raw_item_id = _raw_id(row, "item_id")
        encoded = dict(row)
        encoded["raw_user_id"] = str(row.get("raw_user_id", raw_user_id))
        encoded["raw_item_id"] = str(row.get("raw_item_id", raw_item_id))
        encoded["user_id"] = user_mapping[raw_user_id]
        encoded["item_id"] = item_mapping[raw_item_id]
        encoded_interactions.append(encoded)

    encoded_items = []
    seen_items: set[int] = set()
    for row in item_rows:
        raw_item_id = _raw_id(row, "item_id")
        if raw_item_id not in item_mapping:
            continue
        encoded = dict(row)
        encoded["raw_item_id"] = str(row.get("raw_item_id", raw_item_id))
        encoded["item_id"] = item_mapping[raw_item_id]
        if encoded["item_id"] in seen_items:
            continue
        seen_items.add(encoded["item_id"])
        encoded_items.append(encoded)

    encoded_items.sort(key=lambda row: int(row["item_id"]))
    encoded_interactions.sort(key=lambda row: (int(row["user_id"]), _timestamp(row)))
    return encoded_interactions, encoded_items, user_mapping, item_mapping


def split_train_validation_test(interactions: Iterable[Record]) -> tuple[list[Record], list[Record], list[Record], int]:
    """Split each user's sorted history into train, validation, and test rows."""

    histories: dict[int, list[Record]] = defaultdict(list)
    for row in sort_interactions_chronologically(interactions):
        histories[int(row["user_id"])].append(dict(row))

    train: list[Record] = []
    validation: list[Record] = []
    test: list[Record] = []
    dropped_short_sequence_users = 0

    for user_id in sorted(histories):
        rows = histories[user_id]
        if len(rows) < 3:
            dropped_short_sequence_users += 1
            continue
        train.extend(rows[:-2])
        validation.append(rows[-2])
        test.append(rows[-1])

    return train, validation, test, dropped_short_sequence_users


def write_processed_artifacts(artifacts: ProcessedArtifacts, output_dir: str | Path) -> None:
    """Write processed artifacts to JSONL and JSON files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_path / "train.jsonl", artifacts.train)
    _write_jsonl(output_path / "validation.jsonl", artifacts.validation)
    _write_jsonl(output_path / "test.jsonl", artifacts.test)
    _write_jsonl(output_path / "item_metadata.jsonl", artifacts.item_metadata)
    _write_json(output_path / "user_mapping.json", artifacts.user_mapping)
    _write_json(output_path / "item_mapping.json", artifacts.item_mapping)
    _write_json(output_path / "stats.json", artifacts.stats)


def preprocess_files(
    *,
    interactions_path: str | Path,
    item_metadata_path: str | Path,
    output_dir: str | Path,
    user_min_interactions: int = 5,
    item_min_interactions: int = 10,
) -> ProcessedArtifacts:
    """Load local files and run the preprocessing pipeline."""

    return preprocess_dataset(
        load_interactions(interactions_path),
        load_item_metadata(item_metadata_path),
        output_dir=output_dir,
        user_min_interactions=user_min_interactions,
        item_min_interactions=item_min_interactions,
    )


def _load_records(path: str | Path) -> list[Record]:
    input_path = Path(path)
    suffixes = [suffix.lower() for suffix in input_path.suffixes]
    suffix = suffixes[-1] if suffixes else ""
    logical_suffix = suffixes[-2] if suffix == ".gz" and len(suffixes) >= 2 else suffix

    if logical_suffix == ".csv":
        with input_path.open("r", newline="", encoding="utf-8") as file:
            return [dict(row) for row in csv.DictReader(file)]
    if logical_suffix == ".json":
        with _open_text(input_path) as file:
            payload = json.load(file)
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            for key in ("records", "data", "interactions", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value]
        raise ValueError(f"JSON file must contain a list of records: {input_path}")
    if logical_suffix in {".jsonl", ".ndjson"}:
        with _open_text(input_path) as file:
            return [json.loads(line) for line in file if line.strip()]
    if logical_suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Reading parquet files requires pandas with parquet support installed") from exc
        return pd.read_parquet(input_path).to_dict(orient="records")

    raise ValueError(f"Unsupported file type: {''.join(input_path.suffixes)}")


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _normalize_interaction(row: Record) -> Record:
    normalized = dict(row)
    if "item_id" not in normalized and "parent_asin" in normalized:
        normalized["item_id"] = normalized["parent_asin"]
    if "raw_item_id" not in normalized and "asin" in normalized:
        normalized["raw_item_id"] = normalized["asin"]
    if "review_text" not in normalized and "text" in normalized:
        normalized["review_text"] = normalized["text"]
    return normalized


def _normalize_item(row: Record) -> Record:
    normalized = dict(row)
    if "item_id" not in normalized and "parent_asin" in normalized:
        normalized["item_id"] = normalized["parent_asin"]
    if "category" not in normalized:
        normalized["category"] = normalized.get("main_category") or _first_string(normalized.get("categories")) or ""
    normalized["description"] = _text_blob(normalized.get("description"))
    return normalized


def _first_string(value: object) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                return item
    return ""


def _text_blob(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return str(value)


def _require_fields(records: list[Record], fields: set[str], label: str) -> None:
    if not records:
        raise ValueError(f"{label} file is empty")
    missing = fields - records[0].keys()
    if missing:
        raise ValueError(f"{label} missing required fields: {sorted(missing)}")


def _write_jsonl(path: Path, records: Iterable[Record]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _raw_id(row: Record, key: str) -> str:
    if key not in row:
        raise ValueError(f"Record missing required field: {key}")
    return str(row[key])


def _timestamp(row: Record) -> int:
    if "timestamp" not in row:
        raise ValueError("Record missing required field: timestamp")
    return int(row["timestamp"])


def _validate_threshold(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def preprocess_beauty_input_dir(input_dir: str | Path, output_dir: str | Path) -> None:
    """Load raw Amazon Beauty files, apply 5-core filter, and write Parquet artifacts.

    Input files expected in input_dir:
      - All_Beauty.csv        (reviews: rating,title,text,images,asin,parent_asin,user_id,timestamp,...)
      - meta_All_Beauty.json.gz (JSONL.GZ: parent_asin, title, description, main_category)

    Output Parquet files written to output_dir:
      - interactions.parquet
      - train_sequences.parquet
      - val_labels.parquet
      - test_labels.parquet
      - item_meta.parquet
      - user_encoder.json
      - item_encoder.json
      - dataset_stats.json
    """

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("pandas is required for --input-dir mode") from exc

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load reviews (HuggingFace parquet or legacy CSV) ---
    reviews_parquet = input_path / "All_Beauty_reviews.parquet"
    reviews_csv = input_path / "All_Beauty.csv"

    if reviews_parquet.exists():
        logger.info("Loading reviews from %s", reviews_parquet)
        df_reviews = pd.read_parquet(reviews_parquet)
        df_reviews["item_id"] = df_reviews["parent_asin"].astype(str)
        df_reviews["user_id"] = df_reviews["user_id"].astype(str)
        df_reviews["timestamp"] = pd.to_numeric(df_reviews["timestamp"], errors="coerce")
        df_reviews = df_reviews.dropna(subset=["item_id", "user_id", "timestamp"])
        df_reviews["timestamp"] = df_reviews["timestamp"].astype(int)
        df_reviews["rating"] = pd.to_numeric(df_reviews.get("rating", 0), errors="coerce").fillna(0.0)
    elif reviews_csv.exists():
        logger.info("Loading reviews from %s", reviews_csv)
        df_reviews = pd.read_csv(reviews_csv, dtype=str)
        if "parent_asin" not in df_reviews.columns and df_reviews.columns[0] == "rating":
            df_reviews.columns = ["rating", "title", "text", "images", "asin", "parent_asin",
                                   "user_id", "timestamp", "verified_purchase"]
        df_reviews["item_id"] = df_reviews["parent_asin"].astype(str)
        df_reviews["timestamp"] = pd.to_numeric(df_reviews["timestamp"], errors="coerce")
        df_reviews = df_reviews.dropna(subset=["item_id", "user_id", "timestamp"])
        df_reviews["timestamp"] = df_reviews["timestamp"].astype(int)
        df_reviews["rating"] = pd.to_numeric(df_reviews["rating"], errors="coerce").fillna(0.0)
    else:
        raise FileNotFoundError(
            f"No review file found in {input_path}. "
            "Expected All_Beauty_reviews.parquet (from HuggingFace) or All_Beauty.csv."
        )
    logger.info("Loaded %d raw review rows", len(df_reviews))

    # --- Load item metadata (HuggingFace parquet or legacy JSONL.GZ) ---
    meta_parquet = input_path / "All_Beauty_meta.parquet"
    meta_gz = input_path / "meta_All_Beauty.json.gz"

    if meta_parquet.exists():
        logger.info("Loading item metadata from %s", meta_parquet)
        df_meta = pd.read_parquet(meta_parquet)
    elif meta_gz.exists():
        logger.info("Loading item metadata from %s", meta_gz)
        meta_records: list[dict] = []
        with gzip.open(meta_gz, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    meta_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        df_meta = pd.DataFrame(meta_records)
    else:
        raise FileNotFoundError(
            f"No metadata file found in {input_path}. "
            "Expected All_Beauty_meta.parquet (from HuggingFace) or meta_All_Beauty.json.gz."
        )
    logger.info("Loaded %d raw item meta rows", len(df_meta))

    # --- Build interaction records for k-core filtering ---
    interactions_raw: list[Record] = df_reviews[["user_id", "item_id", "rating", "timestamp"]].rename(
        columns={"user_id": "user_id", "item_id": "item_id"}
    ).to_dict(orient="records")

    logger.info("Applying 5-core filtering ...")
    interactions_filtered = k_core_filter(
        interactions_raw,
        user_min_interactions=5,
        item_min_interactions=5,
    )
    logger.info("After 5-core: %d interactions", len(interactions_filtered))

    # --- Sort by timestamp ---
    interactions_filtered.sort(key=lambda r: (str(r["user_id"]), int(r["timestamp"])))

    # --- Encode IDs: users 0-indexed, items 1-indexed ---
    all_user_ids = sorted({str(r["user_id"]) for r in interactions_filtered})
    all_item_ids = sorted({str(r["item_id"]) for r in interactions_filtered})
    user_encoder: dict[str, int] = {uid: idx for idx, uid in enumerate(all_user_ids)}
    item_encoder: dict[str, int] = {iid: idx + 1 for idx, iid in enumerate(all_item_ids)}
    n_users = len(user_encoder)
    n_items = len(item_encoder)
    logger.info("Users: %d | Items: %d", n_users, n_items)

    # --- Compute total interaction count per user for cold_start flag ---
    user_total_counts: Counter = Counter(str(r["user_id"]) for r in interactions_filtered)

    # --- Build encoded interaction list ---
    encoded_interactions = []
    for r in interactions_filtered:
        uid_str = str(r["user_id"])
        iid_str = str(r["item_id"])
        encoded_interactions.append({
            "user_id": user_encoder[uid_str],
            "item_id": item_encoder[iid_str],
            "rating": float(r["rating"]),
            "timestamp": int(r["timestamp"]),
            "cold_start": user_total_counts[uid_str] <= 5,
        })

    df_interactions = pd.DataFrame(encoded_interactions)
    df_interactions["user_id"] = df_interactions["user_id"].astype(int)
    df_interactions["item_id"] = df_interactions["item_id"].astype(int)

    # --- Build per-user histories sorted by timestamp ---
    histories: dict[int, list[dict]] = defaultdict(list)
    for row in encoded_interactions:
        histories[row["user_id"]].append(row)

    for uid in histories:
        histories[uid].sort(key=lambda r: r["timestamp"])

    # --- Split: last = test, second-to-last = val, rest = train ---
    train_sequences: list[dict] = []
    val_labels: list[dict] = []
    test_labels: list[dict] = []

    seq_lengths = []
    for uid in sorted(histories):
        hist = histories[uid]
        if len(hist) < 3:
            continue
        seq = [r["item_id"] for r in hist[:-2]]
        train_sequences.append({"user_id": uid, "sequence": seq})
        val_labels.append({"user_id": uid, "item_id": hist[-2]["item_id"]})
        test_labels.append({"user_id": uid, "item_id": hist[-1]["item_id"]})
        seq_lengths.append(len(seq))

    avg_seq_len = float(sum(seq_lengths)) / max(len(seq_lengths), 1)
    n_interactions = len(encoded_interactions)
    sparsity = 1.0 - n_interactions / max(n_users * n_items, 1)

    # --- Build item_meta parquet ---
    meta_by_asin: dict[str, dict] = {}
    if "parent_asin" in df_meta.columns:
        for _, row in df_meta.iterrows():
            pa = str(row.get("parent_asin", ""))
            if pa:
                desc_raw = row.get("description", "")
                if isinstance(desc_raw, list):
                    desc_str = " ".join(str(x) for x in desc_raw if x)
                else:
                    desc_str = str(desc_raw) if desc_raw else ""
                meta_by_asin[pa] = {
                    "asin": pa,
                    "title": str(row.get("title", "") or ""),
                    "category": str(row.get("main_category", "") or ""),
                    "description": desc_str,
                }

    item_meta_rows = []
    for asin, encoded_id in item_encoder.items():
        meta = meta_by_asin.get(asin, {})
        item_meta_rows.append({
            "item_id": encoded_id,
            "asin": asin,
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "description": meta.get("description", ""),
        })
    item_meta_rows.sort(key=lambda r: r["item_id"])
    df_item_meta = pd.DataFrame(item_meta_rows)
    df_item_meta["item_id"] = df_item_meta["item_id"].astype(int)

    # --- Write outputs ---
    logger.info("Writing interactions.parquet (%d rows)", len(df_interactions))
    df_interactions.to_parquet(output_path / "interactions.parquet", index=False)

    logger.info("Writing train_sequences.parquet (%d users)", len(train_sequences))
    df_train = pd.DataFrame(train_sequences)
    df_train.to_parquet(output_path / "train_sequences.parquet", index=False)

    logger.info("Writing val_labels.parquet (%d rows)", len(val_labels))
    pd.DataFrame(val_labels).to_parquet(output_path / "val_labels.parquet", index=False)

    logger.info("Writing test_labels.parquet (%d rows)", len(test_labels))
    pd.DataFrame(test_labels).to_parquet(output_path / "test_labels.parquet", index=False)

    logger.info("Writing item_meta.parquet (%d items)", len(item_meta_rows))
    df_item_meta.to_parquet(output_path / "item_meta.parquet", index=False)

    with (output_path / "user_encoder.json").open("w") as f:
        json.dump(user_encoder, f, indent=2)
    with (output_path / "item_encoder.json").open("w") as f:
        json.dump(item_encoder, f, indent=2)

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "avg_seq_len": round(avg_seq_len, 4),
        "sparsity": round(sparsity, 6),
    }
    with (output_path / "dataset_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Dataset stats: %s", stats)
    logger.info("Preprocessing complete. Artifacts written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SeqRec interaction data.")
    parser.add_argument("--input-dir", default=None, help="Directory with raw Amazon Beauty files (All_Beauty.csv + meta_All_Beauty.json.gz)")
    parser.add_argument("--interactions", default=None, help="Path to interactions CSV/JSON/JSONL/Parquet")
    parser.add_argument("--items", default=None, help="Path to item metadata CSV/JSON/JSONL/Parquet")
    parser.add_argument("--output-dir", required=True, help="Directory for processed artifacts")
    parser.add_argument("--user-min-interactions", type=int, default=5)
    parser.add_argument("--item-min-interactions", type=int, default=10)
    args = parser.parse_args()

    if args.input_dir is not None:
        # Beauty-specific path
        preprocess_beauty_input_dir(args.input_dir, args.output_dir)
    else:
        if args.interactions is None or args.items is None:
            parser.error("--interactions and --items are required when --input-dir is not provided")
        artifacts = preprocess_files(
            interactions_path=args.interactions,
            item_metadata_path=args.items,
            output_dir=args.output_dir,
            user_min_interactions=args.user_min_interactions,
            item_min_interactions=args.item_min_interactions,
        )
        print(json.dumps(artifacts.stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
