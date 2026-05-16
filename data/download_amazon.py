"""Download Amazon Beauty dataset from HuggingFace (McAuley-Lab/Amazon-Reviews-2023)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_beauty_dataset(output_dir: str | Path, *, overwrite: bool = False) -> dict[str, Path]:
    """Download Amazon All_Beauty reviews and metadata via HuggingFace datasets library."""

    try:
        from datasets import load_dataset
    except ImportError:
        print("[error] 'datasets' library not found. Install with: pip install datasets", file=sys.stderr)
        raise SystemExit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    reviews_path = out / "All_Beauty_reviews.parquet"
    meta_path = out / "All_Beauty_meta.parquet"
    results: dict[str, Path] = {}

    if reviews_path.exists() and not overwrite:
        print(f"[skip] {reviews_path.name} already exists")
    else:
        print("Downloading Amazon Beauty reviews from HuggingFace (McAuley-Lab/Amazon-Reviews-2023)...")
        reviews = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_review_All_Beauty",
            split="full",
            trust_remote_code=True,
        )
        print(f"  {len(reviews):,} reviews loaded")
        reviews.to_parquet(str(reviews_path))
        print(f"[ok] Saved → {reviews_path}")
    results["reviews"] = reviews_path

    if meta_path.exists() and not overwrite:
        print(f"[skip] {meta_path.name} already exists")
    else:
        print("Downloading Amazon Beauty metadata from HuggingFace...")
        meta = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_All_Beauty",
            split="full",
            trust_remote_code=True,
        )
        print(f"  {len(meta):,} items loaded")
        meta.to_parquet(str(meta_path))
        print(f"[ok] Saved → {meta_path}")
    results["meta"] = meta_path

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Amazon All_Beauty dataset via HuggingFace.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where files will be saved (e.g. data/raw/beauty)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even if files already exist")
    args = parser.parse_args()
    paths = download_beauty_dataset(args.output_dir, overwrite=args.overwrite)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
