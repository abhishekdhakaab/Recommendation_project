"""Download Amazon Beauty dataset files for SeqRec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm


BEAUTY_REVIEWS_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/All_Beauty.csv"
)
BEAUTY_META_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz"
)

FILES = [
    (BEAUTY_REVIEWS_URL, "All_Beauty.csv"),
    (BEAUTY_META_URL, "meta_All_Beauty.json.gz"),
]


def download_file(
    url: str,
    destination: Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a single file with progress bar. Returns destination path."""

    if destination.exists() and not overwrite:
        print(f"[skip] {destination.name} already exists at {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0)) or None
        with (
            tmp_path.open("wb") as fout,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
                file=sys.stdout,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fout.write(chunk)
                    pbar.update(len(chunk))

        tmp_path.rename(destination)
        print(f"[ok] Saved {destination.name} → {destination}")
        return destination

    except requests.HTTPError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"[error] HTTP {exc.response.status_code} when downloading {url}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"[error] Failed to download {url}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def download_beauty_dataset(output_dir: str | Path, *, overwrite: bool = False) -> dict[str, Path]:
    """Download both Beauty files to output_dir."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    for url, filename in FILES:
        dest = output_path / filename
        results[filename] = download_file(url, dest, overwrite=overwrite)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Amazon All_Beauty dataset files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where downloaded files will be saved (e.g. data/raw/beauty)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    paths = download_beauty_dataset(args.output_dir, overwrite=args.overwrite)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
