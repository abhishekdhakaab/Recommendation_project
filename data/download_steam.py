"""Download Steam dataset files from McAuley Lab (UCSD).

Dataset home: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data

Files downloaded:
  - australian_user_reviews.json.gz   user reviews with timestamps
  - steam_games.json.gz               game metadata (title, description, genres)

Usage:
    python data/download_steam.py --output-dir data/raw/steam

If the automated download fails (the UCSD server is sometimes slow), the
script prints manual download instructions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request
import urllib.error


STEAM_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/steam"

FILES = {
    "australian_user_reviews.json.gz": f"{STEAM_BASE}/australian_user_reviews.json.gz",
    "steam_games.json.gz": "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
}

MANUAL_INSTRUCTIONS = """
Manual download instructions
─────────────────────────────
Visit: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
Download:
  • australian_user_reviews.json.gz
  • steam_games.json.gz
Place both files in: {output_dir}
"""


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    mb = downloaded / 1_048_576
    total_mb = total_size / 1_048_576
    print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_steam_files(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download all required Steam dataset files to ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}

    for filename, url in FILES.items():
        dest = out / filename
        if dest.exists() and not overwrite:
            print(f"  already exists, skipping: {dest}")
            downloaded[filename] = dest
            continue

        print(f"Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
            print()  # newline after progress bar
            print(f"  saved → {dest}  ({dest.stat().st_size / 1_048_576:.1f} MB)")
            downloaded[filename] = dest
        except (urllib.error.URLError, OSError) as exc:
            print(f"\n  FAILED: {exc}")
            print(MANUAL_INSTRUCTIONS.format(output_dir=out))
            sys.exit(1)

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Steam dataset files.")
    parser.add_argument("--output-dir", default="data/raw/steam",
                        help="Directory to save downloaded files (default: data/raw/steam)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even if file already exists")
    args = parser.parse_args()

    paths = download_steam_files(args.output_dir, overwrite=args.overwrite)
    print("\nReady:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
