"""Preprocess Steam dataset files from McAuley Lab into SeqRec pipeline format.

Input files (from data/download_steam.py):
  - australian_user_reviews.json.gz   user-level records with nested reviews list
  - steam_games.json.gz               game metadata (one JSON object per line)

Output (data/processed/steam/):
  train.jsonl, validation.jsonl, test.jsonl, item_metadata.jsonl,
  user_mapping.json, item_mapping.json, stats.json

Usage:
    python data/preprocess_steam.py \
        --reviews  data/raw/steam/australian_user_reviews.json.gz \
        --games    data/raw/steam/steam_games.json.gz \
        --output-dir data/processed/steam
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.preprocess import preprocess_dataset

Record = dict[str, Any]

# Month abbreviation → number for "Posted Month Day, Year." parsing.
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Formats tried in order for the "posted" field.
_DATE_FORMATS = [
    "%B %d, %Y",    # November 5, 2011
    "%B %d, %Y @ %I:%M%p",  # November 5, 2011 @ 4:20pm
    "%b %d, %Y",    # Nov 5, 2011
]


def _parse_posted(posted: str) -> int | None:
    """Parse 'Posted November 5, 2011.' → Unix timestamp (seconds).

    Returns None if the date cannot be parsed (interaction will be dropped).
    """
    text = posted.strip()
    # Strip leading "Posted " and trailing punctuation.
    if text.lower().startswith("posted "):
        text = text[len("posted "):]
    text = text.rstrip(".").strip()
    # Remove sub-day "@" annotations that vary in format.
    if " @ " in text:
        text = text.split(" @ ")[0].strip()

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def _load_reviews(path: Path) -> list[Record]:
    """Parse australian_user_reviews.json.gz into flat (user_id, item_id, timestamp) records.

    Each line is a user object: {"user_id": "...", "reviews": [...], ...}
    Each nested review has "item_id" (as string) and "posted" date string.
    Duplicate (user, item) pairs are resolved by keeping the earliest timestamp.
    """
    seen: dict[tuple[str, str], int] = {}  # (user_id, item_id) → earliest timestamp
    review_text_map: dict[tuple[str, str], str] = {}

    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Steam files occasionally use Python literal syntax — fall back.
                try:
                    obj = ast.literal_eval(line)
                except Exception:
                    continue

            user_id = str(obj.get("user_id") or obj.get("steam_id") or "")
            if not user_id:
                continue

            for review in obj.get("reviews") or []:
                item_id = str(review.get("item_id") or "").strip()
                if not item_id:
                    continue

                posted = str(review.get("posted") or "")
                ts = _parse_posted(posted) if posted else None
                if ts is None:
                    # Fall back to year 2000 sentinel — still usable for ordering.
                    ts = 946_684_800

                key = (user_id, item_id)
                if key not in seen or ts < seen[key]:
                    seen[key] = ts
                    review_text_map[key] = str(review.get("review") or "")

    interactions: list[Record] = [
        {
            "user_id": uid,
            "item_id": iid,
            "timestamp": ts,
            "review_text": review_text_map[(uid, iid)],
        }
        for (uid, iid), ts in seen.items()
    ]
    return interactions


def _load_games(path: Path) -> list[Record]:
    """Parse steam_games.json.gz into item metadata records.

    Each line is either a JSON object or a Python dict literal (known quirk of the
    McAuley Lab steam_games.json file).  Malformed lines are silently skipped.
    """
    items: list[Record] = []
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line in ("{}", "[]"):
                continue
            obj: dict | None = None
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(line)
                except Exception:
                    pass
            if not obj or not isinstance(obj, dict):
                continue

            item_id = str(obj.get("id") or obj.get("app_id") or "").strip()
            if not item_id:
                continue

            title = str(obj.get("app_name") or obj.get("title") or obj.get("name") or "").strip()
            genres = obj.get("genre") or obj.get("genres") or []
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(",") if g.strip()]
            tags = obj.get("popular_tags") or obj.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            desc_parts: list[str] = []
            for key in ("game_description", "desc_snippet", "description"):
                val = obj.get(key)
                if val and str(val).strip():
                    desc_parts.append(str(val).strip())
                    break

            # Build rich description: narrative + genre + tags for BGE text enrichment.
            description_text = " ".join(desc_parts)
            if genres:
                description_text += " Genres: " + ", ".join(str(g) for g in genres) + "."
            if tags:
                description_text += " Tags: " + ", ".join(str(t) for t in tags[:15]) + "."

            price = str(obj.get("original_price") or obj.get("price") or "").strip()

            items.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "description": description_text,
                    "category": genres[0] if genres else "",
                    "price": price,
                }
            )

    return items


def preprocess_steam(
    *,
    reviews_path: str | Path,
    games_path: str | Path,
    output_dir: str | Path,
    user_min_interactions: int = 5,
    item_min_interactions: int = 5,
) -> dict[str, Any]:
    """Parse Steam files and write processed artifacts to output_dir."""

    print("Loading user reviews...")
    interactions = _load_reviews(Path(reviews_path))
    print(f"  {len(interactions):,} raw (user, game) interaction pairs")

    print("Loading game metadata...")
    item_metadata = _load_games(Path(games_path))
    print(f"  {len(item_metadata):,} game records")

    print(f"Preprocessing (user_min={user_min_interactions}, item_min={item_min_interactions})...")
    artifacts = preprocess_dataset(
        interactions,
        item_metadata,
        output_dir=output_dir,
        user_min_interactions=user_min_interactions,
        item_min_interactions=item_min_interactions,
    )

    stats = dict(artifacts.stats)
    stats["raw_interactions"] = len(interactions)
    stats["raw_games"] = len(item_metadata)
    out = Path(output_dir)
    with (out / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("\nStats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess McAuley Lab Steam dataset.")
    parser.add_argument("--reviews", required=True,
                        help="Path to australian_user_reviews.json.gz")
    parser.add_argument("--games", required=True,
                        help="Path to steam_games.json.gz")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for processed artifacts (e.g. data/processed/steam)")
    parser.add_argument("--user-min-interactions", type=int, default=5)
    parser.add_argument("--item-min-interactions", type=int, default=5)
    args = parser.parse_args()

    stats = preprocess_steam(
        reviews_path=args.reviews,
        games_path=args.games,
        output_dir=args.output_dir,
        user_min_interactions=args.user_min_interactions,
        item_min_interactions=args.item_min_interactions,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
