"""Resumable, verification-first EFEM full-attribute importer.

The existing striker file already supplies an exact EFEM player URL for every
record. This importer never searches by name: it fetches only that source URL,
checks the returned player name, and writes a row only when every detailed
attribute group is present. It deliberately does not modify the Streamlit app.
"""

from __future__ import annotations

import argparse
import csv
import html as html_module
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = BASE_DIR / "data" / "strikers.csv"
OUTPUT_DIR = BASE_DIR / "data" / "efem_full_attributes"
OUTPUT_PATH = OUTPUT_DIR / "efem_striker_full_attributes.csv"
FAILURES_PATH = OUTPUT_DIR / "efem_full_attribute_failures.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

ATTRIBUTE_GROUPS = [
    "aerialAttribute", "ballControlAttribute", "defendingAttribute", "goalkeepingAttribute",
    "mentalAttribute", "personalityAttribute", "physicalAttribute", "playmakingAttribute",
    "setPieceAttribute", "shootingAttribute",
]
ATTRIBUTE_LABELS = {
    "jumpingReach": "Jumping Reach", "heading": "Heading", "bravery": "Bravery",
    "firstTouch": "First Touch", "dribbling": "Dribbling", "agility": "Agility", "balance": "Balance", "technique": "Technique", "offTheBall": "Off The Ball",
    "marking": "Marking", "tackling": "Tackling", "positioning": "Positioning",
    "aerialReach": "Aerial Reach", "communication": "Communication", "commandOfArea": "Command of Area", "eccentricity": "Eccentricity", "handling": "Handling", "kicking": "Kicking", "oneVsOne": "One vs One", "punching": "Punching", "reflexes": "Reflexes", "rushingOut": "Rushing Out", "throwing": "Throwing",
    "workrate": "Work Rate", "teamwork": "Teamwork", "leadership": "Leadership", "flair": "Flair", "determination": "Determination", "decisions": "Decisions", "concentration": "Concentration", "composure": "Composure", "anticipation": "Anticipation", "aggression": "Aggression", "importantMatches": "Important Matches", "consistency": "Consistency",
    "adaptability": "Adaptability", "compliance": "Compliance", "ambition": "Ambition", "versatility": "Versatility", "temperament": "Temperament", "sportsmanship": "Sportsmanship", "professionalism": "Professionalism", "pressure": "Pressure", "loyalty": "Loyalty", "fairness": "Fairness",
    "acceleration": "Acceleration", "naturalFitness": "Natural Fitness", "pace": "Pace", "strength": "Strength", "stamina": "Stamina", "injuryResistance": "Injury Resistance",
    "crossing": "Crossing", "passing": "Passing", "vision": "Vision",
    "corners": "Corners", "freekicks": "Free Kicks", "longThrow": "Long Throw", "penalties": "Penalties",
    "finishing": "Finishing", "longshots": "Long Shots",
}
OUTPUT_FIELDS = ["row_id", "Name", "Club", "player_url", "efem_name", *ATTRIBUTE_LABELS.values(), "retrieved_at_utc"]
FAILURE_FIELDS = ["row_id", "Name", "Club", "player_url", "reason", "retrieved_at_utc"]
HEADERS = {"User-Agent": "SunderlandScoutingData/1.0 (resumable public-data importer)"}


def normalise_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def extract_page_name(html: str) -> str | None:
    match = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return html_module.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()


def extract_attributes(html: str) -> dict[str, int] | None:
    values: dict[str, int] = {}
    for group in ATTRIBUTE_GROUPS:
        pattern = rf'\\"{group}\\":\{{(.*?)\}}'
        match = re.search(pattern, html, flags=re.DOTALL)
        if not match:
            return None
        members = re.findall(r'\\"([A-Za-z0-9]+)\\":(\d+)', match.group(1))
        if not members:
            return None
        for raw_name, raw_value in members:
            if raw_name not in ATTRIBUTE_LABELS:
                return None
            value = int(raw_value)
            if not 0 <= value <= 100:
                return None
            values[ATTRIBUTE_LABELS[raw_name]] = value
    if set(values) != set(ATTRIBUTE_LABELS.values()):
        return None
    return values


def read_seen(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        return set(pd.read_csv(path, usecols=["row_id"], dtype={"row_id": str})["row_id"].dropna().astype(str))
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return set()


def append_row(path: Path, fields: list[str], row: dict[str, object]) -> None:
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def write_manifest(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch verified EFEM attribute groups for the existing striker database.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of pending records to process; 0 means all.")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between public page requests (default: 1.0).")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--retry-failures", action="store_true", help="Re-check previously rejected records after an importer fix.")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of independent workers; each writes to its own shard file.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based index of this worker's shard.")
    args = parser.parse_args()

    if not SOURCE_PATH.exists():
        raise SystemExit(f"Missing source file: {SOURCE_PATH}")
    if args.delay < 0.5:
        raise SystemExit("Use a delay of at least 0.5 seconds to avoid overloading the public source.")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Shard index must be between 0 and shard count - 1.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_PATH, low_memory=False, dtype={"row_id": str})
    required = {"row_id", "Name", "Club", "player_url"}
    missing = required.difference(source.columns)
    if missing:
        raise SystemExit(f"Source file is missing columns: {', '.join(sorted(missing))}")

    source = source.dropna(subset=["row_id", "player_url"]).copy()
    source["player_url"] = source["player_url"].astype(str).str.strip()
    source = source[source["player_url"].str.startswith("https://efem.club/players/")]
    source = source.drop_duplicates(subset=["row_id"], keep="first")

    if args.shard_count == 1:
        active_output = OUTPUT_PATH
        active_failures = FAILURES_PATH
        active_manifest = MANIFEST_PATH
    else:
        shard_dir = OUTPUT_DIR / "shards"
        suffix = f"shard-{args.shard_index + 1:02d}-of-{args.shard_count:02d}"
        active_output = shard_dir / f"efem_striker_full_attributes_{suffix}.csv"
        active_failures = shard_dir / f"efem_full_attribute_failures_{suffix}.csv"
        active_manifest = shard_dir / f"manifest_{suffix}.json"

    completed = read_seen(OUTPUT_PATH) | read_seen(active_output)
    failed = read_seen(FAILURES_PATH) | read_seen(active_failures)
    if args.shard_count > 1:
        # Every worker must see completed records from the other shards when
        # resuming or retrying exceptions; otherwise it could re-fetch them.
        shard_dir = OUTPUT_DIR / "shards"
        for path in shard_dir.glob("efem_striker_full_attributes_shard-*.csv"):
            completed |= read_seen(path)
        for path in shard_dir.glob("efem_full_attribute_failures_shard-*.csv"):
            failed |= read_seen(path)
    if args.retry_failures:
        failed = set()
    pending = source[~source["row_id"].astype(str).isin(completed | failed)].copy()
    pending = pending.iloc[args.shard_index::args.shard_count].copy()
    if args.limit:
        pending = pending.head(args.limit)

    write_manifest(active_manifest,
        source="https://efem.club",
        imported_at_utc=datetime.now(timezone.utc).isoformat(),
        source_records=int(len(source)),
        completed_before_run=len(completed),
        failed_before_run=len(failed),
        pending_this_run=int(len(pending)),
        delay_seconds=args.delay,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        status="running",
    )

    session = requests.Session()
    session.headers.update(HEADERS)
    imported = 0
    rejected = 0
    for index, (_, player) in enumerate(pending.iterrows(), start=1):
        now = datetime.now(timezone.utc).isoformat()
        source_name = str(player["Name"])
        row_id = str(player["row_id"])
        url = str(player["player_url"])
        reason: str | None = None
        parsed: dict[str, int] | None = None
        page_name: str | None = None
        for attempt in range(args.retry + 1):
            try:
                response = session.get(url, timeout=args.timeout)
                if response.status_code != 200:
                    reason = f"HTTP {response.status_code}"
                else:
                    page_name = extract_page_name(response.text)
                    if not page_name:
                        reason = "Player name not found on page"
                    elif normalise_name(page_name) != normalise_name(source_name):
                        reason = f"Name mismatch: source={source_name!r}; page={page_name!r}"
                    else:
                        parsed = extract_attributes(response.text)
                        reason = None if parsed else "One or more attribute groups were not found"
                if parsed or attempt == args.retry:
                    break
            except requests.RequestException as exc:
                reason = f"Request error: {type(exc).__name__}"
            time.sleep(min(8, args.delay * (attempt + 1)))

        common = {
            "row_id": row_id,
            "Name": source_name,
            "Club": str(player["Club"]),
            "player_url": url,
            "retrieved_at_utc": now,
        }
        if parsed:
            append_row(active_output, OUTPUT_FIELDS, {**common, "efem_name": page_name, **parsed})
            imported += 1
        else:
            append_row(active_failures, FAILURE_FIELDS, {**common, "reason": reason or "Unknown error"})
            rejected += 1

        if index % 25 == 0 or index == len(pending):
            print(f"Processed {index}/{len(pending)} | imported {imported} | rejected {rejected}", flush=True)
        if index < len(pending):
            time.sleep(args.delay)

    write_manifest(active_manifest,
        source="https://efem.club",
        imported_at_utc=datetime.now(timezone.utc).isoformat(),
        source_records=int(len(source)),
        completed_after_run=len(read_seen(OUTPUT_PATH) | read_seen(active_output)),
        rejected_after_run=len(read_seen(FAILURES_PATH) | read_seen(active_failures)),
        processed_this_run=int(len(pending)),
        imported_this_run=imported,
        rejected_this_run=rejected,
        delay_seconds=args.delay,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        status="complete",
    )
    print(f"Complete: imported {imported}, rejected {rejected}. Re-run the same command to resume remaining records.")


if __name__ == "__main__":
    main()
