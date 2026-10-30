"""Cache the published withqwerty Premier League pass-map season files.

The source publishes one static JSON file per club.  This script requests each
file once, with a pause, and stores a compact local snapshot for DNA comparison.
It supplements (rather than replaces) Understat: it covers pass progression,
crossing and xT but has no carries, duels, recoveries or PPDA.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "passmaps"
BASE_URL = "https://withqwerty.com/data/passmaps"
HEADERS = {"User-Agent": "Sunderland scouting research prototype (cached public-data refresh)"}
CLUBS = {
    "arsenal": "Arsenal", "aston-villa": "Aston Villa", "bournemouth": "Bournemouth", "brentford": "Brentford",
    "brighton": "Brighton", "burnley": "Burnley", "chelsea": "Chelsea", "crystal-palace": "Crystal Palace",
    "everton": "Everton", "fulham": "Fulham", "leeds-united": "Leeds", "liverpool": "Liverpool",
    "manchester-city": "Manchester City", "manchester-united": "Manchester United", "newcastle-united": "Newcastle",
    "nottingham-forest": "Nottingham Forest", "sunderland": "Sunderland", "tottenham": "Tottenham",
    "west-ham": "West Ham", "wolves": "Wolves",
}


def fetch(slug: str) -> dict:
    response = requests.get(f"{BASE_URL}/{slug}.json", headers=HEADERS, timeout=45)
    response.raise_for_status()
    return response.json()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    sunderland_data: dict | None = None
    failures: dict[str, str] = {}
    for index, (slug, expected_name) in enumerate(CLUBS.items()):
        try:
            data = fetch(slug)
            stats = data.get("seasonAggregate", {}).get("stats", {})
            match_count = len(data.get("matches", []))
            per90_divisor = max(match_count * 90, 1)
            rows.append({
                "Team": data.get("clubName", expected_name), "Slug": slug, "Season": data.get("season"), "Matches": match_count,
                "Passes": stats.get("totalPasses"), "Successful passes": stats.get("successfulPasses"),
                "Pass completion %": round(100 * stats.get("completionRate", 0), 2), "xT generated": stats.get("totalXtGenerated"),
                "Progressive passes": stats.get("progressivePasses"), "Long balls": stats.get("longBalls"),
                "Crosses": stats.get("crosses"), "Through balls": stats.get("throughBalls"), "Switches of play": stats.get("switchesOfPlay"),
                "Passes per 90": round(90 * stats.get("totalPasses", 0) / per90_divisor, 2),
                "Progressive passes per 90": round(90 * stats.get("progressivePasses", 0) / per90_divisor, 2),
                "Crosses per 90": round(90 * stats.get("crosses", 0) / per90_divisor, 2),
                "xT per 90": round(90 * stats.get("totalXtGenerated", 0) / per90_divisor, 3),
            })
            if slug == "sunderland":
                sunderland_data = data
        except (requests.RequestException, ValueError) as exc:
            failures[slug] = str(exc)
        if index < len(CLUBS) - 1:
            sleep(1.25)

    if sunderland_data is None:
        raise RuntimeError("Sunderland data did not download; no files were replaced.")

    metrics = pd.DataFrame(rows).sort_values("Team").reset_index(drop=True)
    players = pd.DataFrame(sunderland_data["seasonAggregate"]["players"])
    match_rows = []
    for match in sunderland_data.get("matches", []):
        stats = match.get("network", {}).get("stats", {})
        match_rows.append({
            "Date": match.get("date"), "Gameweek": match.get("gameweek"), "Opponent": match.get("opponent"), "Home": match.get("isHome"),
            "Score": match.get("score"), "Result": match.get("result"), "Passes": stats.get("totalPasses"),
            "Successful passes": stats.get("successfulPasses"), "Pass completion %": round(100 * stats.get("completionRate", 0), 2),
            "xT generated": stats.get("totalXtGenerated"), "Progressive passes": stats.get("progressivePasses"),
            "Long balls": stats.get("longBalls"), "Crosses": stats.get("crosses"), "Through balls": stats.get("throughBalls"), "Switches of play": stats.get("switchesOfPlay"),
        })

    metrics.to_csv(OUTPUT_DIR / "premier_league_pass_metrics.csv", index=False)
    players.to_csv(OUTPUT_DIR / "sunderland_passmap_players.csv", index=False)
    pd.DataFrame(match_rows).to_csv(OUTPUT_DIR / "sunderland_passmap_matches.csv", index=False)
    (OUTPUT_DIR / "source_status.json").write_text(json.dumps({
        "source": "withqwerty public pass-map data (source page describes WhoScored/Opta event data)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": "https://withqwerty.com/stories/sunderland/passmaps/",
        "teams_downloaded": len(rows), "failures": failures,
        "limits": ["No carries or progressive carries", "No duels, aerials, recoveries or PPDA", "Use one cached refresh rather than high-frequency polling"],
    }, indent=2), encoding="utf-8")
    print(f"Saved pass-map snapshots for {len(rows)} clubs; failed: {len(failures)}")


if __name__ == "__main__":
    main()
