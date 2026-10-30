"""Refresh the public-data layer used by the Sunderland scouting prototype.

This intentionally uses public web pages rather than any logged-in provider.  It is
an evidence layer, not a replacement for Wyscout / Opta / StatsBomb event data.
Run it sparingly (for example once a week) and review the resulting CSVs before
making recruitment decisions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from time import sleep

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "open_data"
SOURCE = "https://statz.ai"
HEADERS = {"User-Agent": "Sunderland scouting research prototype (public-data refresh)"}
URLS = {
    "team": f"{SOURCE}/team/sunderland",
    "squad": f"{SOURCE}/team/sunderland/squad",
    "league_attacking": f"{SOURCE}/competitions/premier-league/stats/players?group=attacking",
    "league_defending": f"{SOURCE}/competitions/premier-league/stats/players?group=defending",
    "league_possession": f"{SOURCE}/competitions/premier-league/stats/players?group=possession",
}


def fetch_tables(url: str) -> list[pd.DataFrame]:
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def first_table_with(tables: list[pd.DataFrame], required: set[str]) -> pd.DataFrame:
    for table in tables:
        if required.issubset(set(table.columns)):
            return table.copy()
    raise ValueError(f"Could not find table with columns: {sorted(required)}")


def number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")


def consolidate_leaderboards(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Join public top-10 leaderboards into one transparent, partial market file."""
    result: pd.DataFrame | None = None
    for table in tables:
        if not {"Player", "Team"}.issubset(table.columns) or len(table.columns) != 4:
            continue
        metric = [column for column in table.columns if column not in {"Rank", "Player", "Team"}]
        if len(metric) != 1:
            continue
        metric_name = metric[0]
        part = table[["Player", "Team", metric_name]].copy()
        part[metric_name] = number(part[metric_name])
        result = part if result is None else result.merge(part, on=["Player", "Team"], how="outer")
    return result.sort_values(["Player", "Team"]).reset_index(drop=True) if result is not None else pd.DataFrame()


def build_dna_snapshot(matches: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "Match sample": (len(matches), "matches", "Available completed matches on the source page"),
        "Possession": (number(matches["Possession"]).mean(), "%", "Territory / ball-control indicator"),
        "Passes": (number(matches["Passes"]).mean(), "per match", "Build-up volume indicator"),
        "Expected goals": (number(matches["Expected goals"]).mean(), "per match", "Chance-quality indicator"),
        "Shots": (number(matches["Shots"]).mean(), "per match", "Final-third volume indicator"),
        "Shots on target": (number(matches["Shots on target"]).mean(), "per match", "Shot-quality / execution indicator"),
        "Corners": (number(matches["Corners"]).mean(), "per match", "Set-piece / width proxy"),
        "Tackles": (number(matches["Tackles"]).mean(), "per match", "Defensive activity proxy; not a pressing count"),
        "Fouls": (number(matches["Fouls"]).mean(), "per match", "Defensive aggression / game-state context"),
    }
    return pd.DataFrame(
        [{"Metric": name, "Value": round(float(value), 2), "Unit": unit, "What it measures": note} for name, (value, unit, note) in metrics.items()]
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    team_tables = fetch_tables(URLS["team"])
    sleep(1.25)
    squad_tables = fetch_tables(URLS["squad"])

    matches = first_table_with(team_tables, {"Date", "Opponent", "Expected goals", "Possession", "Passes"})
    team_summary = first_table_with(team_tables, {"Scope", "P", "GF", "GA", "PPG"})
    squad = first_table_with(squad_tables, {"Player", "Position", "Mins", "Goals", "xG"})
    current_strikers = squad[squad["Position"].astype(str).str.upper().eq("ST")].copy()
    for column in ["Mins", "Goals", "Assists", "xG", "Shots on target", "Chances created", "Tackles"]:
        if column in current_strikers:
            current_strikers[column] = number(current_strikers[column])
    current_strikers["xG per 90"] = (90 * current_strikers["xG"] / current_strikers["Mins"].replace(0, pd.NA)).round(2)
    current_strikers["Goals per 90"] = (90 * current_strikers["Goals"] / current_strikers["Mins"].replace(0, pd.NA)).round(2)

    leaderboards: list[pd.DataFrame] = []
    for key in ["league_attacking", "league_defending", "league_possession"]:
        sleep(1.25)
        leaderboards.extend(fetch_tables(URLS[key]))
    leaders = consolidate_leaderboards(leaderboards)

    matches.to_csv(DATA_DIR / "sunderland_team_matches.csv", index=False)
    team_summary.to_csv(DATA_DIR / "sunderland_team_summary.csv", index=False)
    squad.to_csv(DATA_DIR / "sunderland_squad_stats.csv", index=False)
    current_strikers.to_csv(DATA_DIR / "sunderland_current_strikers_open.csv", index=False)
    build_dna_snapshot(matches).to_csv(DATA_DIR / "sunderland_dna_open_snapshot.csv", index=False)
    leaders.to_csv(DATA_DIR / "premier_league_open_leaders.csv", index=False)

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (DATA_DIR / "data_status.json").write_text(
        json.dumps(
            {
                "retrieved_at_utc": retrieved_at,
                "source": "Statz public web pages",
                "source_urls": URLS,
                "limitations": [
                    "The DNA snapshot is match-level public data and uses labelled proxies.",
                    "The league leaders file is a partial top-10 leaderboard union, not a complete recruitment database.",
                    "No pressing-event, player-location, contract, injury, or transfer-fee data is supplied.",
                    "Validate names, season scope and unusual values before any decision.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Open-data refresh complete: {retrieved_at}")
    print(f"Sunderland striker rows: {len(current_strikers)} | League leader rows: {len(leaders)}")


if __name__ == "__main__":
    main()
