"""Build the free, evidence-backed subset of the original DNA schema.

Understat's published web data provides team match history (including PPDA and
deep completions), team formation splits, and league player totals. It does not
provide carries, progressive passes, touches in box, aerial duels or full pass
events. Those fields are deliberately marked unavailable instead of estimated.

Example:
    python scripts\sync_understat_dna.py --season 2025
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "understat"
BASE_URL = "https://understat.com"
HEADERS = {
    "User-Agent": "Sunderland scouting research prototype (public-data refresh)",
    "X-Requested-With": "XMLHttpRequest",
}


def get_json(path: str) -> dict:
    """Fetch slowly and fail loudly rather than retrying aggressively."""
    response = requests.get(f"{BASE_URL}/{path.lstrip('/')}", headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.json()


def ratio(numerator: object, denominator: object) -> float:
    denominator = float(denominator or 0)
    return float(numerator or 0) / denominator if denominator else np.nan


def ppda(row: dict, key: str = "ppda") -> float:
    values = row.get(key, {}) or {}
    return ratio(values.get("att"), values.get("def"))


def as_number(value: object) -> float:
    return pd.to_numeric(value, errors="coerce")


def percentile(series: pd.Series, value: float, higher_is_better: bool = True) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or pd.isna(value):
        return np.nan
    score = float((clean <= value).mean() * 100)
    return round(score if higher_is_better else 100 - score, 1)


def team_season_rates(league: dict) -> pd.DataFrame:
    rows = []
    for team in league.get("teams", {}).values():
        history = team.get("history", [])
        if not history:
            continue
        matches = len(history)
        totals = {
            "Team": team.get("title", "Unknown"),
            "Matches": matches,
            "xG per match": sum(as_number(match.get("xG")) for match in history) / matches,
            "xGA per match": sum(as_number(match.get("xGA")) for match in history) / matches,
            "Deep completions per match": sum(as_number(match.get("deep")) for match in history) / matches,
            "Deep allowed per match": sum(as_number(match.get("deep_allowed")) for match in history) / matches,
            "PPDA": ratio(sum(as_number((match.get("ppda") or {}).get("att")) for match in history), sum(as_number((match.get("ppda") or {}).get("def")) for match in history)),
            "Opponent PPDA": ratio(sum(as_number((match.get("ppda_allowed") or {}).get("att")) for match in history), sum(as_number((match.get("ppda_allowed") or {}).get("def")) for match in history)),
            "Goals per match": sum(as_number(match.get("scored")) for match in history) / matches,
            "Goals conceded per match": sum(as_number(match.get("missed")) for match in history) / matches,
            "xPoints per match": sum(as_number(match.get("xpts")) for match in history) / matches,
        }
        rows.append(totals)
    return pd.DataFrame(rows).sort_values("Team").reset_index(drop=True)


def flatten_history(team: dict) -> pd.DataFrame:
    rows = []
    for match in team.get("history", []):
        rows.append({
            "Date": match.get("date"), "Home/Away": match.get("h_a"), "Result": match.get("result"),
            "xG": as_number(match.get("xG")), "xGA": as_number(match.get("xGA")),
            "npxG": as_number(match.get("npxG")), "npxGA": as_number(match.get("npxGA")),
            "Deep completions": as_number(match.get("deep")), "Deep allowed": as_number(match.get("deep_allowed")),
            "PPDA": ppda(match), "Opponent PPDA": ppda(match, "ppda_allowed"),
            "Goals": as_number(match.get("scored")), "Goals conceded": as_number(match.get("missed")),
            "xPoints": as_number(match.get("xpts")),
        })
    return pd.DataFrame(rows)


def formation_table(team: dict) -> pd.DataFrame:
    rows = []
    for formation, values in (team.get("statistics", {}).get("formation", {}) or {}).items():
        against = values.get("against", {}) or {}
        minutes = as_number(values.get("time"))
        rows.append({
            "Formation": formation, "Minutes": minutes, "UsageShare%": round(100 * minutes / 3420, 2),
            "Shots": as_number(values.get("shots")), "Goals": as_number(values.get("goals")), "xG": as_number(values.get("xG")),
            "ShotsAgainst": as_number(against.get("shots")), "GA": as_number(against.get("goals")), "xGA": as_number(against.get("xG")),
            "xG per 90": round(90 * as_number(values.get("xG")) / minutes, 2) if minutes else np.nan,
            "xGA per 90": round(90 * as_number(against.get("xG")) / minutes, 2) if minutes else np.nan,
        })
    return pd.DataFrame(rows).sort_values("Minutes", ascending=False).reset_index(drop=True)


def player_table(players: list[dict]) -> pd.DataFrame:
    wanted = ["player_name", "team_title", "position", "games", "time", "goals", "assists", "xG", "xA", "shots", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]
    frame = pd.DataFrame(players)
    columns = [column for column in wanted if column in frame.columns]
    frame = frame[columns].rename(columns={
        "player_name": "Name", "team_title": "Club", "position": "Position", "games": "Matches", "time": "Minutes",
        "goals": "Goals", "assists": "Assists", "shots": "Shots", "key_passes": "Key Passes",
        "npg": "Non-penalty Goals", "npxG": "Non-penalty xG", "xGChain": "xG Chain", "xGBuildup": "xG Buildup",
    })
    for column in frame.columns:
        if column not in {"Name", "Club", "Position"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["xG per 90"] = (90 * frame["xG"] / frame["Minutes"].replace(0, np.nan)).round(3)
    frame["xA per 90"] = (90 * frame["xA"] / frame["Minutes"].replace(0, np.nan)).round(3)
    return frame.sort_values(["Minutes", "Name"], ascending=[False, True]).reset_index(drop=True)


def dna_coverage(rates: pd.DataFrame, team_players: pd.DataFrame) -> pd.DataFrame:
    sunderland = rates[rates["Team"].str.casefold().eq("sunderland")].iloc[0]
    team_buildup = team_players.groupby("Club", dropna=False)["xG Buildup"].sum(min_count=1)
    team_chain = team_players.groupby("Club", dropna=False)["xG Chain"].sum(min_count=1)
    club_match = next((club for club in team_buildup.index if str(club).casefold() == "sunderland"), None)
    rows = [
        ("Build-up Control", percentile(team_buildup, team_buildup.get(club_match, np.nan)), "xG Buildup (team sum)", "partial"),
        ("Progression", percentile(team_chain, team_chain.get(club_match, np.nan)), "xG Chain (team sum)", "partial"),
        ("Final Third Access", np.nanmean([percentile(rates["xG per match"], sunderland["xG per match"]), percentile(rates["Deep completions per match"], sunderland["Deep completions per match"])]), "xG + deep completions", "supported"),
        ("Pressing / Regain", percentile(rates["PPDA"], sunderland["PPDA"], higher_is_better=False), "PPDA (lower is more intense)", "supported"),
        ("Defensive Control", np.nanmean([percentile(rates["xGA per match"], sunderland["xGA per match"], higher_is_better=False), percentile(rates["Deep allowed per match"], sunderland["Deep allowed per match"], higher_is_better=False)]), "xGA against + deep completions allowed", "supported"),
    ]
    return pd.DataFrame(rows, columns=["DNA pillar", "League percentile", "Free-data input", "Coverage"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025, help="Season start year: 2025 means 2025/26")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    league = get_json(f"getLeagueData/EPL/{args.season}")
    sleep(2)
    team = get_json(f"getTeamData/Sunderland/{args.season}")
    rates = team_season_rates(league)
    players = player_table(league.get("players", []))
    history = flatten_history(next(value for value in league["teams"].values() if value.get("title", "").casefold() == "sunderland"))
    formation = formation_table(team)
    coverage = dna_coverage(rates, players)

    rates.to_csv(OUTPUT_DIR / "premier_league_team_rates.csv", index=False)
    history.to_csv(OUTPUT_DIR / "sunderland_match_history.csv", index=False)
    formation.to_csv(OUTPUT_DIR / "sunderland_formation_summary.csv", index=False)
    players.to_csv(OUTPUT_DIR / "premier_league_player_metrics.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "sunderland_dna_coverage.csv", index=False)
    (OUTPUT_DIR / "source_status.json").write_text(json.dumps({
        "source": "Understat public web data", "season_start_year": args.season,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "urls": {"league": f"{BASE_URL}/league/EPL/{args.season}", "team": f"{BASE_URL}/team/Sunderland/{args.season}"},
        "important": "Unsupported pillars are intentionally blank. Do not backfill them with guessed values.",
    }, indent=2), encoding="utf-8")
    print(f"Saved Understat DNA files for {args.season}/{str(args.season + 1)[-2:]}")
    print(f"Team rows: {len(rates)} | Player rows: {len(players)} | Sunderland matches: {len(history)}")


if __name__ == "__main__":
    main()
