# Sunderland AFC Hidden-Gem Striker Scouting Tool

This is a Sunderland AFC adaptation of the original Al Gharafa Streamlit tool. It preserves its seven-page workflow: executive overview, club DNA, current striker assessment, hidden-gem discovery, player card, comparison, similarity scouting and downloadable HTML reports.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local address Streamlit displays, normally `http://localhost:8501`.

## Sunderland changes

- Red, white and black Sunderland presentation theme with a neutral local scouting badge (not the official club crest).
- A 4-3-3 / 4-2-3-1 striker model that rewards channel running and first-line defensive work alongside box threat and link play.
- Current-striker benchmark group: Brian Brobbey, Wilson Isidor, Eliezer Mayenda and Timur Tutierov.
- Sunderland-specific DNA and formation tables replace the original Al Gharafa tables.

## Free data refresh — no Wyscout account needed

Run this before opening the app, and then periodically when you need a new snapshot:

```powershell
python scripts\sync_understat_dna.py --season 2025
python scripts\sync_open_data.py
python scripts\sync_passmap_data.py
streamlit run app.py
```

`sync_understat_dna.py` is the primary free replacement for the original Wyscout input layer. It creates:

- `sunderland_match_history.csv` — xG/xGA, PPDA, deep completions, goals and xPoints, match by match.
- `sunderland_formation_summary.csv` — formation usage, shots, goals, xG, xGA and their per-90 values.
- `premier_league_team_rates.csv` — comparable team context for all 20 Premier League clubs.
- `premier_league_player_metrics.csv` — 537-player performance layer with minutes, goals, assists, xG, xA, shots, key passes, xG Chain and xG Buildup.
- `sunderland_dna_coverage.csv` — the source-backed DNA pillars and their actual free-data inputs.

It currently supports Final Third Access, Pressing / Regain and Defensive Control directly, with partial source-backed measures for Build-up and Progression. Dimensions without a valid source are removed from the DNA view; they are not guessed.

`sync_passmap_data.py` adds pass-progression and width evidence from [withqwerty's published Sunderland pass-map data](https://withqwerty.com/stories/sunderland/passmaps/): passes, completion, progressive passes, xT, long balls, crosses, through balls and switches for all 20 Premier League teams and every Sunderland match. It uses one cached refresh rather than polling. Measures that the public sources cannot support are excluded from the DNA view.

The refresh uses public pages from [Statz](https://statz.ai) and writes these files to `data/open_data/`:

- `sunderland_team_matches.csv` — observed completed match totals.
- `sunderland_dna_open_snapshot.csv` — team baseline calculated from those matches: possession, passes, xG, shots, corners, tackles and fouls.
- `sunderland_current_strikers_open.csv` — observed public totals for players listed as ST in Sunderland's current squad.
- `premier_league_open_leaders.csv` — a partial market layer: the union of public Premier League top-10 attacking, defending and possession leaderboards.
- `data_status.json` — refresh timestamp, source URLs and limitations.

The DNA page now labels raw measures and proxies honestly. It does not turn them into pretend event-data scores. The Current ST page shows observed public output and deliberately avoids estimating pressing, link play, movement or tactical role from limited totals.

## Important limits

`data/strikers.csv` is still the original tool's legacy broad player universe. Its role scores feed the original hidden-gem calculations, so it must **not** be presented as a current, verified market database. The open Premier League leaders file is current but is a partial leaderboard union, not a complete player database.

Public match and player totals cannot replace Wyscout/Opta/StatsBomb for pressing actions, locations, sequences, comparable per-90 event profiles, contracts, availability or transfer feasibility. Use the free layer to set a fact-based baseline and create an initial watchlist; validate every target with video and a club-authorised source before any decision.

## Current squad checks

Sunderland's official site identifies Brobbey and Isidor as centre-forward options, names Mayenda in the matchday squad, and records Tutierov leading the line:

- [Brobbey and Isidor update](https://www.safc.com/news/2026/october/21/le-bris-confirms-brobbey-timeline/)
- [Isidor leads the line](https://www.safc.com/news/2026/october/10/le-bris-makes-three-sunderland-changes/)
- [Mayenda in Sunderland's squad](https://www.safc.com/news/2026/may/09/le-bris-makes-one-change-to-safc-side-to-face-man-utd/)
- [Tutierov leads the line](https://www.safc.com/news/2026/october/18/tutierov-to-make-premier-league-debut-at-afc-bournemouth/)
