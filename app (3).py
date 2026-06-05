from __future__ import annotations

import base64
import html
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

# -----------------------------
# App configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "strikers.csv"
LOGO_PATH = BASE_DIR / "assets" / "al_gharafa_logo.png"
DNA_SEASON_PATH = BASE_DIR / "data" / "dna_by_season.csv"
FORMATION_SUMMARY_PATH = BASE_DIR / "data" / "formation_summary.csv"
FORMATION_SEASON_PATH = BASE_DIR / "data" / "formation_by_season.csv"

NAVY = "#0B1F3A"
NAVY_2 = "#132B4F"
GOLD = "#D6A500"
GOLD_2 = "#F2C94C"
WHITE = "#FFFFFF"
OFFWHITE = "#F6F8FB"
MUTED = "#6B7280"
RED = "#C2410C"
BLUE = "#2454A6"
LIGHT_NAVY = "#D8E2F2"

st.set_page_config(
    page_title="Al Gharafa SC | Striker Scouting Tool",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {OFFWHITE} 0%, #FFFFFF 42%, #FFFFFF 100%);
        color: {NAVY};
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, {NAVY_2} 100%);
        border-right: 4px solid {GOLD};
    }}
    section[data-testid="stSidebar"] * {{
        color: {WHITE};
    }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {{
        background-color: {WHITE} !important;
        color: {NAVY} !important;
        border-color: rgba(214,165,0,0.55) !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] div[data-baseweb="select"] input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg,
    section[data-testid="stSidebar"] div[data-baseweb="select"] [role="button"] {{
        color: {NAVY} !important;
        fill: {NAVY} !important;
    }}
    section[data-testid="stSidebar"] input::placeholder,
    section[data-testid="stSidebar"] textarea::placeholder {{
        color: #334155 !important;
        opacity: 1 !important;
    }}
    div[data-baseweb="popover"] * {{
        color: {NAVY} !important;
    }}
    div[data-baseweb="popover"] [role="option"] {{
        background: {WHITE} !important;
    }}
    div[data-baseweb="popover"] [role="option"]:hover {{
        background: #EEF2F7 !important;
    }}
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{
        color: {WHITE} !important;
        font-weight: 700;
    }}
    .main-title {{
        font-size: 34px;
        font-weight: 900;
        color: {NAVY};
        margin-bottom: 0px;
        line-height: 1.05;
    }}
    .subtitle {{
        color: {MUTED};
        font-size: 15px;
        margin-top: 4px;
    }}
    .card {{
        background: {WHITE};
        border: 1px solid #E5E7EB;
        border-radius: 22px;
        padding: 20px;
        box-shadow: 0 12px 30px rgba(11,31,58,0.08);
        margin-bottom: 12px;
    }}
    .navy-card {{
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_2} 100%);
        color: {WHITE};
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 18px 35px rgba(11,31,58,0.20);
        border: 1px solid rgba(214,165,0,0.35);
    }}
    .metric-card {{
        background: {WHITE};
        border-radius: 20px;
        padding: 18px;
        border-top: 5px solid {GOLD};
        box-shadow: 0 10px 24px rgba(11,31,58,0.08);
        min-height: 112px;
    }}
    .metric-label {{color: {MUTED}; font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: .06em;}}
    .metric-value {{font-size: 30px; color: {NAVY}; font-weight: 900; margin-top: 6px;}}
    .tag {{
        display:inline-block;
        background: rgba(214,165,0,0.16);
        color: {NAVY};
        border: 1px solid rgba(214,165,0,0.45);
        border-radius: 999px;
        padding: 5px 10px;
        font-weight: 800;
        font-size: 12px;
        margin: 2px 3px 2px 0;
    }}
    .gold-line {{height:4px;background:{GOLD};border-radius:6px;margin:12px 0 22px 0;}}
    .small-note {{color:{MUTED};font-size:13px;}}
    .section-header {{
        font-size: 24px;
        font-weight: 900;
        color: {NAVY};
        margin-top: 4px;
        margin-bottom: 8px;
    }}
    div[data-testid="stDataFrame"] {{
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 26px rgba(11,31,58,0.08);
    }}
    .stTabs [data-baseweb="tab-list"] {{gap: 8px;}}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 999px;
        padding: 10px 16px;
        background: #EEF2F7;
        color: {NAVY};
        font-weight: 800;
    }}
    .stTabs [aria-selected="true"] {{
        background: {NAVY} !important;
        color: {WHITE} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clamp_series(s: pd.Series, low: float = 0, high: float = 100) -> pd.Series:
    return s.clip(lower=low, upper=high)


def clean_positions(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace(",", " ").strip()


def normalise_country_value(v: object) -> List[str]:
    if pd.isna(v):
        return []
    text = str(v).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "null", "-"}:
        return []
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    invalid = {"", "nan", "none", "unknown", "null", "-", "europe", "africa", "asia", "asia/oceania", "asia / oceania", "south america", "north america", "oceania", "other"}
    parts = re.split(r"[;,/|]+", text)
    return [p.strip() for p in parts if p.strip() and p.strip().lower() not in invalid]


def country_to_continent(country: str) -> str:
    # Lightweight mapping for common football nationalities. Unknown stays Other.
    europe = {
        "Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic","Denmark","England","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Israel","Italy","Kosovo","Latvia","Lithuania","Luxembourg","Malta","Moldova","Montenegro","Netherlands","North Macedonia","Northern Ireland","Norway","Poland","Portugal","Romania","Russia","Scotland","Serbia","Slovakia","Slovenia","Spain","Sweden","Switzerland","Turkey","Ukraine","Wales"
    }
    africa = {
        "Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon","Cape Verde","Central African Republic","Chad","Comoros","Congo","DR Congo","Democratic Republic of Congo","Egypt","Equatorial Guinea","Eritrea","Ethiopia","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau","Ivory Coast","Côte d'Ivoire","Kenya","Liberia","Libya","Madagascar","Mali","Mauritania","Morocco","Mozambique","Namibia","Niger","Nigeria","Rwanda","Senegal","Sierra Leone","Somalia","South Africa","Sudan","Tanzania","Togo","Tunisia","Uganda","Zambia","Zimbabwe"
    }
    asia = {
        "Afghanistan","Australia","Bahrain","Bangladesh","Cambodia","China","Chinese Taipei","Hong Kong","India","Indonesia","Iran","Iraq","Japan","Jordan","Kazakhstan","Kuwait","Kyrgyzstan","Lebanon","Malaysia","Myanmar","North Korea","Oman","Pakistan","Palestine","Philippines","Qatar","Saudi Arabia","Singapore","South Korea","Syria","Tajikistan","Thailand","Turkmenistan","United Arab Emirates","UAE","Uzbekistan","Vietnam","Yemen"
    }
    south_america = {"Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador","Paraguay","Peru","Uruguay","Venezuela"}
    north_america = {"Antigua and Barbuda","Bahamas","Barbados","Belize","Canada","Costa Rica","Cuba","Dominican Republic","El Salvador","Grenada","Guatemala","Haiti","Honduras","Jamaica","Mexico","Nicaragua","Panama","Puerto Rico","Trinidad and Tobago","United States","USA"}
    if country in europe: return "Europe"
    if country in africa: return "Africa"
    if country in asia: return "Asia/Oceania"
    if country in south_america: return "South America"
    if country in north_america: return "North America"
    return "Other"


def fig_layout(fig: go.Figure, height: int = 430, bottom_legend: bool = True) -> go.Figure:
    legend_cfg = dict(
        orientation="h",
        yanchor="top",
        y=-0.18,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
        tracegroupgap=6,
    ) if bottom_legend else dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color=NAVY, size=12),
        margin=dict(l=35, r=30, t=86, b=110 if bottom_legend else 55),
        legend=legend_cfg,
        title=dict(x=0.01, xanchor="left", y=0.97, font=dict(size=18, color=NAVY)),
    )
    return fig


def score_badge(value: float) -> str:
    if value >= 85:
        return "Elite"
    if value >= 75:
        return "Strong"
    if value >= 65:
        return "Good"
    if value >= 55:
        return "Monitor"
    return "Low"


def role_columns(df: pd.DataFrame) -> List[str]:
    roles = [
        "Advanced Forward",
        "Deep-Lying Forward",
        "Target Forward",
        "Poacher",
        "Complete Forward",
        "Trequartista (ST)",
        "Pressing Forward",
        "False Nine",
        "Second Striker",
        "Centre Forward",
        "Channel Forward",
        "Central-Outlet Centre Forward",
        "Splitting-Outlet Centre Forward",
        "Tracking Centre Forward",
    ]
    return [c for c in roles if c in df.columns]


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalise_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return text


ELITE_CLUB_KEYWORDS = {
    "real madrid", "barcelona", "man city", "manchester city", "arsenal", "liverpool", "chelsea",
    "man utd", "manchester united", "tottenham", "bayern", "bayern munich", "psg", "paris saint germain",
    "juventus", "inter", "internazionale", "ac milan", "milan", "napoli", "atletico madrid",
    "dortmund", "borussia dortmund", "leverkusen", "bayer leverkusen", "rb leipzig", "benfica", "porto",
    "sporting", "sporting cp", "ajax", "psv", "feyenoord", "al hilal", "al nassr", "al ittihad",
    "al ahli", "al ahly", "flamengo", "palmeiras", "river plate", "boca juniors"
}

FAMOUS_PLAYER_KEYWORDS = {
    "haaland", "erling haaland", "mbappe", "kylian mbappe", "lamine yamal", "vinicius", "vinicius junior",
    "bellingham", "jude bellingham", "rodrygo", "endrick", "messi", "lionel messi", "ronaldo", "cristiano ronaldo",
    "neymar", "harry kane", "kane", "salah", "mohamed salah", "osimhen", "victor osimhen", "lautaro",
    "lautaro martinez", "isak", "alexander isak", "sesko", "benjamin sesko", "gyokeres", "viktor gyokeres",
    "vitor roque", "vini", "rashford", "marcus rashford", "darwin nunez", "nunez", "julian alvarez",
    "joao felix", "dusan vlahovic", "vlahovic", "lewandowski", "robert lewandowski", "son heung min"
}


def contains_keyword(text: object, keywords: set[str]) -> bool:
    t = normalise_text(text)
    return any(k in t for k in keywords)


def obvious_player_flags(df: pd.DataFrame) -> pd.DataFrame:
    club_norm = df["Club"].apply(normalise_text) if "Club" in df.columns else pd.Series("", index=df.index)
    name_norm = df["Name"].apply(normalise_text) if "Name" in df.columns else pd.Series("", index=df.index)
    df["Elite Club Flag"] = club_norm.apply(lambda x: any(k in x for k in ELITE_CLUB_KEYWORDS))
    df["Famous Name Flag"] = name_norm.apply(lambda x: any(k in x for k in FAMOUS_PLAYER_KEYWORDS))
    df["Obvious Elite Player"] = df["Elite Club Flag"] | df["Famous Name Flag"] | ((df["Generic Role Level"] >= 83) & (df["Al Gharafa ST Fit"] >= 80))
    return df

# -----------------------------
# Data loading and scoring
# -----------------------------
@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Standard identity columns
    if "Name" not in df.columns and "player_name" in df.columns:
        df["Name"] = df["player_name"]
    if "Club" not in df.columns:
        df["Club"] = "Unknown"
    if "Age" not in df.columns:
        df["Age"] = np.nan
    if "Positions" not in df.columns:
        df["Positions"] = "ST"

    df["Name"] = df["Name"].fillna(df.get("player_name", "Unknown")).astype(str)
    df["Club"] = df["Club"].fillna("Unknown").astype(str)
    df["Age"] = safe_num(df["Age"])
    df["Positions"] = df["Positions"].apply(clean_positions)

    roles = role_columns(df)
    for c in roles:
        df[c] = safe_num(df[c]).fillna(0).clip(0, 100)

    # Ensure every scoring input exists.
    for c in [
        "Advanced Forward","Deep-Lying Forward","Target Forward","Poacher","Complete Forward","Trequartista (ST)",
        "Pressing Forward","False Nine","Second Striker","Centre Forward","Channel Forward",
        "Central-Outlet Centre Forward","Splitting-Outlet Centre Forward","Tracking Centre Forward"
    ]:
        if c not in df.columns:
            df[c] = 0.0

    # Club-specific striker phases.
    df["Box Threat"] = clamp_series(
        0.34*df["Poacher"] + 0.26*df["Advanced Forward"] + 0.18*df["Complete Forward"] + 0.14*df["Centre Forward"] + 0.08*df["Channel Forward"]
    )
    df["Link-up Play"] = clamp_series(
        0.32*df["Deep-Lying Forward"] + 0.25*df["Complete Forward"] + 0.15*df["False Nine"] + 0.12*df["Trequartista (ST)"] + 0.10*df["Second Striker"] + 0.06*df["Centre Forward"]
    )
    df["Channel Running"] = clamp_series(
        0.36*df["Channel Forward"] + 0.25*df["Advanced Forward"] + 0.18*df["Splitting-Outlet Centre Forward"] + 0.11*df["Complete Forward"] + 0.10*df["Centre Forward"]
    )
    df["Wide Combination"] = clamp_series(
        0.24*df["Complete Forward"] + 0.21*df["Deep-Lying Forward"] + 0.16*df["Second Striker"] + 0.15*df["Trequartista (ST)"] + 0.13*df["Channel Forward"] + 0.11*df["False Nine"]
    )
    df["Aerial Outlet"] = clamp_series(
        0.55*df["Target Forward"] + 0.25*df["Central-Outlet Centre Forward"] + 0.10*df["Centre Forward"] + 0.10*df["Complete Forward"]
    )
    df["Defensive Work"] = clamp_series(
        0.50*df["Pressing Forward"] + 0.30*df["Tracking Centre Forward"] + 0.10*df["Deep-Lying Forward"] + 0.10*df["Complete Forward"]
    )

    df["Al Gharafa ST Fit"] = clamp_series(
        0.25*df["Box Threat"] + 0.20*df["Link-up Play"] + 0.20*df["Channel Running"] + 0.15*df["Wide Combination"] + 0.10*df["Aerial Outlet"] + 0.10*df["Defensive Work"]
    )
    df["4-2-3-1 Fit"] = clamp_series(
        0.27*df["Box Threat"] + 0.24*df["Link-up Play"] + 0.19*df["Wide Combination"] + 0.14*df["Channel Running"] + 0.10*df["Defensive Work"] + 0.06*df["Aerial Outlet"]
    )
    df["3-4-3 Fit"] = clamp_series(
        0.24*df["Channel Running"] + 0.22*df["Wide Combination"] + 0.19*df["Box Threat"] + 0.14*df["Link-up Play"] + 0.12*df["Defensive Work"] + 0.09*df["Aerial Outlet"]
    )
    df["4-4-2 Fit"] = clamp_series(
        0.25*df["Box Threat"] + 0.20*df["Aerial Outlet"] + 0.18*df["Link-up Play"] + 0.15*df["Channel Running"] + 0.12*df["Defensive Work"] + 0.10*df["Wide Combination"]
    )
    df["Best System Fit"] = df[["4-2-3-1 Fit", "3-4-3 Fit", "4-4-2 Fit"]].max(axis=1)
    df["Best System"] = df[["4-2-3-1 Fit", "3-4-3 Fit", "4-4-2 Fit"]].idxmax(axis=1).str.replace(" Fit", "", regex=False)

    role_cols = role_columns(df)
    df["Generic Role Level"] = df[role_cols].mean(axis=1) if role_cols else df["Al Gharafa ST Fit"]
    age = df["Age"].fillna(26)
    df["Age Upside"] = np.select(
        [age <= 20, age <= 23, age <= 25, age <= 27, age <= 30],
        [100, 92, 80, 66, 52],
        default=35,
    )
    df["Specialist Fit Gap"] = (df["Al Gharafa ST Fit"] - df["Generic Role Level"]).clip(-30, 30)
    gap_score = (50 + df["Specialist Fit Gap"] * 2.2).clip(0, 100)
    inverse_generic = (100 - df["Generic Role Level"]).clip(0, 100)

    # Realistic scouting logic: famous / elite-club players are treated as benchmarks,
    # not hidden-gem recommendations. The default ranking rewards fit + under-the-radar value.
    df = obvious_player_flags(df)
    elite_penalty = np.select(
        [df["Famous Name Flag"], df["Elite Club Flag"], df["Generic Role Level"] >= 83],
        [42, 32, 18],
        default=0,
    )
    smaller_club_bonus = np.where(~df["Elite Club Flag"], 6, 0)
    realistic_age_bonus = np.where(df["Age"].between(20, 25, inclusive="both"), 5, 0)

    df["Under-the-Radar Score"] = clamp_series(
        0.38*inverse_generic + 0.26*gap_score + 0.20*df["Age Upside"] + 0.10*(100-df["Best System Fit"]).clip(0, 100)
        + smaller_club_bonus + realistic_age_bonus - elite_penalty
    )
    df["Hidden Gem Opportunity"] = clamp_series(
        0.48*df["Al Gharafa ST Fit"] + 0.24*df["Under-the-Radar Score"] + 0.16*df["Age Upside"] + 0.12*df["Best System Fit"] - 0.55*elite_penalty
    )
    df["Scouting Opportunity"] = clamp_series(
        0.42*df["Al Gharafa ST Fit"] + 0.25*df["Hidden Gem Opportunity"] + 0.18*df["Best System Fit"] + 0.15*df["Age Upside"] - 0.75*elite_penalty
    )
    df["Recruitment Priority"] = df["Scouting Opportunity"]

    # Best striker role and probabilities.
    role_cols = role_columns(df)
    df["Best Role"] = df[role_cols].idxmax(axis=1) if role_cols else "Striker"
    temperature = 8.0
    if role_cols:
        scores = df[role_cols].to_numpy(dtype=float)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores / temperature)
        probs = probs / probs.sum(axis=1, keepdims=True)
        for i, c in enumerate(role_cols):
            df[f"Prob: {c}"] = probs[:, i] * 100

    # Scout tag: keep famous players out of hidden-gem labels.
    not_obvious = ~df["Obvious Elite Player"]
    conditions = [
        (df["Obvious Elite Player"]),
        (not_obvious & (df["Scouting Opportunity"] >= 76) & (df["Al Gharafa ST Fit"] >= 58)),
        (not_obvious & (df["Hidden Gem Opportunity"] >= 68) & (df["Al Gharafa ST Fit"] >= 56)),
        (not_obvious & (df["Age"] <= 21) & (df["Al Gharafa ST Fit"] >= 54)),
        (not_obvious & (df["Best System Fit"] >= 72) & (df["Al Gharafa ST Fit"] >= 55)),
        (not_obvious & (df["Age"] <= 24) & (df["Hidden Gem Opportunity"] >= 58)),
        (df["Al Gharafa ST Fit"] >= 58),
    ]
    tags = ["Obvious Benchmark", "Priority Hidden Gem", "Hidden Gem", "U21 Development Target", "System Fit Opportunity", "Development Target", "Watchlist"]
    df["Scout Tag"] = np.select(conditions, tags, default="Low Fit")

    # Nationality / country support.
    # Important: EFEM enrichment stores reliable data mainly in Nationalities, while older patches may
    # leave Nationality_1 or Country as Unknown. Therefore Country is rebuilt row-by-row from the best source.
    for c in ["Country", "Nationality_1", "Nationality_2", "Nationalities", "Continent"]:
        if c not in df.columns:
            df[c] = ""

    continent_labels = {"Europe", "Africa", "Asia/Oceania", "Asia / Oceania", "Asia", "South America", "North America", "Oceania", "Other"}

    def first_valid_country_from_row(row: pd.Series) -> str:
        for col in ["Nationalities", "Nationality_1", "Country", "Nationality", "Nation", "Citizenship"]:
            if col in row.index:
                vals = normalise_country_value(row.get(col, ""))
                for val in vals:
                    if val and val not in continent_labels:
                        return val
        return "Unknown"

    df["Country"] = df.apply(first_valid_country_from_row, axis=1)

    # Manual correction for the current Al Gharafa ST benchmark group.
    current_manual = {
        "joselu": ("Spain", ""),
        "jose luis mato sanmartin": ("Spain", ""),
        "mohammed muntari": ("Qatar", "Ghana"),
        "isslam el awwad": ("Egypt", ""),
        "isslam hossam el awwad": ("Egypt", ""),
        "islam hossam elawwad": ("Egypt", ""),
        "zaid faisal ktit": ("Qatar", ""),
        "zaid faisal": ("Qatar", ""),
        "zaid ktit": ("Qatar", ""),
    }
    name_norm = df["Name"].apply(normalise_text)
    club_norm = df["Club"].apply(normalise_text)
    for n_key, (c1, c2) in current_manual.items():
        mask = (name_norm == n_key) & (club_norm.str.contains("gharafa", na=False) | name_norm.isin(current_manual.keys()))
        df.loc[mask, "Country"] = c1
        df.loc[mask, "Nationality_1"] = c1
        df.loc[mask, "Nationality_2"] = c2
        df.loc[mask, "Nationalities"] = (f"{c1}, {c2}" if c2 else c1)

    # Keep Nationality_1 aligned for display/filtering when it was previously blank or wrong.
    df["Nationality_1"] = df["Country"].where(df["Country"].astype(str).str.lower().ne("unknown"), df["Nationality_1"])
    df["Continent"] = df["Country"].apply(country_to_continent)
    df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    df["Continent"] = df["Continent"].fillna("Other").astype(str).str.strip().replace({"": "Other", "nan": "Other", "None": "Other"})

    return df


@st.cache_data(show_spinner=False)
def load_dna_tables():
    season = pd.read_csv(DNA_SEASON_PATH) if DNA_SEASON_PATH.exists() else pd.DataFrame()
    formation = pd.read_csv(FORMATION_SUMMARY_PATH) if FORMATION_SUMMARY_PATH.exists() else pd.DataFrame()
    formation_season = pd.read_csv(FORMATION_SEASON_PATH) if FORMATION_SEASON_PATH.exists() else pd.DataFrame()
    return season, formation, formation_season

players = load_players()
dna_by_season, formation_summary, formation_by_season = load_dna_tables()
PHASES = ["Box Threat", "Link-up Play", "Channel Running", "Wide Combination", "Aerial Outlet", "Defensive Work"]
SCORE_COLS = ["Scouting Opportunity", "Al Gharafa ST Fit", "Hidden Gem Opportunity", "Under-the-Radar Score", "Best System Fit"]
ROLE_COLS = role_columns(players)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=150)
    st.markdown("### Al Gharafa SC")
    st.caption("Striker recruitment intelligence")

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Al Gharafa DNA",
            "Current ST Assessment",
            "Hidden Gems Discovery",
            "Player Card",
            "Compare Players",
            "Similarity Scouting",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Filters")
    min_age = int(np.nanmin(players["Age"])) if players["Age"].notna().any() else 15
    max_age = int(np.nanmax(players["Age"])) if players["Age"].notna().any() else 35
    age_range = st.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, min(25, max_age)))

    hide_obvious = st.checkbox("Hide obvious elite / famous players", value=True)
    min_fit = st.slider("Minimum Al Gharafa ST Fit", 0, 100, 48, 1)
    min_hidden = st.slider("Minimum Hidden Gem Opportunity", 0, 100, 45, 1)

    tag_options = ["All"] + sorted([t for t in players["Scout Tag"].dropna().unique().tolist() if t != "Obvious Benchmark"])
    scout_tag = st.selectbox("Scout Tag", tag_options, index=0)

    role_options = ["All"] + ROLE_COLS
    role_filter = st.selectbox("Best Role", role_options, index=0)

    continent_filter = "All"
    country_filter = "All"

    if "Continent" in players.columns and players["Continent"].notna().any():
        continent_values = [x for x in players["Continent"].dropna().astype(str).str.strip().unique().tolist() if x and x.lower() not in {"unknown", "nan"}]
        continent_options = ["All"] + sorted(continent_values)
        continent_filter = st.selectbox("Continent", continent_options, index=0)

    if "Country" in players.columns and players["Country"].notna().any():
        country_pool = players.copy()
        if continent_filter != "All" and "Continent" in country_pool.columns:
            country_pool = country_pool[country_pool["Continent"].astype(str) == continent_filter]
        country_values = [x for x in country_pool["Country"].dropna().astype(str).str.strip().unique().tolist() if x and x.lower() not in {"unknown", "nan"}]
        country_options = ["All"] + sorted(country_values)
        country_filter = st.selectbox("Nationality", country_options, index=0)
    else:
        st.caption("Nationality filters appear automatically when the CSV includes Nationality_1 / Country columns.")

# Apply filters
df = players.copy()
if hide_obvious and "Obvious Elite Player" in df.columns:
    df = df[~df["Obvious Elite Player"]]
df = df[(df["Age"].fillna(0) >= age_range[0]) & (df["Age"].fillna(999) <= age_range[1])]
df = df[(df["Al Gharafa ST Fit"] >= min_fit) & (df["Hidden Gem Opportunity"] >= min_hidden)]
if scout_tag != "All":
    df = df[df["Scout Tag"] == scout_tag]
if role_filter != "All":
    df = df[df["Best Role"] == role_filter]
if "Continent" in df.columns and continent_filter != "All":
    df = df[df["Continent"].astype(str) == continent_filter]
if "Country" in df.columns and country_filter != "All":
    df = df[df["Country"].astype(str) == country_filter]

df = df.sort_values(["Scouting Opportunity", "Hidden Gem Opportunity", "Al Gharafa ST Fit"], ascending=False)

# -----------------------------
# Header
# -----------------------------
logo_col, title_col = st.columns([0.10, 0.90], vertical_alignment="center")
with logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=82)
with title_col:
    st.markdown('<div class="main-title">Al Gharafa SC Striker Scouting Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Hidden-gem discovery model aligned to Al Gharafa tactical DNA under Pedro Martins</div>', unsafe_allow_html=True)
st.markdown('<div class="gold-line"></div>', unsafe_allow_html=True)

# -----------------------------
# Plot functions
# -----------------------------
def radar_chart(row: pd.Series, title: str = "Player phase profile") -> go.Figure:
    values = [float(row[p]) for p in PHASES]
    cats = PHASES + [PHASES[0]]
    vals = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals,
        theta=cats,
        fill="toself",
        name=str(row.get("Name", "Player")),
        line=dict(color=GOLD, width=4),
        fillcolor="rgba(214,165,0,0.22)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#D7DEE9", tickfont=dict(color=MUTED)),
            angularaxis=dict(gridcolor="#D7DEE9", tickfont=dict(size=12, color=NAVY)),
        ),
        showlegend=False,
        title=dict(text=title, font=dict(size=20, color=NAVY)),
    )
    return fig_layout(fig, 470)


def multi_radar(rows: pd.DataFrame, title: str) -> go.Figure:
    colors = [GOLD, NAVY, BLUE, RED, "#7C3AED"]
    fig = go.Figure()
    cats = PHASES + [PHASES[0]]
    for idx, (_, row) in enumerate(rows.iterrows()):
        vals = [float(row[p]) for p in PHASES] + [float(row[PHASES[0]])]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats,
            fill="toself",
            name=str(row["Name"]),
            line=dict(color=colors[idx % len(colors)], width=3),
            fillcolor="rgba(214,165,0,0.07)" if idx == 0 else None,
            opacity=0.78,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#D7DEE9"),
            angularaxis=dict(gridcolor="#D7DEE9", tickfont=dict(color=NAVY)),
        ),
        title=dict(text=title, font=dict(size=20, color=NAVY)),
    )
    return fig_layout(fig, 520)


def phase_bar(row: pd.Series) -> go.Figure:
    vals = [float(row[p]) for p in PHASES]
    fig = go.Figure(go.Bar(
        x=vals,
        y=PHASES,
        orientation="h",
        marker=dict(color=[GOLD if v >= 75 else NAVY if v >= 62 else "#9CA3AF" for v in vals]),
        text=[f"{v:.1f}" for v in vals],
        textposition="outside",
    ))
    fig.update_xaxes(range=[0, 105], gridcolor="#E5E7EB")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Phase strengths", showlegend=False)
    return fig_layout(fig, 395)


def role_probability_chart(row: pd.Series) -> go.Figure:
    prob_cols = [c for c in row.index if c.startswith("Prob: ")]
    data = []
    for c in prob_cols:
        data.append((c.replace("Prob: ", ""), float(row[c])))
    top = pd.DataFrame(data, columns=["Role", "Probability"]).sort_values("Probability", ascending=True).tail(8)
    fig = go.Figure(go.Bar(
        x=top["Probability"], y=top["Role"], orientation="h",
        marker=dict(color=GOLD),
        text=[f"{v:.1f}%" for v in top["Probability"]], textposition="outside",
    ))
    fig.update_xaxes(range=[0, max(35, top["Probability"].max()+7)], gridcolor="#E5E7EB")
    fig.update_layout(title="Role probability distribution", showlegend=False)
    return fig_layout(fig, 390)


def hidden_gem_scatter(data: pd.DataFrame) -> go.Figure:
    sample = data.head(3000).copy()
    fig = px.scatter(
        sample,
        x="Generic Role Level",
        y="Al Gharafa ST Fit",
        size="Hidden Gem Opportunity",
        color="Scout Tag",
        hover_name="Name",
        hover_data={"Club": True, "Age": True, "Best Role": True, "Scouting Opportunity": ":.1f", "Under-the-Radar Score": ":.1f", "Generic Role Level": ":.1f", "Al Gharafa ST Fit": ":.1f"},
        color_discrete_sequence=[GOLD, NAVY, BLUE, "#8B5CF6", RED, "#6B7280"],
    )
    fig.add_shape(type="line", x0=0, x1=100, y0=0, y1=100, line=dict(color="#9CA3AF", dash="dash"))
    fig.add_annotation(x=45, y=82, text="High tactical fit / less generic profile", showarrow=False, font=dict(color=NAVY, size=12), bgcolor="rgba(214,165,0,0.18)")
    fig.update_xaxes(title="Generic striker profile level", range=[35, 100], gridcolor="#E5E7EB")
    fig.update_yaxes(title="Al Gharafa ST Fit", range=[35, 100], gridcolor="#E5E7EB")
    fig.update_layout(title="Hidden Gem Map: realistic fit versus generic profile", legend_title_text="")
    return fig_layout(fig, 540)


def difference_bars(rows: pd.DataFrame) -> go.Figure:
    if len(rows) < 2:
        return go.Figure()
    mean_vals = rows[PHASES].mean()
    plot_rows = []
    for _, r in rows.iterrows():
        for p in PHASES:
            plot_rows.append({"Player": r["Name"], "Phase": p, "Difference vs group": float(r[p] - mean_vals[p])})
    d = pd.DataFrame(plot_rows)
    fig = px.bar(
        d,
        x="Difference vs group",
        y="Phase",
        color="Player",
        barmode="group",
        orientation="h",
        color_discrete_sequence=[GOLD, NAVY, BLUE, RED, "#7C3AED"],
    )
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#9CA3AF")
    fig.update_layout(title="Where each player is stronger/weaker than the comparison group")
    fig.update_xaxes(gridcolor="#E5E7EB")
    return fig_layout(fig, 460)


def display_metric(label: str, value: object, suffix: str = ""):
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}{suffix}</div></div>""",
        unsafe_allow_html=True,
    )


def export_safe(data: pd.DataFrame) -> pd.DataFrame:
    banned = [c for c in data.columns if "url" in c.lower() or "link" in c.lower()]
    cols = [
        "Name", "Club", "Age", "Positions",
        "Scouting Opportunity", "Al Gharafa ST Fit", "Hidden Gem Opportunity", "Under-the-Radar Score", "Best System", "Best System Fit", "Best Role", "Scout Tag",
        "Box Threat", "Link-up Play", "Channel Running", "Wide Combination", "Aerial Outlet", "Defensive Work",
    ]
    if "Country" in data.columns:
        cols.insert(4, "Country")
        cols.insert(5, "Continent")
    keep = [c for c in cols if c in data.columns and c not in banned]
    return data[keep].round(2)


def system_fit_chart(row: pd.Series) -> go.Figure:
    system_df = pd.DataFrame({
        "System": ["4-2-3-1", "3-4-3", "4-4-2"],
        "Fit": [row["4-2-3-1 Fit"], row["3-4-3 Fit"], row["4-4-2 Fit"]]
    }).sort_values("Fit", ascending=True)
    fig = go.Figure(go.Bar(
        x=system_df["Fit"],
        y=system_df["System"],
        orientation="h",
        marker=dict(color=[GOLD, NAVY, BLUE]),
        text=[f"{v:.1f}" for v in system_df["Fit"]],
        textposition="outside",
    ))
    fig.update_xaxes(range=[0, 105], gridcolor="#E5E7EB")
    fig.update_layout(title="System fit", showlegend=False)
    return fig_layout(fig, 390)


def player_identity_line(row: pd.Series) -> str:
    country = row.get("Country", "") if "Country" in row.index else ""
    bits = [str(row.get("Club", "Unknown"))]
    if pd.notna(row.get("Age", np.nan)):
        bits.append(f"Age {int(row['Age'])}")
    if country and str(country).lower() not in {"unknown", "nan"}:
        bits.append(str(country))
    if row.get("Positions", ""):
        bits.append(str(row.get("Positions")))
    return " · ".join(bits)


def qualitative_summary(row: pd.Series) -> tuple[str, str, str]:
    strengths = sorted([(p, float(row[p])) for p in PHASES], key=lambda x: x[1], reverse=True)
    best_1, best_2 = strengths[0], strengths[1]
    weaker = strengths[-1]
    verdict = score_badge(float(row["Scouting Opportunity"]))

    summary = (
        f"{row['Name']} profiles as a {row['Best Role']} with the strongest value in {best_1[0]} "
        f"({best_1[1]:.1f}) and {best_2[0]} ({best_2[1]:.1f}). His best tactical context is "
        f"{row['Best System']} with a system-fit score of {row['Best System Fit']:.1f}."
    )
    role_fit = (
        "For Al Gharafa, the ideal striker is not a poacher-only profile. The recruitment model rewards a forward "
        "who can link the No.10 and wide players, attack the box, run channels in transition, and still give enough "
        "defensive work to support the mid-block."
    )
    risk = (
        f"Main follow-up area: {weaker[0]} ({weaker[1]:.1f}). This should be checked by video before any final recommendation. "
        f"Current recommendation level: {verdict}."
    )
    return summary, role_fit, risk


def logo_as_base64() -> str:
    if not LOGO_PATH.exists():
        return ""
    try:
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def build_scouting_report_html(row: pd.Series) -> str:
    summary, role_fit, risk = qualitative_summary(row)
    logo_src = logo_as_base64()
    image_html = ""
    if "image_url" in row.index and pd.notna(row.get("image_url")) and str(row.get("image_url")).strip():
        image_html = f'<img class="player-img" src="{html.escape(str(row.get("image_url")))}" />'
    elif logo_src:
        image_html = f'<img class="player-img" src="{logo_src}" />'

    radar_html = radar_chart(row, "Al Gharafa striker phase fit").to_html(full_html=False, include_plotlyjs="cdn")
    phase_html = phase_bar(row).to_html(full_html=False, include_plotlyjs=False)
    role_html = role_probability_chart(row).to_html(full_html=False, include_plotlyjs=False)
    system_html = system_fit_chart(row).to_html(full_html=False, include_plotlyjs=False)

    country_line = ""
    if "Country" in row.index and str(row.get("Country", "")).strip() and str(row.get("Country", "")).lower() != "unknown":
        country_line = f"<b>Nationality:</b> {html.escape(str(row.get('Country')))}"
        if "Continent" in row.index:
            country_line += f" · <b>Region:</b> {html.escape(str(row.get('Continent')))}"

    report = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Al Gharafa Scouting Report - {html.escape(str(row['Name']))}</title>
<style>
body {{font-family: Arial, sans-serif; margin: 0; background:#F6F8FB; color:#0B1F3A;}}
.wrapper {{max-width: 1180px; margin: 0 auto; padding: 34px;}}
.header {{display:flex; align-items:center; gap:18px; background:#0B1F3A; color:white; padding:22px 26px; border-radius:24px; border-bottom:6px solid #D6A500;}}
.logo {{width:76px; height:76px; object-fit:contain; background:white; border-radius:16px; padding:6px;}}
.title h1 {{margin:0; font-size:30px;}}
.title p {{margin:6px 0 0; color:#D8E2F2;}}
.card {{background:white; border:1px solid #E5E7EB; border-radius:22px; padding:22px; box-shadow:0 12px 30px rgba(11,31,58,.08); margin-top:18px;}}
.hero {{display:grid; grid-template-columns:160px 1fr 360px; gap:20px; align-items:center;}}
.player-img {{width:140px; height:140px; object-fit:contain; border-radius:20px; background:#EEF2F7;}}
.kpi-grid {{display:grid; grid-template-columns:repeat(4,1fr); gap:12px;}}
.kpi {{border-top:5px solid #D6A500; background:#FFFFFF; padding:14px; border-radius:18px; box-shadow:0 8px 18px rgba(11,31,58,.08);}}
.kpi small {{display:block; color:#6B7280; text-transform:uppercase; font-weight:800; letter-spacing:.06em;}}
.kpi b {{font-size:25px; color:#0B1F3A;}}
.badge {{display:inline-block; border:1px solid rgba(214,165,0,.55); background:rgba(214,165,0,.15); padding:6px 10px; border-radius:999px; font-size:12px; font-weight:800; margin:4px 4px 0 0;}}
.section-title {{font-size:21px; font-weight:900; margin:0 0 10px;}}
.grid-2 {{display:grid; grid-template-columns:1fr 1fr; gap:18px;}}
.text-block p {{line-height:1.62; font-size:15px;}}
.table {{width:100%; border-collapse:collapse; margin-top:8px;}}
.table th, .table td {{border-bottom:1px solid #E5E7EB; padding:10px; text-align:left; font-size:14px;}}
.table th {{background:#EEF2F7;}}
.footer {{color:#6B7280; font-size:12px; margin-top:20px;}}
@media print {{body {{background:white;}} .card {{box-shadow:none;}}}}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    {f'<img class="logo" src="{logo_src}" />' if logo_src else ''}
    <div class="title">
      <h1>Al Gharafa SC — Striker Scouting Report</h1>
      <p>Hidden-gem recruitment report aligned to the club's tactical DNA</p>
    </div>
  </div>

  <div class="card hero">
    <div>{image_html}</div>
    <div>
      <h2 style="margin:0 0 8px;font-size:30px;">{html.escape(str(row['Name']))}</h2>
      <p style="margin:0 0 10px;color:#6B7280;">{html.escape(player_identity_line(row))}</p>
      <div>
        <span class="badge">{html.escape(str(row['Scout Tag']))}</span>
        <span class="badge">{html.escape(str(row['Best Role']))}</span>
        <span class="badge">Best system: {html.escape(str(row['Best System']))}</span>
      </div>
      <p style="margin-top:12px;">{country_line}</p>
    </div>
    <div class="kpi-grid" style="grid-template-columns:1fr 1fr;">
      <div class="kpi"><small>ST Fit</small><b>{float(row['Al Gharafa ST Fit']):.1f}</b></div>
      <div class="kpi"><small>Hidden Gem</small><b>{float(row['Hidden Gem Opportunity']):.1f}</b></div>
      <div class="kpi"><small>Opportunity</small><b>{float(row['Scouting Opportunity']):.1f}</b></div>
      <div class="kpi"><small>System</small><b>{float(row['Best System Fit']):.1f}</b></div>
    </div>
  </div>

  <div class="card text-block">
    <h3 class="section-title">Qualitative scouting summary</h3>
    <p>{html.escape(summary)}</p>
    <p>{html.escape(role_fit)}</p>
    <p>{html.escape(risk)}</p>
  </div>

  <div class="card text-block">
    <h3 class="section-title">Al Gharafa DNA context</h3>
    <p>Under Pedro Martins, Al Gharafa are best interpreted as a 4-2-3-1-led hybrid side: controlled build-up, structured progression, wide creation and situational pressing. The striker model is therefore built around six needs: box threat, link-up play, channel running, wide combination, aerial outlet and defensive work.</p>
    <p>The report does not rank players by name value. It tests whether the player gives the specific forward behaviours that Al Gharafa require, while the tool's hidden-gem logic reduces the priority of obvious superstar or elite-market names.</p>
  </div>

  <div class="card"><h3 class="section-title">Player phase radar</h3>{radar_html}</div>
  <div class="grid-2">
    <div class="card"><h3 class="section-title">Phase strengths</h3>{phase_html}</div>
    <div class="card"><h3 class="section-title">System fit</h3>{system_html}</div>
  </div>
  <div class="card"><h3 class="section-title">Role probability profile</h3>{role_html}</div>

  <div class="card text-block">
    <h3 class="section-title">Recommended next scouting checks</h3>
    <table class="table">
      <tr><th>Area</th><th>Video / live-scouting question</th></tr>
      <tr><td>Box threat</td><td>Does the player arrive in the box at the right timing, or does he only occupy static central positions?</td></tr>
      <tr><td>Link-up</td><td>Can he receive under pressure and connect first or second touch with the No.10 and wide players?</td></tr>
      <tr><td>Channel threat</td><td>Does he attack space behind the centre-back/full-back channel during transitions?</td></tr>
      <tr><td>Wide combination</td><td>Can he combine with wingers and attack cutbacks/crosses instead of drifting away from goal?</td></tr>
      <tr><td>Defensive work</td><td>Can he support the first line of pressure and mid-block without losing attacking sharpness?</td></tr>
    </table>
  </div>

  <div class="footer">Confidential scouting prototype — Al Gharafa recruitment use only. This report is a first-layer recruitment filter and must be supported by video scouting and contextual review.</div>
</div>
</body>
</html>
"""
    return report


def safe_report_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")
    return f"al_gharafa_scouting_report_{clean or 'player'}.html"



def report_table(data: pd.DataFrame, cols: list[str], limit: int = 50) -> pd.DataFrame:
    available = [c for c in cols if c in data.columns]
    if not available:
        return pd.DataFrame()
    out = data[available].head(limit).copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(2)
    return out


def build_page_report_html(
    title: str,
    subtitle: str,
    narrative_blocks: list[str],
    figures: list[tuple[str, go.Figure]] | None = None,
    tables: list[tuple[str, pd.DataFrame]] | None = None,
    footer_note: str = "This report is a club-facing scouting support output and should be combined with video review.",
) -> str:
    logo_src = logo_as_base64()
    fig_sections = []
    for i, (fig_title, fig) in enumerate(figures or []):
        if fig is None:
            continue
        try:
            fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
            fig_sections.append(f'<div class="card"><h3 class="section-title">{html.escape(fig_title)}</h3>{fig_html}</div>')
        except Exception:
            continue

    table_sections = []
    for table_title, table_df in tables or []:
        if table_df is None or table_df.empty:
            continue
        table_sections.append(
            f'<div class="card"><h3 class="section-title">{html.escape(table_title)}</h3>'
            + table_df.to_html(index=False, escape=True, classes="table")
            + '</div>'
        )

    paragraphs = "".join(f"<p>{html.escape(str(p))}</p>" for p in narrative_blocks if str(p).strip())
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{font-family: Arial, sans-serif; margin:0; background:#F6F8FB; color:#0B1F3A;}}
.wrapper {{max-width:1180px; margin:0 auto; padding:34px;}}
.header {{display:flex; align-items:center; gap:18px; background:#0B1F3A; color:white; padding:22px 26px; border-radius:24px; border-bottom:6px solid #D6A500;}}
.logo {{width:76px; height:76px; object-fit:contain; background:white; border-radius:16px; padding:6px;}}
.title h1 {{margin:0; font-size:30px;}}
.title p {{margin:6px 0 0; color:#D8E2F2;}}
.card {{background:white; border:1px solid #E5E7EB; border-radius:22px; padding:22px; box-shadow:0 12px 30px rgba(11,31,58,.08); margin-top:18px;}}
.section-title {{font-size:21px; font-weight:900; margin:0 0 10px; color:#0B1F3A;}}
.text-block p {{line-height:1.62; font-size:15px;}}
.table {{width:100%; border-collapse:collapse; margin-top:8px; font-size:13px;}}
.table th, .table td {{border-bottom:1px solid #E5E7EB; padding:9px; text-align:left;}}
.table th {{background:#EEF2F7; color:#0B1F3A;}}
.footer {{color:#6B7280; font-size:12px; margin-top:20px;}}
@media print {{body {{background:white;}} .card {{box-shadow:none;}}}}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    {f'<img class="logo" src="{logo_src}" />' if logo_src else ''}
    <div class="title"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>
  </div>
  <div class="card text-block"><h3 class="section-title">Qualitative context</h3>{paragraphs}</div>
  {''.join(fig_sections)}
  {''.join(table_sections)}
  <div class="footer">{html.escape(footer_note)}</div>
</div>
</body>
</html>
"""


def page_report_download(
    label: str,
    title: str,
    subtitle: str,
    narrative_blocks: list[str],
    figures: list[tuple[str, go.Figure]] | None = None,
    tables: list[tuple[str, pd.DataFrame]] | None = None,
    filename: str = "al_gharafa_page_report.html",
    key: str = "page_report_download",
):
    html_report = build_page_report_html(title, subtitle, narrative_blocks, figures=figures, tables=tables)
    st.download_button(
        label,
        data=html_report.encode("utf-8"),
        file_name=filename,
        mime="text/html",
        key=key,
        help="Download a browser-readable HTML report. Open it like a normal local web page / URL.",
    )


def current_squad_strikers(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "Club" not in data.columns:
        return pd.DataFrame()
    mask = data["Club"].astype(str).str.contains("gharafa", case=False, na=False)
    current = data[mask].copy()
    if current.empty and "Name" in data.columns:
        current_names = {"joselu", "mohammed muntari", "isslam el awwad", "zaid faisal ktit", "zaid faisal", "zaid ktit"}
        current = data[data["Name"].apply(normalise_text).isin(current_names)].copy()
    if current.empty:
        return current
    return current.sort_values(["Al Gharafa ST Fit", "Scouting Opportunity"], ascending=False)


def similarity_feature_columns(data: pd.DataFrame) -> list[str]:
    candidate_cols = PHASES + [
        "Al Gharafa ST Fit", "Best System Fit", "4-2-3-1 Fit", "3-4-3 Fit", "4-4-2 Fit",
        "Advanced Forward", "Deep-Lying Forward", "Complete Forward", "Channel Forward", "Centre Forward",
        "Target Forward", "Pressing Forward", "False Nine", "Second Striker", "Poacher",
    ]
    return [c for c in candidate_cols if c in data.columns]


def similar_players(benchmark_row: pd.Series, candidate_pool: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if candidate_pool.empty:
        return pd.DataFrame()
    pool = candidate_pool.copy()
    if "Name" in pool.columns:
        pool = pool[pool["Name"].astype(str) != str(benchmark_row.get("Name", ""))]
    if "player_url" in pool.columns and "player_url" in benchmark_row.index:
        b_url = str(benchmark_row.get("player_url", "")).strip()
        if b_url:
            pool = pool[pool["player_url"].astype(str).str.strip() != b_url]
    cols = similarity_feature_columns(pool)
    if not cols or pool.empty:
        return pd.DataFrame()
    bench = pd.to_numeric(pd.Series({c: benchmark_row.get(c, 0) for c in cols}), errors="coerce").fillna(0).astype(float).to_numpy()
    mat = pool[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).to_numpy()
    bench_norm = float(np.linalg.norm(bench))
    row_norms = np.linalg.norm(mat, axis=1)
    if bench_norm == 0:
        sims = np.zeros(len(pool))
    else:
        sims = (mat @ bench) / np.where(row_norms * bench_norm == 0, np.nan, row_norms * bench_norm)
        sims = np.nan_to_num(sims, nan=0.0)
    pool["Similarity Score"] = np.clip(sims * 100, 0, 100)
    if "Al Gharafa ST Fit" in pool.columns:
        pool["ST Fit Gap vs Benchmark"] = (pool["Al Gharafa ST Fit"] - float(benchmark_row.get("Al Gharafa ST Fit", 0))).abs()
    else:
        pool["ST Fit Gap vs Benchmark"] = np.nan
    return pool.sort_values(["Similarity Score", "Scouting Opportunity"], ascending=False).head(top_n)

# -----------------------------
# Pages
# -----------------------------
if page == "Executive Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1: display_metric("Players screened", f"{len(players):,}")
    with c2: display_metric("Current filtered pool", f"{len(df):,}")
    with c3:
        top_fit = df["Al Gharafa ST Fit"].max() if not df.empty else 0
        display_metric("Best ST Fit", f"{top_fit:.1f}")
    with c4:
        top_hidden = df["Hidden Gem Opportunity"].max() if not df.empty else 0
        display_metric("Best Hidden Gem", f"{top_hidden:.1f}")

    st.markdown('<div class="section-header">Recruitment logic</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="navy-card">
        <b>Tool objective:</b> identify under-the-radar strikers who match Al Gharafa's tactical striker profile, not simply the highest generic scorer.<br><br>
        <b>Ideal target:</b> a striker who can act as a Complete Forward / Deep-Lying Forward hybrid: attack the box, link with the No.10 and wide players, run channels, and still support the team's mid-block defensive work.<br><br>
        <b>Main decision score:</b> Scouting Opportunity = Al Gharafa fit + hidden-gem value + system fit + age upside, with obvious superstar/elite-club players suppressed by default.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.18, 1])
    with left:
        st.plotly_chart(hidden_gem_scatter(df if not df.empty else players), width="stretch")
    with right:
        st.markdown('<div class="section-header">Realistic hidden-gem recommendations</div>', unsafe_allow_html=True)
        top = df.head(8)
        for _, r in top.iterrows():
            cols = st.columns([0.18, 0.82], vertical_alignment="center")
            with cols[0]:
                if "image_url" in r.index:
                    img_url = "" if pd.isna(r.get("image_url", "")) else str(r.get("image_url", "")).strip()
                    if img_url and img_url.lower() not in ["nan", "none", "unknown", "-", ""]:
                        try:
                            st.image(img_url, width=135)
                        except Exception:
                            pass
            with cols[1]:
                st.markdown(
                    f"""<div class="card" style="padding:12px 14px;margin-bottom:8px;">
                    <b style="color:{NAVY};font-size:16px;">{r['Name']}</b><br>
                    <span class="small-note">{r['Club']} · Age {int(r['Age']) if pd.notna(r['Age']) else '-'} · {r['Best Role']}</span><br>
                    <span class="tag">ST Fit {r['Al Gharafa ST Fit']:.1f}</span>
                    <span class="tag">Hidden {r['Hidden Gem Opportunity']:.1f}</span>
                    <span class="tag">{r['Scout Tag']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )


    overview_pool = df if not df.empty else players.sort_values("Scouting Opportunity", ascending=False)
    overview_cols = ["Name","Club","Age","Country","Continent","Best Role","Scout Tag","Scouting Opportunity","Al Gharafa ST Fit","Hidden Gem Opportunity","Best System"]
    page_report_download(
        "Download overview report HTML",
        "Al Gharafa SC — Executive Scouting Overview",
        "Recruitment overview aligned to the club striker DNA",
        [
            "This overview summarises the filtered striker market through Al Gharafa's tactical DNA rather than simple generic striker ratings.",
            "The main logic prioritises players who combine box threat, link-up play, channel running and wide combinations, while reducing obvious elite-market targets by default.",
            "Use this page as the first meeting layer: it explains the model, the filtered pool and the strongest realistic hidden-gem recommendations before moving into individual player reports.",
        ],
        figures=[("Hidden-gem map", hidden_gem_scatter(overview_pool))],
        tables=[("Top recommended players", report_table(overview_pool, overview_cols, 15))],
        filename="al_gharafa_executive_overview_report.html",
        key="overview_report_download",
    )


elif page == "Current ST Assessment":
    st.markdown('<div class="section-header">Current ST Assessment</div>', unsafe_allow_html=True)
    st.caption("Assessment of current Al Gharafa striker options as internal benchmarks, not hidden-gem recommendations.")

    current = current_squad_strikers(players)
    if current.empty:
        st.warning("No current Al Gharafa strikers were detected in the CSV. Run the Al Gharafa ST scraper first, then refresh the app.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: display_metric("Current STs", f"{len(current)}")
        with c2: display_metric("Avg age", f"{current['Age'].mean():.1f}" if current['Age'].notna().any() else "-")
        with c3: display_metric("Best ST Fit", f"{current['Al Gharafa ST Fit'].max():.1f}")
        with c4: display_metric("Best System Fit", f"{current['Best System Fit'].max():.1f}")

        st.markdown(
            f"""
            <div class="navy-card">
            <b>Interpretation:</b> this page turns the current Al Gharafa striker group into a benchmark layer. The idea is to understand the present striker room first — who gives box threat, who connects play, who works as an aerial outlet, and who is closest to the ideal 4-2-3-1-led striker profile. The hidden-gem page should then be read against this current internal benchmark.
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_cols = ["Name","Club","Age","Country","Continent","Best Role","Best System","Al Gharafa ST Fit","Hidden Gem Opportunity","Scouting Opportunity","Box Threat","Link-up Play","Channel Running","Wide Combination","Aerial Outlet","Defensive Work"]
        st.dataframe(report_table(current, current_cols, 20), width="stretch", hide_index=True)

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(multi_radar(current.head(5), "Current Al Gharafa ST phase profiles"), width="stretch")
        with right:
            st.plotly_chart(difference_bars(current.head(5)), width="stretch")

        selected_current = st.selectbox("Inspect current striker", current["Name"].astype(str).tolist(), index=0, key="current_st_select")
        cur_row = current[current["Name"].astype(str) == selected_current].iloc[0]
        s1, s2 = st.columns([0.38, 0.62], vertical_alignment="center")
        with s1:
            if "image_url" in cur_row.index:
                img_url = "" if pd.isna(cur_row.get("image_url", "")) else str(cur_row.get("image_url", "")).strip()
                if img_url and img_url.lower() not in ["nan", "none", "unknown", "-", ""]:
                    try:
                        st.image(img_url, width=135)
                    except Exception:
                        if LOGO_PATH.exists():
                            st.image(str(LOGO_PATH), width=140)
                elif LOGO_PATH.exists():
                    st.image(str(LOGO_PATH), width=140)
            elif LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=140)
            st.markdown(f"### {cur_row['Name']}")
            st.markdown(f"**{player_identity_line(cur_row)}**")
            st.markdown(f"<span class='tag'>{cur_row['Best Role']}</span><span class='tag'>{cur_row['Best System']}</span>", unsafe_allow_html=True)
        with s2:
            st.plotly_chart(radar_chart(cur_row, "Selected current ST radar"), width="stretch")

        summary, role_fit, risk = qualitative_summary(cur_row)
        st.markdown(
            f"""
            <div class="navy-card">
            <b>Selected striker reading:</b> {summary}<br><br>
            <b>Benchmark meaning:</b> this player can be used as a reference point when searching similar or upgraded profiles in the Similarity Scouting page.<br><br>
            {risk}
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_report_download(
            "Download current ST assessment report HTML",
            "Al Gharafa SC — Current ST Assessment",
            "Internal striker benchmark before external recruitment",
            [
                "This report assesses the current Al Gharafa striker group as an internal benchmark layer before moving to hidden-gem recruitment.",
                "The purpose is not to label current players as targets. It is to understand the reference profile already inside the squad: box threat, link-up play, channel running, wide combination, aerial outlet and defensive work.",
                "The external shortlist should be judged against this benchmark: does a candidate replace, complement or upgrade the current striker profile?",
            ],
            figures=[("Current ST radar comparison", multi_radar(current.head(5), "Current ST radar comparison")), ("Current ST phase differences", difference_bars(current.head(5)))],
            tables=[("Current striker assessment table", report_table(current, current_cols, 20))],
            filename="al_gharafa_current_st_assessment_report.html",
            key="current_st_report_download",
        )

elif page == "Hidden Gems Discovery":
    st.markdown('<div class="section-header">Hidden Gems Discovery</div>', unsafe_allow_html=True)
    st.caption("Sorted by Scouting Opportunity. Obvious superstar and elite-club players are hidden by default.")

    st.plotly_chart(hidden_gem_scatter(df if not df.empty else players), width="stretch")

    if not df.empty:
        inspect_pool = df.head(250).copy()
        selected_hidden = st.selectbox(
            "Inspect hidden-gem candidate",
            inspect_pool["Name"].astype(str).tolist(),
            index=0,
            key="hidden_gem_inspector",
        )
        selected_row = inspect_pool[inspect_pool["Name"].astype(str) == selected_hidden].iloc[0]
        c_img, c_profile, c_radar = st.columns([0.16, 0.34, 0.50], vertical_alignment="center")
        with c_img:
            if "image_url" in selected_row.index:
                img_url = "" if pd.isna(selected_row.get("image_url", "")) else str(selected_row.get("image_url", "")).strip()
                if img_url and img_url.lower() not in ["nan", "none", "unknown", "-", ""]:
                    try:
                        st.image(img_url, width=135)
                    except Exception:
                        if LOGO_PATH.exists():
                            st.image(str(LOGO_PATH), width=120)
                elif LOGO_PATH.exists():
                    st.image(str(LOGO_PATH), width=120)
            elif LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=120)
        with c_profile:
            st.markdown(f"### {selected_row['Name']}")
            st.markdown(f"**Club:** {selected_row['Club']}  ")
            st.markdown(f"**Age:** {int(selected_row['Age']) if pd.notna(selected_row['Age']) else '-'}  ")
            st.markdown(f"<span class='tag'>{selected_row['Scout Tag']}</span><span class='tag'>Opportunity {selected_row['Scouting Opportunity']:.1f}</span><span class='tag'>ST Fit {selected_row['Al Gharafa ST Fit']:.1f}</span>", unsafe_allow_html=True)
        with c_radar:
            st.plotly_chart(radar_chart(selected_row, "Selected hidden-gem phase profile"), width="stretch")

    view = export_safe(df).head(500)
    st.dataframe(view, width="stretch", height=560, hide_index=True)
    hidden_report_pool = df if not df.empty else players.sort_values("Scouting Opportunity", ascending=False)
    hidden_cols = ["Name","Club","Age","Country","Continent","Best Role","Scout Tag","Scouting Opportunity","Al Gharafa ST Fit","Hidden Gem Opportunity","Under-the-Radar Score","Best System"]
    page_report_download(
        "Download hidden gems report HTML",
        "Al Gharafa SC — Hidden Gems Shortlist",
        "Filtered shortlist report with qualitative scouting context",
        [
            "This shortlist is ranked by Scouting Opportunity, which combines Al Gharafa tactical fit, hidden-gem value, system fit and age upside.",
            "The page is designed as a first scouting filter: it does not make a final transfer decision, but highlights which players deserve video review and contextual validation.",
            "Obvious superstar or elite-club names are suppressed by default so the list stays useful for realistic recruitment discovery.",
        ],
        figures=[("Hidden-gem opportunity map", hidden_gem_scatter(hidden_report_pool))],
        tables=[("Top hidden-gem candidates", report_table(hidden_report_pool, hidden_cols, 50))],
        filename="al_gharafa_hidden_gems_report.html",
        key="hidden_gems_report_download",
    )
    st.caption("Raw CSV shortlist download is disabled. Use the HTML report button above for a club-facing report.")

elif page == "Player Card":
    st.markdown('<div class="section-header">Player Card</div>', unsafe_allow_html=True)
    pool = df if not df.empty else players.sort_values("Scouting Opportunity", ascending=False)
    names = pool["Name"].astype(str).tolist()
    selected = st.selectbox("Select player", names, index=0 if names else None)
    row = pool[pool["Name"].astype(str) == selected].iloc[0]

    top_cols = st.columns([0.18, 0.42, 0.40], vertical_alignment="center")
    with top_cols[0]:
        if "image_url" in row and pd.notna(row.get("image_url")):
            img_url = "" if pd.isna(row.get("image_url", "")) else str(row.get("image_url", "")).strip()
            if img_url and img_url.lower() not in ["nan", "none", "unknown", "-", ""]:
                try:
                    st.image(img_url, width=135)
                except Exception:
                    pass
        elif LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=135)
    with top_cols[1]:
        st.markdown(f"### {row['Name']}")
        st.markdown(f"**Club:** {row['Club']}  ")
        st.markdown(f"**Age:** {int(row['Age']) if pd.notna(row['Age']) else '-'}  ")
        st.markdown(f"**Positions:** {row['Positions']}  ")
        if "Country" in row.index:
            st.markdown(f"**Country:** {row['Country']}  ")
        st.markdown(f"<span class='tag'>{row['Scout Tag']}</span><span class='tag'>{row['Best Role']}</span><span class='tag'>{row['Best System']}</span>", unsafe_allow_html=True)
    with top_cols[2]:
        m1, m2 = st.columns(2)
        with m1: display_metric("ST Fit", f"{row['Al Gharafa ST Fit']:.1f}")
        with m2: display_metric("Hidden Gem", f"{row['Hidden Gem Opportunity']:.1f}")
        m3, m4 = st.columns(2)
        with m3: display_metric("Opportunity", f"{row['Scouting Opportunity']:.1f}")
        with m4: display_metric("System", f"{row['Best System Fit']:.1f}")

    st.markdown("---")
    r1, r2 = st.columns([1, 1])
    with r1:
        st.plotly_chart(radar_chart(row, "Al Gharafa striker phase fit"), width="stretch")
    with r2:
        st.plotly_chart(phase_bar(row), width="stretch")

    r3, r4 = st.columns([1, 1])
    with r3:
        st.plotly_chart(role_probability_chart(row), width="stretch")
    with r4:
        system_df = pd.DataFrame({
            "System": ["4-2-3-1", "3-4-3", "4-4-2"],
            "Fit": [row["4-2-3-1 Fit"], row["3-4-3 Fit"], row["4-4-2 Fit"]]
        }).sort_values("Fit", ascending=True)
        fig = go.Figure(go.Bar(x=system_df["Fit"], y=system_df["System"], orientation="h", marker=dict(color=[GOLD, NAVY, BLUE]), text=[f"{v:.1f}" for v in system_df["Fit"]], textposition="outside"))
        fig.update_xaxes(range=[0, 105], gridcolor="#E5E7EB")
        fig.update_layout(title="System fit")
        st.plotly_chart(fig_layout(fig, 390), width="stretch")

    strengths = sorted([(p, float(row[p])) for p in PHASES], key=lambda x: x[1], reverse=True)
    st.markdown(
        f"""
        <div class="navy-card">
        <b>Scout interpretation:</b> {row['Name']} profiles strongest in <b>{strengths[0][0]}</b> ({strengths[0][1]:.1f}) and <b>{strengths[1][0]}</b> ({strengths[1][1]:.1f}).
        His best system fit is <b>{row['Best System']}</b>. Recommendation level: <b>{score_badge(row['Scouting Opportunity'])}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download player scouting report HTML",
        data=build_scouting_report_html(row).encode("utf-8"),
        file_name=safe_report_filename(str(row["Name"])),
        mime="text/html",
        key="player_card_report_download",
        help="Download a browser-readable scouting report for the selected player.",
    )

elif page == "Compare Players":
    st.markdown('<div class="section-header">Compare Players</div>', unsafe_allow_html=True)
    pool = df if not df.empty else players.sort_values("Scouting Opportunity", ascending=False)
    default_names = pool.head(3)["Name"].tolist()
    selected_names = st.multiselect("Choose 2 to 5 players", pool["Name"].astype(str).tolist(), default=default_names, max_selections=5)
    if len(selected_names) < 2:
        st.warning("Choose at least two players to compare.")
    else:
        comp = pool[pool["Name"].isin(selected_names)].drop_duplicates("Name").head(5)
        st.plotly_chart(multi_radar(comp, "Player comparison radar"), width="stretch")
        st.plotly_chart(difference_bars(comp), width="stretch")
        comp_cols = ["Name","Club","Age","Best Role","Scout Tag","Scouting Opportunity","Al Gharafa ST Fit","Hidden Gem Opportunity","Under-the-Radar Score","Best System","Box Threat","Link-up Play","Channel Running","Wide Combination","Aerial Outlet","Defensive Work"]
        st.dataframe(comp[[c for c in comp_cols if c in comp.columns]].round(2), width="stretch", hide_index=True)

        page_report_download(
            "Download comparison report HTML",
            "Al Gharafa SC — Player Comparison Report",
            "Side-by-side striker comparison aligned to Al Gharafa DNA",
            [
                "This report compares selected strikers across the six Al Gharafa striker phases and the main recruitment scores.",
                "The aim is to show profile differences clearly: who offers more box threat, who links play better, who attacks channels, and who fits the tactical system more naturally.",
                "This comparison should be used to support discussion, not as a final transfer decision without video and contextual scouting.",
            ],
            figures=[("Comparison radar", multi_radar(comp, "Player comparison radar")), ("Phase differences", difference_bars(comp))],
            tables=[("Comparison table", report_table(comp, comp_cols, 10))],
            filename="al_gharafa_player_comparison_report.html",
            key="compare_report_download",
        )

elif page == "Al Gharafa DNA":
    st.markdown('<div class="section-header">Al Gharafa DNA Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="navy-card">
        <b>Club identity under Pedro Martins:</b> Al Gharafa are best described as a <b>4-2-3-1 hybrid side</b> with controlled build-up, structured progression, wide individual creation, and situational pressing.
        The team is not a pure high-pressing side and not a pure direct/counter side. The striker therefore must be more than a penalty-box finisher: he has to connect the front four, attack the box, offer channel threat, and support the wide creation structure.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Striker recruitment profile derived from the club DNA")
    req = pd.DataFrame({
        "Striker Requirement": PHASES,
        "Why it matters for Al Gharafa": [
            "The team needs reliable penalty-area presence and shot volume from the central striker.",
            "The 4-2-3-1 needs the No.9 to connect the AM and wide players, not isolate from them.",
            "The team needs acceleration after regains and early forward passes into space.",
            "Wide creation is a key attacking route, so the No.9 must combine with wingers and attack cutbacks/crosses.",
            "Aerial value is useful for crosses and set-pieces, but should not be the only recruitment logic.",
            "The striker must support the mid-block and first defensive line without being a specialist presser only.",
        ],
        "Model Weight": [25, 20, 20, 15, 10, 10],
    })
    fig = px.bar(req, x="Model Weight", y="Striker Requirement", orientation="h", text="Model Weight", color="Model Weight", color_continuous_scale=[[0, LIGHT_NAVY], [1, GOLD]])
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 30], gridcolor="#E5E7EB")
    fig.update_layout(title="Al Gharafa striker model weights", coloraxis_showscale=False)
    st.plotly_chart(fig_layout(fig, 430), width="stretch")
    st.dataframe(req, width="stretch", hide_index=True)

    if not dna_by_season.empty:
        st.markdown("### Team DNA evolution")
        long = dna_by_season.melt(id_vars=["Coach Season"], var_name="Phase", value_name="Score")
        long["Phase"] = long["Phase"].str.replace("DNA ", "", regex=False)
        fig = px.line(long, x="Coach Season", y="Score", color="Phase", markers=True, color_discrete_sequence=[NAVY, GOLD, BLUE, RED, "#6B7280", "#7C3AED", "#A16207", "#334155", "#93C5FD"])
        fig.update_yaxes(range=[0, 100], gridcolor="#E5E7EB")
        fig.update_layout(title="Al Gharafa playing-phase DNA by season", legend_title_text="")
        st.plotly_chart(fig_layout(fig, 520), width="stretch")

    if not formation_summary.empty:
        st.markdown("### Schemas used and their performance impact")
        show_cols = ["Formation","Matches","UsageShare%","PPG","GF","GA","xG","Possession","PPDA","BuildUp","Progression","FinalThird","BallCarrying","Wide","Transition","Pressing","Defensive","SetPieces"]
        available = [c for c in show_cols if c in formation_summary.columns]
        fs = formation_summary[available].copy().sort_values("UsageShare%", ascending=False)
        st.dataframe(fs.round(2), width="stretch", hide_index=True)

        c1, c2 = st.columns([1,1])
        with c1:
            fig = px.scatter(
                fs[fs["Matches"] >= 2] if "Matches" in fs.columns else fs,
                x="UsageShare%", y="PPG", size="xG" if "xG" in fs.columns else None,
                color="Formation", text="Formation",
                color_discrete_sequence=[GOLD, NAVY, BLUE, RED, "#6B7280", "#7C3AED"],
            )
            fig.update_traces(textposition="top center")
            fig.update_xaxes(title="Usage share %", gridcolor="#E5E7EB")
            fig.update_yaxes(title="Points per game", gridcolor="#E5E7EB")
            fig.update_layout(title="Schema impact: usage vs PPG", showlegend=False)
            st.plotly_chart(fig_layout(fig, 470), width="stretch")
        with c2:
            phase_cols = ["BuildUp","Progression","FinalThird","BallCarrying","Wide","Transition","Pressing","Defensive","SetPieces"]
            if all(c in formation_summary.columns for c in phase_cols):
                top_forms = formation_summary.sort_values("UsageShare%", ascending=False).head(8)
                heat = top_forms.set_index("Formation")[phase_cols]
                fig = go.Figure(data=go.Heatmap(
                    z=heat.values,
                    x=["Build-up","Progression","Final Third","1v1/Carrying","Wide","Transition","Pressing","Defensive","Set Pieces"],
                    y=heat.index,
                    colorscale=[[0, "#E8EEF8"], [0.5, "#8FAAD6"], [1, GOLD]],
                    zmin=0, zmax=100,
                    text=np.round(heat.values, 0),
                    texttemplate="%{text:.0f}",
                    textfont={"size": 11, "color": NAVY},
                ))
                fig.update_layout(title="Schema impact on playing phases")
                st.plotly_chart(fig_layout(fig, 470), width="stretch")

    if not formation_by_season.empty:
        st.markdown("### Formation usage by season")
        fig = px.bar(
            formation_by_season,
            x="Coach Season", y="UsageShare%", color="Formation",
            color_discrete_sequence=[NAVY, GOLD, BLUE, RED, "#6B7280", "#7C3AED", "#A16207", "#93C5FD"],
        )
        fig.update_layout(barmode="stack", title="Weighted schema usage across seasons", legend_title_text="")
        fig.update_yaxes(title="Usage share %", range=[0,100], gridcolor="#E5E7EB")
        st.plotly_chart(fig_layout(fig, 520), width="stretch")

    st.markdown("### Methodology formula")
    st.code(
        """Al Gharafa ST Fit =
0.25 Box Threat
+ 0.20 Link-up Play
+ 0.20 Channel Running
+ 0.15 Wide Combination
+ 0.10 Aerial Outlet
+ 0.10 Defensive Work

Scouting Opportunity =
0.42 Al Gharafa ST Fit
+ 0.25 Hidden Gem Opportunity
+ 0.18 Best System Fit
+ 0.15 Age Upside
- superstar / elite-club visibility penalty

Hidden Gem Opportunity =
Al Gharafa tactical fit
+ under-the-radar score
+ age upside
+ system fit""",
        language="text",
    )


    dna_tables = []
    if not formation_summary.empty:
        dna_tables.append(("Formation summary", report_table(formation_summary, ["Formation","Matches","UsageShare%","PPG","GF","GA","xG","Possession","PPDA","BuildUp","Progression","FinalThird","Wide","Transition","Pressing","Defensive"], 25)))
    if not dna_by_season.empty:
        dna_tables.append(("DNA by season", dna_by_season.copy()))
    page_report_download(
        "Download DNA report HTML",
        "Al Gharafa SC — Tactical DNA Report",
        "Club striker model and recruitment logic",
        [
            "Al Gharafa are interpreted as a 4-2-3-1-led hybrid side with controlled build-up, structured progression, wide individual creation and situational pressing.",
            "The striker model is not built for a poacher-only profile. It rewards a forward who can attack the box, link with the No.10 and wide players, run channels, provide an aerial outlet and support the team's mid-block work.",
            "The three key recruitment contexts used in the model are 4-2-3-1, 3-4-3 and 4-4-2 because they cover the main tactical roles a striker is likely to perform inside the Al Gharafa structure.",
        ],
        figures=[],
        tables=dna_tables,
        filename="al_gharafa_tactical_dna_report.html",
        key="dna_report_download",
    )

elif page == "Similarity Scouting":
    st.markdown('<div class="section-header">Similarity Scouting</div>', unsafe_allow_html=True)
    st.caption("Choose a benchmark striker, then the tool recommends similar profiles using role scores, phase DNA and system-fit behaviour.")

    benchmark_pool = players.sort_values("Al Gharafa ST Fit", ascending=False).copy()
    benchmark_names = benchmark_pool["Name"].astype(str).tolist()
    benchmark_name = st.selectbox("Benchmark player", benchmark_names, index=0, key="similarity_benchmark")
    benchmark_row = benchmark_pool[benchmark_pool["Name"].astype(str) == benchmark_name].iloc[0]

    c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
    with c1:
        top_n = st.slider("Number of similar players", 5, 30, 12, 1)
    with c2:
        apply_current_filters = st.checkbox("Apply current sidebar filters to candidates", value=True)
    with c3:
        exclude_current_club = st.checkbox("Exclude current Al Gharafa players", value=True)

    candidate_pool = df.copy() if apply_current_filters else players.copy()
    if exclude_current_club and "Club" in candidate_pool.columns:
        candidate_pool = candidate_pool[~candidate_pool["Club"].astype(str).str.contains("gharafa", case=False, na=False)]

    sims = similar_players(benchmark_row, candidate_pool, top_n=top_n)

    b1, b2 = st.columns([0.36, 0.64], vertical_alignment="center")
    with b1:
        if "image_url" in benchmark_row.index:
            img_url = "" if pd.isna(benchmark_row.get("image_url", "")) else str(benchmark_row.get("image_url", "")).strip()
            if img_url and img_url.lower() not in ["nan", "none", "unknown", "-", ""]:
                try:
                    st.image(img_url, width=135)
                except Exception:
                    if LOGO_PATH.exists():
                        st.image(str(LOGO_PATH), width=145)
            elif LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=145)
        elif LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=145)
        st.markdown(f"### Benchmark: {benchmark_row['Name']}")
        st.markdown(f"**{player_identity_line(benchmark_row)}**")
        st.markdown(f"<span class='tag'>{benchmark_row['Best Role']}</span><span class='tag'>{benchmark_row['Best System']}</span><span class='tag'>ST Fit {benchmark_row['Al Gharafa ST Fit']:.1f}</span>", unsafe_allow_html=True)
    with b2:
        st.plotly_chart(radar_chart(benchmark_row, "Benchmark striker profile"), width="stretch")

    if sims.empty:
        st.warning("No similar candidates found under the current filters. Try removing some filters or allowing current Al Gharafa players.")
    else:
        sim_cols = ["Name","Club","Age","Country","Continent","Best Role","Scout Tag","Similarity Score","ST Fit Gap vs Benchmark","Scouting Opportunity","Al Gharafa ST Fit","Hidden Gem Opportunity","Best System"]
        st.markdown("### Recommended similar profiles")
        st.dataframe(report_table(sims, sim_cols, top_n), width="stretch", hide_index=True)

        sim_comp = pd.concat([benchmark_row.to_frame().T, sims.head(4)], ignore_index=True)
        st.plotly_chart(multi_radar(sim_comp, "Benchmark vs closest similar profiles"), width="stretch")

        top_match = sims.iloc[0]
        st.markdown(
            f"""
            <div class="navy-card">
            <b>Similarity interpretation:</b> the closest profile to <b>{benchmark_row['Name']}</b> under the current filters is <b>{top_match['Name']}</b> with a similarity score of <b>{top_match['Similarity Score']:.1f}%</b>.<br><br>
            This does not mean he is better. It means his role/phase/system profile is close enough to justify video review as a comparable or replacement-style profile.
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_report_download(
            "Download similarity scouting report HTML",
            "Al Gharafa SC — Similarity Scouting Report",
            f"Benchmark: {benchmark_row['Name']}",
            [
                f"This report uses {benchmark_row['Name']} as the benchmark profile and searches for similar strikers based on role scores, Al Gharafa striker phases and system-fit behaviour.",
                "Similarity is not a final quality ranking. It is a scouting workflow that helps answer: if the club likes this benchmark profile, which players in the database behave most similarly?",
                "The top similar players should be treated as video-review recommendations and checked for league context, physical profile, contract situation and tactical role before any recruitment decision.",
            ],
            figures=[("Benchmark vs closest profiles", multi_radar(sim_comp, "Benchmark vs closest profiles"))],
            tables=[("Similarity recommendations", report_table(sims, sim_cols, top_n))],
            filename="al_gharafa_similarity_scouting_report.html",
            key="similarity_report_download",
        )
