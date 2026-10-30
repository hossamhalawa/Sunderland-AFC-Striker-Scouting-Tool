from __future__ import annotations

import base64
import html
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import unquote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# -----------------------------
# App configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "strikers.csv"
LOGO_PATH = BASE_DIR / "assets" / "sunderland_afc_logo.png"
DNA_SEASON_PATH = BASE_DIR / "data" / "dna_by_season.csv"
FORMATION_SUMMARY_PATH = BASE_DIR / "data" / "formation_summary.csv"
FORMATION_SEASON_PATH = BASE_DIR / "data" / "formation_by_season.csv"
OPEN_DATA_DIR = BASE_DIR / "data" / "open_data"
OPEN_DNA_PATH = OPEN_DATA_DIR / "sunderland_dna_open_snapshot.csv"
OPEN_CURRENT_ST_PATH = OPEN_DATA_DIR / "sunderland_current_strikers_open.csv"
OPEN_LEADERS_PATH = OPEN_DATA_DIR / "premier_league_open_leaders.csv"
OPEN_STATUS_PATH = OPEN_DATA_DIR / "data_status.json"
UNDERSTAT_DIR = BASE_DIR / "data" / "understat"
UNDERSTAT_DNA_PATH = UNDERSTAT_DIR / "sunderland_dna_coverage.csv"
UNDERSTAT_FORMATION_PATH = UNDERSTAT_DIR / "sunderland_formation_summary.csv"
UNDERSTAT_MATCHES_PATH = UNDERSTAT_DIR / "sunderland_match_history.csv"
UNDERSTAT_PLAYERS_PATH = UNDERSTAT_DIR / "premier_league_player_metrics.csv"
UNDERSTAT_STATUS_PATH = UNDERSTAT_DIR / "source_status.json"
PASSMAP_DIR = BASE_DIR / "data" / "passmaps"
PASSMAP_TEAM_PATH = PASSMAP_DIR / "premier_league_pass_metrics.csv"
PASSMAP_PLAYERS_PATH = PASSMAP_DIR / "sunderland_passmap_players.csv"
PASSMAP_STATUS_PATH = PASSMAP_DIR / "source_status.json"
EFEM_FULL_ATTRIBUTES_DIR = BASE_DIR / "data" / "efem_full_attributes"
EFEM_FULL_ATTRIBUTES_PATH = EFEM_FULL_ATTRIBUTES_DIR / "efem_striker_full_attributes.csv"

NAVY = "#111111"
NAVY_2 = "#242424"
GOLD = "#E30613"
GOLD_2 = "#FF5A5F"
WHITE = "#FFFFFF"
OFFWHITE = "#F6F8FB"
MUTED = "#6B7280"
RED = "#A9000B"
BLUE = "#8B8B8B"
LIGHT_NAVY = "#F6D6D8"

EFEM_ATTRIBUTE_WEIGHTS = {
    "Finishing & box craft": {"Finishing": .40, "Composure": .20, "Off The Ball": .20, "Anticipation": .10, "Technique": .10},
    "Link-up & creativity": {"First Touch": .25, "Passing": .25, "Vision": .20, "Decisions": .15, "Teamwork": .15},
    "Carrying & 1v1": {"Dribbling": .30, "Agility": .20, "Acceleration": .20, "Balance": .15, "Flair": .15},
    "Channel threat": {"Pace": .35, "Acceleration": .25, "Off The Ball": .25, "Anticipation": .15},
    "Aerial presence": {"Heading": .35, "Jumping Reach": .25, "Strength": .20, "Bravery": .15, "Balance": .05},
    "Pressing engine": {"Work Rate": .35, "Aggression": .20, "Stamina": .20, "Teamwork": .15, "Bravery": .10},
    "Set pieces": {"Penalties": .40, "Free Kicks": .30, "Corners": .20, "Long Throw": .10},
}
EFEM_ATTRIBUTE_GROUPS = {label: list(weights) for label, weights in EFEM_ATTRIBUTE_WEIGHTS.items()}
EFEM_RADAR_LABELS = {
    "Finishing & box craft": "Finishing",
    "Link-up & creativity": "Link-up",
    "Carrying & 1v1": "1v1 carrying",
    "Channel threat": "Channel runs",
    "Aerial presence": "Aerial",
    "Pressing engine": "Pressing",
    "Set pieces": "Set pieces",
}

st.set_page_config(
    page_title="Sunderland AFC | Striker Scouting Tool",
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
        background: linear-gradient(180deg, {OFFWHITE} 0%, #FFFFFF 32%, #FFFFFF 100%);
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
        border-color: rgba(227,6,19,0.55) !important;
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
    section[data-testid="stSidebar"] .stImage {{
        margin: 2px 0 2px 0;
    }}
    .sidebar-club-name {{
        color: {WHITE};
        font-size: 20px;
        font-weight: 850;
        line-height: 1.1;
        margin: 4px 0 2px;
    }}
    .sidebar-club-note {{
        color: rgba(255,255,255,.70);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .03em;
        margin-bottom: 8px;
    }}
    .main-title {{
        font-size: 30px;
        font-weight: 850;
        color: {NAVY};
        margin-bottom: 0px;
        line-height: 1.12;
    }}
    .subtitle {{
        color: {MUTED};
        font-size: 14px;
        line-height: 1.45;
        margin-top: 5px;
    }}
    .card {{
        background: {WHITE};
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(11,31,58,0.06);
        margin-bottom: 10px;
    }}
    .navy-card {{
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_2} 100%);
        color: {WHITE};
        border-radius: 14px;
        padding: 21px 22px;
        line-height: 1.55;
        box-shadow: 0 10px 24px rgba(11,31,58,0.15);
        border: 1px solid rgba(227,6,19,0.35);
    }}
    .metric-card {{
        background: {WHITE};
        border-radius: 14px;
        padding: 16px 17px;
        border-top: 3px solid {GOLD};
        box-shadow: 0 6px 18px rgba(11,31,58,0.06);
        min-height: 104px;
    }}
    .metric-label {{color: {MUTED}; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .06em;}}
    .metric-value {{font-size: 27px; color: {NAVY}; font-weight: 850; margin-top: 6px; line-height: 1.1;}}
    .tag {{
        display:inline-flex;
        align-items:center;
        background: rgba(227,6,19,0.10);
        color: {NAVY};
        border: 1px solid rgba(227,6,19,0.32);
        border-radius: 8px;
        padding: 4px 8px;
        font-weight: 800;
        font-size: 11px;
        margin: 2px 3px 2px 0;
    }}
    .gold-line {{height:3px;background:{GOLD};border-radius:4px;margin:12px 0 20px 0;}}
    .small-note {{color:{MUTED};font-size:12px;line-height:1.45;}}
    .section-header {{
        font-size: 22px;
        font-weight: 850;
        color: {NAVY};
        line-height: 1.2;
        margin-top: 6px;
        margin-bottom: 9px;
    }}
    .dna-hero {{
        background: radial-gradient(circle at 88% 18%, rgba(227,6,19,.92) 0 5%, transparent 5.4%), linear-gradient(125deg, #0B0B0B 0%, #242424 56%, #4A0A0D 100%);
        color: {WHITE};
        border-radius: 26px;
        padding: 30px 32px;
        margin: 8px 0 18px 0;
        box-shadow: 0 18px 36px rgba(11,11,11,.22);
        overflow: hidden;
    }}
    .dna-kicker {{
        color: #FFB7BA;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .dna-hero-title {{
        font-size: 31px;
        line-height: 1.06;
        font-weight: 900;
        margin-bottom: 10px;
    }}
    .dna-hero-copy {{
        color: rgba(255,255,255,.78);
        font-size: 15px;
        max-width: 780px;
        line-height: 1.55;
    }}
    .dna-strip {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
    }}
    .dna-strip span {{
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.18);
        color: {WHITE};
        font-size: 12px;
        font-weight: 800;
        padding: 7px 10px;
        border-radius: 999px;
    }}
    .visual-caption {{
        color: {MUTED};
        font-size: 12px;
        line-height: 1.45;
        margin: -4px 0 8px 0;
    }}
    .scout-reading {{
        background: linear-gradient(135deg, #111111 0%, #292929 100%);
        border-left: 4px solid {GOLD};
        border-radius: 12px;
        color: {WHITE};
        padding: 17px 19px;
        line-height: 1.55;
        margin: 4px 0 12px;
    }}
    .scout-reading .eyebrow {{
        color: #FFB7BA;
        font-size: 11px;
        font-weight: 850;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}
    .scout-reading strong {{ color: {WHITE}; }}
    .similarity-card {{
        background: {WHITE};
        border: 1px solid #E4E7EB;
        border-radius: 16px;
        padding: 14px;
        margin: 0 0 12px;
        box-shadow: 0 7px 18px rgba(17,17,17,.06);
        min-height: 126px;
    }}
    .similarity-rank {{
        color: {GOLD};
        font-size: 11px;
        font-weight: 900;
        letter-spacing: .08em;
        text-transform: uppercase;
    }}
    .similarity-name {{
        color: {NAVY};
        font-size: 17px;
        font-weight: 850;
        line-height: 1.15;
        margin: 3px 0;
    }}
    .similarity-meta {{ color: {MUTED}; font-size: 12px; line-height: 1.35; }}
    .similarity-score {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 58px;
        height: 31px;
        padding: 0 9px;
        border-radius: 999px;
        background: rgba(227,6,19,.10);
        color: {GOLD};
        font-size: 14px;
        font-weight: 900;
        font-variant-numeric: tabular-nums;
    }}
    .similarity-trait {{
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #EEF0F2;
        color: #3F4650;
        font-size: 12px;
        line-height: 1.4;
    }}
    div[data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 6px 18px rgba(11,31,58,0.06);
    }}
    .stTabs [data-baseweb="tab-list"] {{gap: 8px;}}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 13px;
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


def player_image_url(value: object) -> str:
    """Use the original portrait URL where available, avoiding a fragile image proxy."""
    if pd.isna(value):
        return ""
    url = str(value).strip()
    if not url or url.lower() in {"nan", "none", "unknown", "-"}:
        return ""
    if "efem.club/_next/image" in url:
        source_match = re.search(r"[?&]url=([^&]+)", url)
        if source_match:
            source_url = unquote(source_match.group(1)).strip()
            if source_url.startswith(("https://", "http://")):
                return source_url
    return url


def render_player_image(row: pd.Series, width: int = 132) -> None:
    """Show every available player image at the highest available resolution."""
    img_url = player_image_url(row.get("image_url", ""))
    if img_url:
        try:
            st.image(img_url, width=width)
            return
        except Exception:
            pass
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=width)


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


def relative_band(value: float) -> str:
    """Short visual label for non-comparison charts."""
    if value >= 75:
        return "High"
    if value >= 55:
        return "Above average"
    if value >= 40:
        return "Balanced"
    return "Lower"


def passmap_percentile(teams: pd.DataFrame, row: pd.Series, metric: str) -> float:
    values = pd.to_numeric(teams[metric], errors="coerce").dropna()
    return float((values <= float(row[metric])).mean() * 100) if not values.empty else 50.0


def make_dna_visual_profile(dna: pd.DataFrame, passmap_teams: pd.DataFrame) -> pd.DataFrame:
    """Blend only source-backed public measures into a clean visual profile."""
    core = {
        str(row["DNA pillar"]): float(row["League percentile"])
        for _, row in dna.dropna(subset=["League percentile"]).iterrows()
    }
    profile = [
        ("Build-up", core.get("Build-up Control", 50.0)),
        ("Progression", core.get("Progression", 50.0)),
        ("Final-third access", core.get("Final Third Access", 50.0)),
        ("Press / regain", core.get("Pressing / Regain", 50.0)),
        ("Defensive control", core.get("Defensive Control", 50.0)),
    ]
    if not passmap_teams.empty:
        sunderland = passmap_teams[passmap_teams["Team"].astype(str).str.casefold().eq("sunderland")]
        if not sunderland.empty:
            row = sunderland.iloc[0]
            progression_overlay = np.mean([
                passmap_percentile(passmap_teams, row, "Progressive passes per 90"),
                passmap_percentile(passmap_teams, row, "xT per 90"),
            ])
            width_overlay = passmap_percentile(passmap_teams, row, "Crosses per 90")
            profile = [
                (name, float(np.mean([score, progression_overlay])) if name == "Progression" else score)
                for name, score in profile
            ]
            profile.insert(3, ("Width / crossing", width_overlay))
    result = pd.DataFrame(profile, columns=["Dimension", "Percentile"])
    result["Band"] = result["Percentile"].map(relative_band)
    return result


def make_forward_evidence(raw_players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a transparent forward profile from the metrics Understat actually provides."""
    if raw_players.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = raw_players.copy()
    data = data[data["Position"].astype(str).str.contains("F", na=False)].copy()
    numeric_cols = ["Minutes", "Goals", "Assists", "xG", "xA", "Shots", "Key Passes", "Non-penalty Goals", "Non-penalty xG", "xG Chain", "xG Buildup"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    data = data[data["Minutes"] >= 180].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    per_90 = lambda col: data[col] * 90 / data["Minutes"].replace(0, np.nan)
    data["NPG per 90"] = per_90("Non-penalty Goals")
    data["xG per 90 live"] = per_90("Non-penalty xG")
    data["xA per 90 live"] = per_90("xA")
    data["Shots per 90"] = per_90("Shots")
    data["Key passes per 90"] = per_90("Key Passes")
    data["xG Chain per 90"] = per_90("xG Chain")
    data["xG Buildup per 90"] = per_90("xG Buildup")

    # A percentile is a contextual visual benchmark, not an invented football metric.
    dimensions = {
        "Finishing": ["NPG per 90", "xG per 90 live"],
        "Shot volume": ["Shots per 90"],
        "Creation": ["xA per 90 live", "Key passes per 90"],
        "Attack involvement": ["xG Chain per 90", "xG Buildup per 90"],
    }
    for label, cols in dimensions.items():
        metric_pct = pd.concat([data[col].rank(pct=True) * 100 for col in cols], axis=1).mean(axis=1)
        data[label] = metric_pct
    sunderland = data[data["Club"].astype(str).str.contains("sunderland", case=False, na=False)].copy()
    return data, sunderland


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
    df["Obvious Elite Player"] = df["Elite Club Flag"] | df["Famous Name Flag"] | ((df["Generic Role Level"] >= 83) & (df["Sunderland AFC ST Fit"] >= 80))
    return df

# -----------------------------
# Data loading and scoring
# -----------------------------
@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    # Current Sunderland players are deliberately not added as estimated role-score
    # rows. Their public, observed data is shown separately on Current ST Assessment.
    if "Club" in df.columns:
        # Remove the source tool's former club benchmarks from the Sunderland universe.
        df = df[~df["Club"].astype(str).str.contains("gharafa", case=False, na=False)].copy()
        df.loc[df["Club"].astype(str).str.contains("sunderland", case=False, na=False), "Club"] = "Sunderland AFC"
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

    # Sunderland striker phases: vertical running and first-line defensive work
    # carry more weight than the source model.
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

    df["Sunderland AFC ST Fit"] = clamp_series(
        0.22*df["Box Threat"] + 0.18*df["Link-up Play"] + 0.22*df["Channel Running"] + 0.15*df["Wide Combination"] + 0.08*df["Aerial Outlet"] + 0.15*df["Defensive Work"]
    )
    df["4-2-3-1 Fit"] = clamp_series(
        0.27*df["Box Threat"] + 0.24*df["Link-up Play"] + 0.19*df["Wide Combination"] + 0.14*df["Channel Running"] + 0.10*df["Defensive Work"] + 0.06*df["Aerial Outlet"]
    )
    df["3-4-3 Fit"] = clamp_series(
        0.24*df["Channel Running"] + 0.22*df["Wide Combination"] + 0.19*df["Box Threat"] + 0.14*df["Link-up Play"] + 0.12*df["Defensive Work"] + 0.09*df["Aerial Outlet"]
    )
    df["4-3-3 Fit"] = clamp_series(
        0.20*df["Box Threat"] + 0.14*df["Link-up Play"] + 0.25*df["Channel Running"] + 0.21*df["Wide Combination"] + 0.07*df["Aerial Outlet"] + 0.13*df["Defensive Work"]
    )
    df["Best System Fit"] = df[["4-2-3-1 Fit", "3-4-3 Fit", "4-3-3 Fit"]].max(axis=1)
    df["Best System"] = df[["4-2-3-1 Fit", "3-4-3 Fit", "4-3-3 Fit"]].idxmax(axis=1).str.replace(" Fit", "", regex=False)

    role_cols = role_columns(df)
    df["Generic Role Level"] = df[role_cols].mean(axis=1) if role_cols else df["Sunderland AFC ST Fit"]
    age = df["Age"].fillna(26)
    df["Age Upside"] = np.select(
        [age <= 20, age <= 23, age <= 25, age <= 27, age <= 30],
        [100, 92, 80, 66, 52],
        default=35,
    )
    df["Specialist Fit Gap"] = (df["Sunderland AFC ST Fit"] - df["Generic Role Level"]).clip(-30, 30)
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
        0.48*df["Sunderland AFC ST Fit"] + 0.24*df["Under-the-Radar Score"] + 0.16*df["Age Upside"] + 0.12*df["Best System Fit"] - 0.55*elite_penalty
    )
    df["Scouting Opportunity"] = clamp_series(
        0.42*df["Sunderland AFC ST Fit"] + 0.25*df["Hidden Gem Opportunity"] + 0.18*df["Best System Fit"] + 0.15*df["Age Upside"] - 0.75*elite_penalty
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
        (not_obvious & (df["Scouting Opportunity"] >= 76) & (df["Sunderland AFC ST Fit"] >= 58)),
        (not_obvious & (df["Hidden Gem Opportunity"] >= 68) & (df["Sunderland AFC ST Fit"] >= 56)),
        (not_obvious & (df["Age"] <= 21) & (df["Sunderland AFC ST Fit"] >= 54)),
        (not_obvious & (df["Best System Fit"] >= 72) & (df["Sunderland AFC ST Fit"] >= 55)),
        (not_obvious & (df["Age"] <= 24) & (df["Hidden Gem Opportunity"] >= 58)),
        (df["Sunderland AFC ST Fit"] >= 58),
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

    # Verified current Sunderland central-forward benchmark group.
    current_manual = {
        "brian brobbey": ("Netherlands", ""),
        "wilson isidor": ("France", ""),
        "eliezer mayenda": ("Spain", ""),
        "timur tutierov": ("Ukraine", ""),
    }
    name_norm = df["Name"].apply(normalise_text)
    club_norm = df["Club"].apply(normalise_text)
    for n_key, (c1, c2) in current_manual.items():
        mask = (name_norm == n_key) & (club_norm.str.contains("sunderland", na=False) | name_norm.isin(current_manual.keys()))
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


@st.cache_data(show_spinner=False)
def load_open_data():
    dna = pd.read_csv(OPEN_DNA_PATH) if OPEN_DNA_PATH.exists() else pd.DataFrame()
    current = pd.read_csv(OPEN_CURRENT_ST_PATH) if OPEN_CURRENT_ST_PATH.exists() else pd.DataFrame()
    leaders = pd.read_csv(OPEN_LEADERS_PATH) if OPEN_LEADERS_PATH.exists() else pd.DataFrame()
    status = {}
    if OPEN_STATUS_PATH.exists():
        try:
            status = json.loads(OPEN_STATUS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            status = {}
    return dna, current, leaders, status


@st.cache_data(show_spinner=False)
def load_understat_data():
    coverage = pd.read_csv(UNDERSTAT_DNA_PATH) if UNDERSTAT_DNA_PATH.exists() else pd.DataFrame()
    formation = pd.read_csv(UNDERSTAT_FORMATION_PATH) if UNDERSTAT_FORMATION_PATH.exists() else pd.DataFrame()
    matches = pd.read_csv(UNDERSTAT_MATCHES_PATH) if UNDERSTAT_MATCHES_PATH.exists() else pd.DataFrame()
    players = pd.read_csv(UNDERSTAT_PLAYERS_PATH) if UNDERSTAT_PLAYERS_PATH.exists() else pd.DataFrame()
    status = {}
    if UNDERSTAT_STATUS_PATH.exists():
        try:
            status = json.loads(UNDERSTAT_STATUS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            status = {}
    return coverage, formation, matches, players, status


@st.cache_data(show_spinner=False)
def load_passmap_data():
    teams = pd.read_csv(PASSMAP_TEAM_PATH) if PASSMAP_TEAM_PATH.exists() else pd.DataFrame()
    players = pd.read_csv(PASSMAP_PLAYERS_PATH) if PASSMAP_PLAYERS_PATH.exists() else pd.DataFrame()
    status = {}
    if PASSMAP_STATUS_PATH.exists():
        try:
            status = json.loads(PASSMAP_STATUS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            status = {}
    return teams, players, status


def efem_full_attribute_signature() -> tuple[tuple[str, int, int], ...]:
    files = []
    if EFEM_FULL_ATTRIBUTES_PATH.exists():
        files.append(EFEM_FULL_ATTRIBUTES_PATH)
    shard_dir = EFEM_FULL_ATTRIBUTES_DIR / "shards"
    if shard_dir.exists():
        files.extend(sorted(shard_dir.glob("efem_striker_full_attributes_*.csv")))
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)


@st.cache_data(show_spinner=False)
def load_efem_full_attributes(_signature: tuple[tuple[str, int, int], ...]) -> pd.DataFrame:
    """Load only verified, exact-URL EFEM full attribute rows.

    Failed name matches remain absent, so the app can preserve their legacy radar
    rather than attaching another player's data by inference.
    """
    files = []
    if EFEM_FULL_ATTRIBUTES_PATH.exists():
        files.append(EFEM_FULL_ATTRIBUTES_PATH)
    shard_dir = EFEM_FULL_ATTRIBUTES_DIR / "shards"
    if shard_dir.exists():
        files.extend(sorted(shard_dir.glob("efem_striker_full_attributes_*.csv")))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(path, dtype={"row_id": str}) for path in files if path.stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["row_id"] = data["row_id"].astype(str)
    return data.drop_duplicates(subset=["row_id"], keep="last")

players = load_players()
dna_by_season, formation_summary, formation_by_season = load_dna_tables()
open_dna, open_current_st, open_leaders, open_status = load_open_data()
understat_dna, understat_formation, understat_matches, understat_players, understat_status = load_understat_data()
passmap_teams, passmap_players, passmap_status = load_passmap_data()
efem_full_attributes = load_efem_full_attributes(efem_full_attribute_signature())
if not efem_full_attributes.empty and "row_id" in players.columns:
    players["row_id"] = players["row_id"].astype(str)
    attribute_columns = [c for c in efem_full_attributes.columns if c not in {"Name", "Club", "player_url", "efem_name", "retrieved_at_utc"}]
    # A handful of source-list rows are intentional duplicate references to the
    # same EFEM player ID. The imported table itself is unique by ID, so each
    # duplicate safely receives that exact verified profile.
    players = players.merge(efem_full_attributes[attribute_columns], on="row_id", how="left", validate="many_to_one")
    players["EFEM full attributes available"] = players["Finishing"].notna()
    # EFEM provides the source attributes on a 0–100 scale. Percentiles stay in
    # the hover detail only; the report radar must show the actual attribute
    # level, otherwise a strong player misleadingly looks "full" on every axis.
    for attribute in {attr for attrs in EFEM_ATTRIBUTE_GROUPS.values() for attr in attrs}:
        raw = pd.to_numeric(players[attribute], errors="coerce")
        players[f"EFEM percentile · {attribute}"] = raw.rank(pct=True) * 100
    for label, weights in EFEM_ATTRIBUTE_WEIGHTS.items():
        raw_weighted = sum(pd.to_numeric(players[attribute], errors="coerce") * weight for attribute, weight in weights.items())
        players[f"EFEM radar raw · {label}"] = raw_weighted
        players[f"EFEM radar median · {label}"] = raw_weighted.median()
else:
    players["EFEM full attributes available"] = False
PHASES = ["Box Threat", "Link-up Play", "Channel Running", "Wide Combination", "Aerial Outlet", "Defensive Work"]
SCORE_COLS = ["Scouting Opportunity", "Sunderland AFC ST Fit", "Hidden Gem Opportunity", "Under-the-Radar Score", "Best System Fit"]
ROLE_COLS = role_columns(players)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=68)
    st.markdown('<div class="sidebar-club-name">Sunderland AFC</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-club-note">Striker recruitment intelligence</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Sunderland AFC DNA",
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
    min_fit = st.slider("Minimum Sunderland AFC ST Fit", 0, 100, 48, 1)
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
df = df[(df["Sunderland AFC ST Fit"] >= min_fit) & (df["Hidden Gem Opportunity"] >= min_hidden)]
if scout_tag != "All":
    df = df[df["Scout Tag"] == scout_tag]
if role_filter != "All":
    df = df[df["Best Role"] == role_filter]
if "Continent" in df.columns and continent_filter != "All":
    df = df[df["Continent"].astype(str) == continent_filter]
if "Country" in df.columns and country_filter != "All":
    df = df[df["Country"].astype(str) == country_filter]

df = df.sort_values(["Scouting Opportunity", "Hidden Gem Opportunity", "Sunderland AFC ST Fit"], ascending=False)

# -----------------------------
# Header
# -----------------------------
logo_col, title_col = st.columns([0.10, 0.90], vertical_alignment="center")
with logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=58)
with title_col:
    st.markdown('<div class="main-title">Sunderland AFC Striker Scouting Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Hidden-gem discovery model aligned to Sunderland AFC tactical DNA under Regis Le Bris</div>', unsafe_allow_html=True)
st.markdown('<div class="gold-line"></div>', unsafe_allow_html=True)
st.caption("Use the filters to refine the player pool, then open a report for the detailed profile.")

# -----------------------------
# Plot functions
# -----------------------------
def has_full_efem_attributes(row: pd.Series) -> bool:
    return bool(row.get("EFEM full attributes available", False)) and all(pd.notna(row.get(attr)) for attrs in EFEM_ATTRIBUTE_GROUPS.values() for attr in attrs)


def radar_profile(row: pd.Series) -> tuple[list[str], list[float], list[str], bool]:
    """Return source-aware radar dimensions without substituting missing players."""
    if has_full_efem_attributes(row):
        source_labels = list(EFEM_ATTRIBUTE_GROUPS)
        labels = [EFEM_RADAR_LABELS[label] for label in source_labels]
        values = [float(row[f"EFEM radar raw · {label}"]) for label in source_labels]
        details = []
        for label, attrs in EFEM_ATTRIBUTE_GROUPS.items():
            parts = [f"{attr}: {float(row[attr]):.0f} ({float(row[f'EFEM percentile · {attr}']):.0f}th pct)" for attr in attrs]
            details.append(" · ".join(parts))
        return labels, values, details, True
    labels = PHASES
    values = [float(row[p]) for p in PHASES]
    return labels, values, ["Legacy role-model phase; verified EFEM detail unavailable for this player."] * len(labels), False


def radar_baseline(row: pd.Series, verified: bool) -> list[float]:
    """Return the true population median for each displayed radar dimension."""
    if verified:
        return [float(row[f"EFEM radar median · {label}"]) for label in EFEM_ATTRIBUTE_GROUPS]
    return [50.0] * len(PHASES)


def radar_chart(row: pd.Series, title: str = "Player phase profile") -> go.Figure:
    labels, values, details, verified = radar_profile(row)
    cats = labels + [labels[0]]
    vals = values + [values[0]]
    fig = go.Figure()
    if verified:
        baseline = radar_baseline(row, True)
        fig.add_trace(go.Scatterpolar(
            r=baseline + [baseline[0]], theta=cats, mode="lines", name="Striker-universe median",
            line=dict(color="#9CA3AF", width=2, dash="dot"),
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatterpolar(
        r=vals,
        theta=cats,
        mode="lines+markers",
        name=str(row.get("Name", "Player")),
        line=dict(color=GOLD, width=3),
        marker=dict(size=7, color=GOLD),
        customdata=details + [details[0]],
        hovertemplate="<b>%{theta}</b><br>Attribute level: %{r:.1f}/100<br><span style='font-size:11px'>%{customdata}</span><extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0)",
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[0, 25, 50, 75, 100], gridcolor="#D7DEE9", tickfont=dict(color=MUTED)),
            angularaxis=dict(gridcolor="#D7DEE9", tickfont=dict(size=12, color=NAVY)),
        ),
        showlegend=False,
        title=dict(text=title, font=dict(size=20, color=NAVY)),
    )
    return fig_layout(fig, 405)


def multi_radar(rows: pd.DataFrame, title: str) -> go.Figure:
    colors = [GOLD, NAVY, BLUE, RED, "#7C3AED"]
    fig = go.Figure()
    full_only = len(rows) > 0 and rows.apply(has_full_efem_attributes, axis=1).all()
    cats = [EFEM_RADAR_LABELS[label] for label in EFEM_ATTRIBUTE_GROUPS] if full_only else PHASES
    for idx, (_, row) in enumerate(rows.iterrows()):
        labels, values, details, _ = radar_profile(row) if full_only else (PHASES, [float(row[p]) for p in PHASES], ["Legacy phase model"] * len(PHASES), False)
        vals = values + [values[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats,
            mode="lines+markers",
            name=str(row["Name"]),
            line=dict(color=colors[idx % len(colors)], width=3),
            marker=dict(size=6, color=colors[idx % len(colors)]),
            opacity=0.78,
            customdata=details + [details[0]],
            hovertemplate=f"<b>{row['Name']}</b><br>%{{theta}}: %{{r:.1f}} / 100<br><span style='font-size:11px'>%{{customdata}}</span><extra></extra>",
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
    labels, vals, details, verified = radar_profile(row)
    if verified:
        baselines = radar_baseline(row, True)
        # The neutral track keeps every dimension on the same raw 0–100 scale.
        # The coloured segment begins at the actual universe median for that
        # dimension, rather than a generic midpoint of 50.
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[100] * len(labels), y=labels, orientation="h",
            marker=dict(color="#E8EAED"), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Bar(
            x=[abs(value - baseline) for value, baseline in zip(vals, baselines)],
            base=[min(value, baseline) for value, baseline in zip(vals, baselines)],
            y=labels, orientation="h",
            marker=dict(color=[GOLD if value >= baseline else "#70757A" for value, baseline in zip(vals, baselines)]),
            customdata=[[value, baseline, detail] for value, baseline, detail in zip(vals, baselines, details)],
            hovertemplate="<b>%{y}</b><br>Attribute level: %{customdata[0]:.0f}/100<br>Universe median: %{customdata[1]:.0f}<br><span style='font-size:11px'>%{customdata[2]}</span><extra></extra>",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[105] * len(labels), y=labels, mode="text",
            text=[f"{value:.0f}" for value in vals],
            textfont=dict(color=NAVY, size=13), hoverinfo="skip", showlegend=False,
        ))
        fig.update_xaxes(
            range=[0, 110], tickvals=[0, 25, 50, 75, 100],
            gridcolor="rgba(0,0,0,0)", title="Attribute level (0–100)",
        )
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(title="What the radar means", barmode="overlay", showlegend=False)
        return fig_layout(fig, 405, bottom_legend=False)

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker=dict(color=GOLD),
        text=[f"{value:.1f}" for value in vals], textposition="outside",
        customdata=details,
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<br><span style='font-size:11px'>%{customdata}</span><extra></extra>",
    ))
    fig.update_xaxes(range=[0, 105], gridcolor="#E5E7EB")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Phase strengths", showlegend=False)
    return fig_layout(fig, 405, bottom_legend=False)


def player_report_visual(row: pd.Series) -> str:
    """Responsive report visual with a true 0–100 radar and median-aligned bars."""
    labels, values, details, verified = radar_profile(row)
    baselines = radar_baseline(row, verified)
    n = len(labels)
    cx, cy, radius = 230, 205, 130

    def point(index: int, scale: float) -> tuple[float, float]:
        angle = -math.pi / 2 + (2 * math.pi * index / n)
        return cx + math.cos(angle) * radius * scale, cy + math.sin(angle) * radius * scale

    def points(scale: float) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, scale) for i in range(n)))

    def value_points(source_values: list[float]) -> str:
        return " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (point(i, max(0, min(100, value)) / 100) for i, value in enumerate(source_values))
        )

    grids = "".join(f'<polygon points="{points(scale)}" class="spr-grid" />' for scale in (1, .75, .5, .25))
    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{point(i, 1)[0]:.1f}" y2="{point(i, 1)[1]:.1f}" class="spr-grid" />'
        for i in range(n)
    )
    profile_points = value_points(values)
    median_points = value_points(baselines)
    dots = ""
    axis_labels = ""
    for i, (label, value, detail) in enumerate(zip(labels, values, details)):
        x, y = point(i, max(0, min(100, value)) / 100)
        lx, ly = point(i, 1.22)
        anchor = "end" if lx < 185 else "start" if lx > 275 else "middle"
        tooltip = html.escape(f"{label}: {value:.0f}/100. Universe median: {baselines[i]:.0f}. {detail}")
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="spr-dot"><title>{tooltip}</title></circle>'
        axis_labels += f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" class="spr-axis-label">{html.escape(label)}</text>'

    bars = ""
    for label, value, baseline, detail in zip(labels, values, baselines, details):
        safe_value = max(0, min(100, float(value)))
        safe_baseline = max(0, min(100, float(baseline)))
        start = min(safe_value, safe_baseline)
        width = abs(safe_value - safe_baseline)
        bar_class = "spr-bar-above" if safe_value >= safe_baseline else "spr-bar-below"
        bars += f'''
          <div class="spr-bar-row" title="{html.escape(detail)}">
            <div class="spr-bar-label">{html.escape(label)}</div>
            <div class="spr-track"><span class="spr-median" style="left:{safe_baseline:.2f}%"></span><span class="spr-bar {bar_class}" style="left:{start:.2f}%;width:{width:.2f}%"></span></div>
            <div class="spr-value">{safe_value:.0f}</div>
          </div>'''

    status = "Detailed attributes" if verified else "Phase profile"
    return """<style>
    #sunderland-player-report { margin: 6px 0 18px; color: #111111; }
    #sunderland-player-report * { box-sizing: border-box; }
    #sunderland-player-report .spr-shell { border: 1px solid #dededc; border-radius: 18px; overflow: hidden; background: #f7f7f6; }
    #sunderland-player-report .spr-topline { height: 7px; background: #e30613; }
    #sunderland-player-report .spr-header { display:flex; justify-content:space-between; gap:16px; align-items:center; padding:20px 24px 16px; background:#ffffff; }
    #sunderland-player-report .spr-eyebrow { color:#e30613; font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    #sunderland-player-report .spr-heading { margin:4px 0 0; font-size:22px; font-weight:700; }
    #sunderland-player-report .spr-context { margin-top:5px; color:#656565; font-size:13px; }
    #sunderland-player-report .spr-tag { padding:8px 11px; border-radius:999px; background:#fbe4e6; color:#b1000e; font-size:12px; font-weight:700; white-space:nowrap; }
    #sunderland-player-report .spr-grid-layout { display:grid; grid-template-columns:minmax(0, .95fr) minmax(0, 1.05fr); gap:28px; padding:20px 24px 24px; }
    #sunderland-player-report .spr-panel-title { display:flex; align-items:baseline; justify-content:space-between; gap:8px; margin:0 0 8px; font-size:15px; font-weight:700; }
    #sunderland-player-report .spr-panel-note { color:#686868; font-size:12px; font-weight:500; white-space:nowrap; }
    #sunderland-player-report .spr-radar-wrap { min-height:365px; display:flex; align-items:center; justify-content:center; }
    #sunderland-player-report .spr-radar { display:block; width:100%; max-width:460px; overflow:visible; }
    #sunderland-player-report .spr-grid { fill:none; stroke:#d8d8d6; stroke-width:1; }
    #sunderland-player-report .spr-median-ring { fill:none; stroke:#8f8f8f; stroke-width:1.5; stroke-dasharray:4 4; }
    #sunderland-player-report .spr-profile { fill:none; stroke:#e30613; stroke-width:3; stroke-linejoin:round; }
    #sunderland-player-report .spr-dot { fill:#e30613; }
    #sunderland-player-report .spr-axis-label { fill:#242424; font-size:12px; font-weight:600; }
    #sunderland-player-report .spr-tick { fill:#777777; font-size:10px; }
    #sunderland-player-report .spr-bars { padding-top:10px; }
    #sunderland-player-report .spr-bar-row { display:grid; grid-template-columns:116px minmax(80px, 1fr) 34px; gap:10px; align-items:center; min-height:42px; }
    #sunderland-player-report .spr-bar-label { font-size:13px; font-weight:600; }
    #sunderland-player-report .spr-track { position:relative; height:10px; border-radius:99px; background:#e4e4e2; overflow:visible; }
    #sunderland-player-report .spr-median { position:absolute; left:50%; top:-5px; bottom:-5px; width:1px; background:#777777; }
    #sunderland-player-report .spr-bar { position:absolute; top:0; height:10px; border-radius:99px; }
    #sunderland-player-report .spr-bar-above { background:#e30613; }
    #sunderland-player-report .spr-bar-below { background:#70757a; }
    #sunderland-player-report .spr-value { font-size:13px; font-weight:700; text-align:right; font-variant-numeric:tabular-nums; }
    #sunderland-player-report .spr-legend { display:flex; flex-wrap:wrap; gap:14px; color:#626262; font-size:12px; margin-top:14px; }
    #sunderland-player-report .spr-legend-item { display:flex; align-items:center; gap:6px; }
    #sunderland-player-report .spr-swatch { width:16px; height:4px; border-radius:4px; background:#e30613; }
    #sunderland-player-report .spr-swatch-mid { width:1px; height:14px; background:#777777; }
    #sunderland-player-report .spr-footer { margin:0 24px 22px; padding:13px 15px; background:#ededeb; color:#454545; border-radius:10px; font-size:13px; }
    #sunderland-player-report .spr-footer strong { color:#151515; }
    @media (max-width: 860px) {
      #sunderland-player-report .spr-grid-layout { grid-template-columns:1fr; gap:16px; padding:18px; }
      #sunderland-player-report .spr-header { padding:18px; }
      #sunderland-player-report .spr-footer { margin:0 18px 18px; }
      #sunderland-player-report .spr-tag { display:none; }
    }
    @media (max-width: 430px) {
      #sunderland-player-report .spr-bar-row { grid-template-columns:94px minmax(70px, 1fr) 28px; gap:7px; }
      #sunderland-player-report .spr-bar-label { font-size:12px; }
      #sunderland-player-report .spr-axis-label { font-size:11px; }
    }
    </style>""" + f'''
    <section id="sunderland-player-report" aria-label="Sunderland striker profile report">
      <div class="spr-shell">
        <div class="spr-topline"></div>
        <div class="spr-header">
          <div><div class="spr-eyebrow">Sunderland AFC · Player report</div><div class="spr-heading">Striker profile</div><div class="spr-context">Attribute level (0–100) against the striker universe</div></div>
          <div class="spr-tag">{status}</div>
        </div>
        <div class="spr-grid-layout">
          <section aria-label="Radar overview"><div class="spr-panel-title"><span>Radar overview</span><span class="spr-panel-note">Raw level · 0–100</span></div><div class="spr-radar-wrap"><svg class="spr-radar" viewBox="0 0 460 410" role="img" aria-label="Seven-axis striker attribute radar"><title>Striker attribute radar</title>{grids}{axes}<polygon points="{median_points}" class="spr-median-ring" /><polygon points="{profile_points}" class="spr-profile" />{dots}{axis_labels}<text x="235" y="71" class="spr-tick">100</text><text x="235" y="136" class="spr-tick">50</text><text x="235" y="201" class="spr-tick">0</text></svg></div></section>
          <section aria-label="Radar explanation"><div class="spr-panel-title"><span>What the radar means</span><span class="spr-panel-note">Line = universe median</span></div><div class="spr-bars">{bars}</div><div class="spr-legend"><span class="spr-legend-item"><span class="spr-swatch"></span>Player attribute level</span><span class="spr-legend-item"><span class="spr-swatch spr-swatch-mid"></span>Striker-universe median</span></div></section>
        </div>
        <div class="spr-footer"><strong>Scout read:</strong> the radar shows raw attribute levels; the aligned bars show the distance from the striker-universe median.</div>
      </div>
    </section>'''


def detailed_attribute_group_chart(row: pd.Series, group: str) -> go.Figure:
    attrs = EFEM_ATTRIBUTE_GROUPS[group]
    raw_values = [float(row[attr]) for attr in attrs]
    percentiles = [float(row[f"EFEM percentile · {attr}"]) for attr in attrs]
    fig = go.Figure(go.Bar(
        x=percentiles, y=attrs, orientation="h",
        marker=dict(color=[GOLD if value >= 75 else NAVY if value >= 50 else "#9CA3AF" for value in percentiles]),
        text=[f"Raw {raw:.0f}" for raw in raw_values], textposition="outside",
        customdata=raw_values,
        hovertemplate="<b>%{y}</b><br>Striker-universe percentile: %{x:.0f}<br>Raw attribute: %{customdata:.0f}<extra></extra>",
    ))
    fig.add_vline(x=50, line_color="#AEB6C2", line_width=1.5, line_dash="dot", annotation_text="middle of striker universe", annotation_font_color=MUTED)
    fig.update_xaxes(range=[0, 108], gridcolor="#E5E7EB", title="Percentile across all strikers")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"{group} · detailed attributes", showlegend=False)
    return fig_layout(fig, 410)


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
        y="Sunderland AFC ST Fit",
        size="Hidden Gem Opportunity",
        color="Scout Tag",
        hover_name="Name",
        hover_data={"Club": True, "Age": True, "Best Role": True, "Scouting Opportunity": ":.1f", "Under-the-Radar Score": ":.1f", "Generic Role Level": ":.1f", "Sunderland AFC ST Fit": ":.1f"},
        color_discrete_sequence=[GOLD, NAVY, BLUE, "#8B5CF6", RED, "#6B7280"],
    )
    fig.add_shape(type="line", x0=0, x1=100, y0=0, y1=100, line=dict(color="#9CA3AF", dash="dash"))
    fig.add_annotation(x=45, y=82, text="High tactical fit / less generic profile", showarrow=False, font=dict(color=NAVY, size=12), bgcolor="rgba(227,6,19,0.18)")
    fig.update_xaxes(title="Generic striker profile level", range=[35, 100], gridcolor="#E5E7EB")
    fig.update_yaxes(title="Sunderland AFC ST Fit", range=[35, 100], gridcolor="#E5E7EB")
    fig.update_layout(title="Hidden Gem Map: realistic fit versus generic profile", legend_title_text="")
    return fig_layout(fig, 540)


def difference_bars(rows: pd.DataFrame) -> go.Figure:
    if len(rows) < 2:
        return go.Figure()
    full_only = rows.apply(has_full_efem_attributes, axis=1).all()
    labels = [EFEM_RADAR_LABELS[label] for label in EFEM_ATTRIBUTE_GROUPS] if full_only else PHASES
    values_by_row = []
    for _, row in rows.iterrows():
        _, values, _, _ = radar_profile(row) if full_only else (PHASES, [float(row[p]) for p in PHASES], [], False)
        values_by_row.append(values)
    mean_vals = np.mean(np.asarray(values_by_row, dtype=float), axis=0)
    plot_rows = []
    for (_, r), values in zip(rows.iterrows(), values_by_row):
        for label, value, mean_value in zip(labels, values, mean_vals):
            plot_rows.append({"Player": r["Name"], "Phase": label, "Difference vs group": float(value - mean_value)})
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
    fig.update_layout(title="Where each player is stronger or weaker than the comparison group")
    fig.update_xaxes(gridcolor="#E5E7EB")
    return fig_layout(fig, 460)


def profile_phase_values(row: pd.Series) -> list[float]:
    """Keep the six recruitment phases as the common visual language for every player."""
    values = []
    for phase in PHASES:
        value = pd.to_numeric(row.get(phase, 0), errors="coerce")
        values.append(float(np.clip(0 if pd.isna(value) else value, 0, 100)))
    return values


def scout_canvas_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Place profiles on an interpretable scout map; it never changes the similarity ranking."""
    canvas = data.copy()
    for phase in PHASES:
        canvas[phase] = pd.to_numeric(canvas.get(phase, 0), errors="coerce").fillna(0).clip(0, 100)
    # Horizontal: connector / combination forward to vertical box runner.
    canvas["Canvas X"] = ((canvas["Channel Running"] + canvas["Box Threat"]) - (canvas["Link-up Play"] + canvas["Wide Combination"])) / 2
    # Vertical: physical central target to pressing / mobile support forward.
    canvas["Canvas Y"] = ((canvas["Aerial Outlet"] + canvas["Box Threat"]) - (canvas["Defensive Work"] + canvas["Wide Combination"])) / 2
    return canvas


def scout_canvas_figure(context: pd.DataFrame, focus: pd.DataFrame, title: str, benchmark_name: str | None = None) -> go.Figure:
    """Modern role-family map with the selected profiles pulled above the background pool."""
    background = scout_canvas_coordinates(context)
    if len(background) > 1500:
        background = background.sample(1500, random_state=17)
    foreground = scout_canvas_coordinates(focus)
    fig = go.Figure()
    if not background.empty:
        fig.add_trace(go.Scatter(
            x=background["Canvas X"], y=background["Canvas Y"], mode="markers", name="Player universe",
            marker=dict(size=5, color="#C9CDD3", opacity=.28),
            customdata=np.stack([background["Name"], background["Club"], background["Best Role"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<extra></extra>",
        ))
    colors = [GOLD, NAVY, "#617188", "#7C3AED", "#A66B00", "#00897B"]
    for index, (_, row) in enumerate(foreground.iterrows()):
        is_benchmark = str(row.get("Name", "")) == str(benchmark_name or "")
        label = str(row.get("Name", "Player"))
        similarity = row.get("Similarity Score", np.nan)
        extra = f"<br>Similarity: {float(similarity):.1f}%" if pd.notna(similarity) else ""
        fig.add_trace(go.Scatter(
            x=[row["Canvas X"]], y=[row["Canvas Y"]], mode="markers+text",
            name=("Benchmark · " if is_benchmark else "") + label,
            text=[label], textposition="top center",
            marker=dict(
                size=20 if is_benchmark else 15,
                color=GOLD if is_benchmark else colors[index % len(colors)],
                symbol="star" if is_benchmark else "circle",
                line=dict(color=WHITE, width=2.5),
            ),
            hovertemplate=(f"<b>{html.escape(label)}</b><br>{html.escape(str(row.get('Club', '')))} · {html.escape(str(row.get('Best Role', '')))}"
                           f"<br>ST fit: {float(row.get('Sunderland AFC ST Fit', 0)):.1f}{extra}<extra></extra>"),
        ))

    all_x = pd.concat([background["Canvas X"], foreground["Canvas X"]], ignore_index=True) if not foreground.empty else background["Canvas X"]
    all_y = pd.concat([background["Canvas Y"], foreground["Canvas Y"]], ignore_index=True) if not foreground.empty else background["Canvas Y"]
    span_x = max(float(all_x.max() - all_x.min()) if not all_x.empty else 20.0, 20.0)
    span_y = max(float(all_y.max() - all_y.min()) if not all_y.empty else 20.0, 20.0)
    x_mid = float((all_x.max() + all_x.min()) / 2) if not all_x.empty else 0.0
    y_mid = float((all_y.max() + all_y.min()) / 2) if not all_y.empty else 0.0
    x_range = [x_mid - span_x * .66, x_mid + span_x * .66]
    y_range = [y_mid - span_y * .70, y_mid + span_y * .70]
    fig.add_vline(x=0, line_color="#D7DCE3", line_width=1)
    fig.add_hline(y=0, line_color="#D7DCE3", line_width=1)
    fig.add_annotation(x=x_range[0] + span_x * .03, y=y_range[1] - span_y * .05, text="Physical box presence", showarrow=False, font=dict(size=11, color=MUTED), xanchor="left")
    fig.add_annotation(x=x_range[1] - span_x * .03, y=y_range[0] + span_y * .05, text="Pressing / mobile support", showarrow=False, font=dict(size=11, color=MUTED), xanchor="right")
    fig.update_xaxes(title="Connector & combination  ←   →  Vertical box runner", range=x_range, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_yaxes(title="Role family", range=y_range, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_layout(title=title, legend=dict(orientation="h", y=-.22, x=.5, xanchor="center"), hovermode="closest")
    return fig_layout(fig, 535)


def profile_strip_chart(rows: pd.DataFrame, title: str) -> go.Figure:
    """A compact visual fingerprint. Exact values remain in hover instead of cluttering the page."""
    if rows.empty:
        return go.Figure()
    values = np.asarray([profile_phase_values(row) for _, row in rows.iterrows()], dtype=float)
    labels = [str(name) for name in rows["Name"].tolist()]
    fig = go.Figure(go.Heatmap(
        z=values,
        x=PHASES,
        y=labels,
        text=[["" for _ in PHASES] for _ in labels],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=[[0.0, "#D8DDE3"], [0.48, "#A5AEBB"], [0.67, "#677487"], [0.82, GOLD], [1.0, "#9C0009"]],
        zmin=0,
        zmax=100,
        showscale=False,
        customdata=values,
        hovertemplate="<b>%{y}</b><br>%{x}: %{customdata:.1f}/100<extra></extra>",
        xgap=7,
        ygap=7,
    ))
    fig.update_layout(title=title)
    fig.update_xaxes(side="top", tickfont=dict(size=11), fixedrange=True)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12), fixedrange=True)
    return fig_layout(fig, max(240, 105 + 47 * len(rows)), bottom_legend=False)


def similarity_alignment_chart(benchmark: pd.Series, candidates: pd.DataFrame) -> go.Figure:
    """Show the phase deltas that explain a match, without pretending the score is a raw average."""
    if candidates.empty:
        return go.Figure()
    benchmark_values = np.asarray(profile_phase_values(benchmark), dtype=float)
    deltas = np.asarray([np.asarray(profile_phase_values(row), dtype=float) - benchmark_values for _, row in candidates.iterrows()])
    labels = [str(name) for name in candidates["Name"].tolist()]
    fig = go.Figure(go.Heatmap(
        z=deltas,
        x=PHASES,
        y=labels,
        colorscale=[[0.0, "#31405A"], [0.42, "#BFC7D2"], [0.5, "#F6F8FB"], [0.58, "#F2B4B8"], [1.0, GOLD]],
        zmid=0,
        zmin=-30,
        zmax=30,
        text=[[f"{value:+.0f}" for value in row] for row in deltas],
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=False,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.1f} vs benchmark<extra></extra>",
        xgap=7,
        ygap=7,
    ))
    fig.update_layout(title="Where each match differs from the benchmark")
    fig.update_xaxes(side="top", tickfont=dict(size=11), fixedrange=True)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12), fixedrange=True)
    return fig_layout(fig, max(250, 110 + 47 * len(candidates)), bottom_legend=False)


def profile_reading(rows: pd.DataFrame) -> str:
    """One direct decision cue from the selected profiles."""
    if rows.empty:
        return ""
    strengths = {phase: float(pd.to_numeric(rows[phase], errors="coerce").mean()) for phase in PHASES}
    spread = {phase: float(pd.to_numeric(rows[phase], errors="coerce").max() - pd.to_numeric(rows[phase], errors="coerce").min()) for phase in PHASES}
    strongest = max(strengths, key=strengths.get)
    clearest_split = max(spread, key=spread.get)
    return (
        f"<div class='scout-reading'><div class='eyebrow'>Scout canvas · reading</div>"
        f"Shared strength: <strong>{html.escape(strongest)}</strong><br>"
        f"Biggest difference: <strong>{html.escape(clearest_split)}</strong>"
        f"</div>"
    )


def similarity_trait_reading(benchmark: pd.Series, candidate: pd.Series) -> str:
    benchmark_values = np.asarray(profile_phase_values(benchmark), dtype=float)
    candidate_values = np.asarray(profile_phase_values(candidate), dtype=float)
    deltas = candidate_values - benchmark_values
    closest = PHASES[int(np.argmin(np.abs(deltas)))]
    tradeoff = PHASES[int(np.argmax(np.abs(deltas)))]
    direction = "more" if deltas[int(np.argmax(np.abs(deltas)))] > 0 else "less"
    return f"Closest on <b>{html.escape(closest)}</b> · main trade-off: {direction} <b>{html.escape(tradeoff)}</b>"


def display_metric(label: str, value: object, suffix: str = ""):
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}{suffix}</div></div>""",
        unsafe_allow_html=True,
    )


def export_safe(data: pd.DataFrame) -> pd.DataFrame:
    banned = [c for c in data.columns if "url" in c.lower() or "link" in c.lower()]
    cols = [
        "Name", "Club", "Age", "Positions",
        "Scouting Opportunity", "Sunderland AFC ST Fit", "Hidden Gem Opportunity", "Under-the-Radar Score", "Best System", "Best System Fit", "Best Role", "Scout Tag",
        "Box Threat", "Link-up Play", "Channel Running", "Wide Combination", "Aerial Outlet", "Defensive Work",
    ]
    if "Country" in data.columns:
        cols.insert(4, "Country")
        cols.insert(5, "Continent")
    keep = [c for c in cols if c in data.columns and c not in banned]
    return data[keep].round(2)


def system_fit_chart(row: pd.Series) -> go.Figure:
    system_df = pd.DataFrame({
        "System": ["4-2-3-1", "3-4-3", "4-3-3"],
        "Fit": [row["4-2-3-1 Fit"], row["3-4-3 Fit"], row["4-3-3 Fit"]]
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
        "For Sunderland AFC, the ideal striker is not a poacher-only profile. The recruitment model rewards a forward "
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
        mime = "image/svg+xml" if LOGO_PATH.suffix.lower() == ".svg" else "image/png"
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


def build_scouting_report_html(row: pd.Series) -> str:
    summary, role_fit, risk = qualitative_summary(row)
    logo_src = logo_as_base64()
    image_html = ""
    portrait_url = player_image_url(row.get("image_url", ""))
    if portrait_url:
        image_html = f'<img class="player-img" src="{html.escape(portrait_url)}" />'
    elif logo_src:
        image_html = f'<img class="player-img" src="{logo_src}" />'

    radar_html = radar_chart(row, "Radar overview").to_html(full_html=False, include_plotlyjs="cdn")
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
<title>Sunderland AFC Scouting Report - {html.escape(str(row['Name']))}</title>
<style>
body {{font-family: Arial, sans-serif; margin: 0; background:#F6F8FB; color:#111111;}}
.wrapper {{max-width: 1180px; margin: 0 auto; padding: 34px;}}
.header {{display:flex; align-items:center; gap:18px; background:#111111; color:white; padding:22px 26px; border-radius:24px; border-bottom:6px solid #E30613;}}
.logo {{width:76px; height:76px; object-fit:contain; background:white; border-radius:16px; padding:6px;}}
.title h1 {{margin:0; font-size:30px;}}
.title p {{margin:6px 0 0; color:#F6D6D8;}}
.card {{background:white; border:1px solid #E5E7EB; border-radius:22px; padding:22px; box-shadow:0 12px 30px rgba(11,31,58,.08); margin-top:18px;}}
.hero {{display:grid; grid-template-columns:160px 1fr 360px; gap:20px; align-items:center;}}
.player-img {{width:140px; height:140px; object-fit:contain; border-radius:20px; background:#EEF2F7;}}
.kpi-grid {{display:grid; grid-template-columns:repeat(4,1fr); gap:12px;}}
.kpi {{border-top:5px solid #E30613; background:#FFFFFF; padding:14px; border-radius:18px; box-shadow:0 8px 18px rgba(11,31,58,.08);}}
.kpi small {{display:block; color:#6B7280; text-transform:uppercase; font-weight:800; letter-spacing:.06em;}}
.kpi b {{font-size:25px; color:#111111;}}
.badge {{display:inline-block; border:1px solid rgba(227,6,19,.55); background:rgba(227,6,19,.15); padding:6px 10px; border-radius:999px; font-size:12px; font-weight:800; margin:4px 4px 0 0;}}
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
      <h1>Sunderland AFC — Striker Scouting Report</h1>
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
      <div class="kpi"><small>ST Fit</small><b>{float(row['Sunderland AFC ST Fit']):.1f}</b></div>
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
    <h3 class="section-title">Sunderland AFC DNA context</h3>
    <p>The Sunderland recruitment model is currently a provisional hypothesis informed by the public-data baseline. It tests box threat, link-up play, channel running, wide combination, aerial outlet and defensive work, but should be validated with video and authorised event data.</p>
    <p>The report does not rank players by name value. It tests whether the player gives the specific forward behaviours that Sunderland AFC require, while the tool's hidden-gem logic reduces the priority of obvious superstar or elite-market names.</p>
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

  <div class="footer">Confidential scouting prototype — Sunderland AFC recruitment use only. This report is a first-layer recruitment filter and must be supported by video scouting and contextual review.</div>
</div>
</body>
</html>
"""
    return report


def safe_report_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")
    return f"sunderland_afc_scouting_report_{clean or 'player'}.html"



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
body {{font-family: Arial, sans-serif; margin:0; background:#F6F8FB; color:#111111;}}
.wrapper {{max-width:1180px; margin:0 auto; padding:34px;}}
.header {{display:flex; align-items:center; gap:18px; background:#111111; color:white; padding:22px 26px; border-radius:24px; border-bottom:6px solid #E30613;}}
.logo {{width:76px; height:76px; object-fit:contain; background:white; border-radius:16px; padding:6px;}}
.title h1 {{margin:0; font-size:30px;}}
.title p {{margin:6px 0 0; color:#F6D6D8;}}
.card {{background:white; border:1px solid #E5E7EB; border-radius:22px; padding:22px; box-shadow:0 12px 30px rgba(11,31,58,.08); margin-top:18px;}}
.section-title {{font-size:21px; font-weight:900; margin:0 0 10px; color:#111111;}}
.text-block p {{line-height:1.62; font-size:15px;}}
.table {{width:100%; border-collapse:collapse; margin-top:8px; font-size:13px;}}
.table th, .table td {{border-bottom:1px solid #E5E7EB; padding:9px; text-align:left;}}
.table th {{background:#EEF2F7; color:#111111;}}
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
    filename: str = "sunderland_afc_page_report.html",
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
    mask = data["Club"].astype(str).str.contains("sunderland", case=False, na=False)
    current = data[mask].copy()
    if current.empty and "Name" in data.columns:
        current_names = {"brian brobbey", "wilson isidor", "eliezer mayenda", "timur tutierov"}
        current = data[data["Name"].apply(normalise_text).isin(current_names)].copy()
    if current.empty:
        return current
    return current.sort_values(["Sunderland AFC ST Fit", "Scouting Opportunity"], ascending=False)


def similarity_feature_columns(data: pd.DataFrame) -> list[str]:
    candidate_cols = PHASES + [
        "Sunderland AFC ST Fit", "Best System Fit", "4-2-3-1 Fit", "3-4-3 Fit", "4-3-3 Fit",
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
    if "Sunderland AFC ST Fit" in pool.columns:
        pool["ST Fit Gap vs Benchmark"] = (pool["Sunderland AFC ST Fit"] - float(benchmark_row.get("Sunderland AFC ST Fit", 0))).abs()
    else:
        pool["ST Fit Gap vs Benchmark"] = np.nan
    return pool.sort_values(["Similarity Score", "Scouting Opportunity"], ascending=False).head(top_n)

# -----------------------------
# Pages
# -----------------------------
if page == "Evidence Overview":
    market_forwards, sunderland_forwards = make_forward_evidence(understat_players)
    if market_forwards.empty or sunderland_forwards.empty:
        st.warning("The forward-performance view is not available right now. Refresh the app and try again.")
        st.stop()

    st.markdown(
        """
        <div class="dna-hero">
            <div class="dna-kicker">Sunderland AFC · Recruitment evidence</div>
            <div class="dna-hero-title">Start with the player output we can prove.</div>
            <div class="dna-hero-copy">This tool now separates real performance evidence from tactical judgement. Use it to frame the current forward group against the Premier League before adding video, medical, contract and scouting context.</div>
            <div class="dna-strip"><span>Current Sunderland forwards</span><span>Premier League benchmark</span><span>Per-90 output</span><span>No estimated fit score</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1: display_metric("Forwards benchmarked", f"{len(market_forwards):,}")
    with c2: display_metric("Sunderland forward group", str(len(sunderland_forwards)))
    with c3: display_metric("Evidence dimensions", "4")

    executive_view = market_forwards[market_forwards["Minutes"] >= 300].copy()
    executive_view["Sunderland player"] = executive_view["Club"].astype(str).str.contains("sunderland", case=False, na=False)
    exec_fig = px.scatter(
        executive_view, x="xG Chain per 90", y="xG per 90 live", size="Minutes", color="Sunderland player",
        color_discrete_map={True: GOLD, False: "#AEB6C2"}, hover_name="Name",
        hover_data={"Club": True, "NPG per 90": ":.2f", "xA per 90 live": ":.2f", "Shots per 90": ":.2f", "Minutes": ":.0f"}, size_max=44,
    )
    exec_fig.update_layout(
        title="Sunderland forwards in the Premier League forward landscape",
        xaxis=dict(title="Attack involvement · xG Chain per 90", gridcolor="#EEF0F3"),
        yaxis=dict(title="Expected scoring · non-penalty xG per 90", gridcolor="#EEF0F3"),
        legend=dict(title=""),
    )
    st.plotly_chart(fig_layout(exec_fig, 600), width="stretch")
    st.markdown('<div class="visual-caption">Red bubbles are Sunderland players. The plot reveals how much a forward is involved in attacking sequences and how often he gets into expected-scoring positions; hover gives the detailed metrics.</div>', unsafe_allow_html=True)
    page_report_download(
        "Download recruitment evidence overview",
        "Sunderland AFC — Recruitment Evidence Overview",
        "Player output benchmarked against Premier League forwards",
        [
            "This overview intentionally does not produce a transfer recommendation or a reconstructed fit score.",
            "It establishes the observable forward baseline: scoring threat, creation, shot volume and involvement in attacking sequences.",
            "Tactical fit must be added through video and richer authorised data, not inferred from missing event metrics.",
        ],
        figures=[("Premier League forward landscape", exec_fig)],
        tables=[("Sunderland forward output", sunderland_forwards[["Name", "Position", "Minutes", "Goals", "Assists", "xG", "xA", "Shots", "Key Passes", "xG Chain", "xG Buildup"]])],
        filename="sunderland_afc_recruitment_evidence_overview.html",
        key="evidence_overview_report_download",
    )

elif page == "Executive Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1: display_metric("Players screened", f"{len(players):,}")
    with c2: display_metric("Current filtered pool", f"{len(df):,}")
    with c3:
        top_fit = df["Sunderland AFC ST Fit"].max() if not df.empty else 0
        display_metric("Best ST Fit", f"{top_fit:.1f}")
    with c4:
        top_hidden = df["Hidden Gem Opportunity"].max() if not df.empty else 0
        display_metric("Best Hidden Gem", f"{top_hidden:.1f}")

    st.markdown('<div class="section-header">Recruitment logic</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="navy-card">
        <b>Tool objective:</b> identify under-the-radar strikers who match Sunderland AFC's tactical striker profile, not simply the highest generic scorer.<br><br>
        <b>Ideal target:</b> a striker who can act as a Complete Forward / Deep-Lying Forward hybrid: attack the box, link with the No.10 and wide players, run channels, and still support the team's mid-block defensive work.<br><br>
        <b>Main decision score:</b> Scouting Opportunity = Sunderland AFC fit + hidden-gem value + system fit + age upside, with obvious superstar/elite-club players suppressed by default.
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
                render_player_image(r, width=120)
            with cols[1]:
                st.markdown(
                    f"""<div class="card" style="padding:12px 14px;margin-bottom:8px;">
                    <b style="color:{NAVY};font-size:16px;">{r['Name']}</b><br>
                    <span class="small-note">{r['Club']} · Age {int(r['Age']) if pd.notna(r['Age']) else '-'} · {r['Best Role']}</span><br>
                    <span class="tag">ST Fit {r['Sunderland AFC ST Fit']:.1f}</span>
                    <span class="tag">Hidden {r['Hidden Gem Opportunity']:.1f}</span>
                    <span class="tag">{r['Scout Tag']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

    overview_pool = df if not df.empty else players.sort_values("Scouting Opportunity", ascending=False)
    overview_cols = ["Name","Club","Age","Country","Continent","Best Role","Scout Tag","Scouting Opportunity","Sunderland AFC ST Fit","Hidden Gem Opportunity","Best System"]
    page_report_download(
        "Download overview report HTML",
        "Sunderland AFC — Executive Scouting Overview",
        "Recruitment overview aligned to the club striker DNA",
        [
            "This overview summarises the filtered striker market through Sunderland AFC's tactical DNA rather than simple generic striker ratings.",
            "The main logic prioritises players who combine box threat, link-up play, channel running and wide combinations, while reducing obvious elite-market targets by default.",
            "Use this page as the first meeting layer: it explains the model, the filtered pool and the strongest realistic hidden-gem recommendations before moving into individual player reports.",
        ],
        figures=[("Hidden-gem map", hidden_gem_scatter(overview_pool))],
        tables=[("Top recommended players", report_table(overview_pool, overview_cols, 15))],
        filename="sunderland_afc_executive_overview_report.html",
        key="overview_report_download",
    )


elif page == "Current ST Assessment":
    market_forwards, sunderland_forwards = make_forward_evidence(understat_players)
    if market_forwards.empty or sunderland_forwards.empty:
        st.warning("The current forward view is not available right now. Refresh the app and try again.")
        st.stop()

    st.markdown(
        f"""
        <div class="dna-hero">
            <div class="dna-kicker">Sunderland AFC · Forward group</div>
            <div class="dna-hero-title">Actual player output, not estimated roles.</div>
            <div class="dna-hero-copy">This is Sunderland's active forward group for the current season. Every profile is relative to Premier League forwards with at least 180 minutes. Dimensions that are not available in the current view are deliberately excluded.</div>
            <div class="dna-strip"><span>Finishing</span><span>Shot volume</span><span>Creation</span><span>Attack involvement</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, profile_tab, market_tab = st.tabs(["Forward group", "Player profile", "League context"])
    profile_dimensions = ["Finishing", "Shot volume", "Creation", "Attack involvement"]

    with overview_tab:
        group_radar = go.Figure()
        for index, (_, player) in enumerate(sunderland_forwards.sort_values("Minutes", ascending=False).iterrows()):
            vals = [float(player[col]) for col in profile_dimensions]
            color = GOLD if index == 0 else [NAVY, "#6B7280", "#A9000B", "#9CA3AF", "#4B5563", "#D1D5DB"][min(index - 1, 5)]
            group_radar.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=profile_dimensions + profile_dimensions[:1], mode="lines",
                name=str(player["Name"]), line=dict(color=color, width=3 if index == 0 else 2),
                hovertemplate=f"<b>{player['Name']}</b><br>%{{theta}}: %{{r:.0f}}th percentile<extra></extra>",
            ))
        group_radar.update_layout(
            title="Sunderland forward group: four evidence-backed dimensions",
            polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], showticklabels=False, ticks="", gridcolor="#E5E7EB"), angularaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=13, color=NAVY))),
        )
        st.plotly_chart(fig_layout(group_radar, 570), width="stretch")
        st.markdown('<div class="visual-caption">Each line is a Sunderland forward. The profile reflects real season output per 90, benchmarked against the available Premier League forward pool — it is not a tactical-role score.</div>', unsafe_allow_html=True)

    with profile_tab:
        selected_name = st.selectbox("Choose a Sunderland forward", sunderland_forwards.sort_values("Minutes", ascending=False)["Name"].astype(str).tolist(), key="understat_forward_select")
        selected = sunderland_forwards[sunderland_forwards["Name"].astype(str) == selected_name].iloc[0]
        left, right = st.columns([.9, 1.1], gap="large")
        with left:
            selected_radar = go.Figure(go.Scatterpolar(
                r=[float(selected[col]) for col in profile_dimensions] + [float(selected[profile_dimensions[0]])],
                theta=profile_dimensions + profile_dimensions[:1], mode="lines+markers",
                line=dict(color=GOLD, width=3), marker=dict(color=GOLD, size=7),
                hovertemplate="<b>%{theta}</b><br>Premier League forward percentile: %{r:.0f}<extra></extra>",
            ))
            selected_radar.update_layout(
                title=f"{selected_name}: performance shape",
                polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], showticklabels=False, ticks="", gridcolor="#E5E7EB"), angularaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=13, color=NAVY))),
                showlegend=False,
            )
            st.plotly_chart(fig_layout(selected_radar, 490, bottom_legend=False), width="stretch")
        with right:
            selected_bars = go.Figure(go.Bar(
                x=selected[profile_dimensions].astype(float).values,
                y=profile_dimensions, orientation="h",
                marker_color=[GOLD if value >= 55 else "#8B8B8B" for value in selected[profile_dimensions].astype(float).values],
                text=[relative_band(float(value)).replace(" profile", "") for value in selected[profile_dimensions]], textposition="inside",
                textfont=dict(color=WHITE, size=12),
                hovertemplate="<b>%{y}</b><br>Premier League forward percentile: %{x:.0f}<extra></extra>",
            ))
            selected_bars.update_layout(title="Visual reading", xaxis=dict(range=[0, 100], visible=False), yaxis=dict(autorange="reversed", tickfont=dict(size=13, color=NAVY)), showlegend=False)
            st.plotly_chart(fig_layout(selected_bars, 490, bottom_legend=False), width="stretch")
        st.markdown('<div class="visual-caption">Hover for the exact underlying percentile. The dashboard intentionally avoids turning raw data into unsupported labels such as “pressing forward” or “target man”.</div>', unsafe_allow_html=True)

    with market_tab:
        market_view = market_forwards[market_forwards["Minutes"] >= 300].copy()
        market_view["Sunderland player"] = market_view["Club"].astype(str).str.contains("sunderland", case=False, na=False)
        market_fig = px.scatter(
            market_view, x="xG Chain per 90", y="xG per 90 live", size="Minutes", color="Sunderland player",
            color_discrete_map={True: GOLD, False: "#AEB6C2"}, hover_name="Name",
            hover_data={"Club": True, "NPG per 90": ":.2f", "xA per 90 live": ":.2f", "Shots per 90": ":.2f", "Minutes": ":.0f"}, size_max=42,
        )
        market_fig.update_layout(
            title="League context: involvement versus expected scoring",
            xaxis=dict(title="Attack involvement · xG Chain per 90", gridcolor="#EEF0F3"),
            yaxis=dict(title="Expected scoring · non-penalty xG per 90", gridcolor="#EEF0F3"),
            legend=dict(title=""),
        )
        st.plotly_chart(fig_layout(market_fig, 560), width="stretch")
        st.markdown('<div class="visual-caption">Red bubbles are Sunderland forwards; every other bubble is an available Premier League forward. This makes the internal baseline and possible external profiles visible without a fabricated fit score.</div>', unsafe_allow_html=True)

    report_cols = ["Name", "Club", "Position", "Minutes", "Goals", "Assists", "xG", "xA", "Shots", "Key Passes", "Non-penalty Goals", "Non-penalty xG", "xG Chain", "xG Buildup"]
    page_report_download(
        "Download forward evidence report HTML",
        "Sunderland AFC — Forward Evidence Profile",
        "Observed player output benchmarked against Premier League forwards",
        [
            "The visual profile uses finishing, shot-volume, creation and attack-involvement metrics.",
            "Percentiles contextualise output against Premier League forwards with at least 180 minutes; they are not tactical role or transfer-fit scores.",
            "Unavailable metrics — including dribbles, aerial duels, box touches and pressing actions — are excluded rather than estimated.",
        ],
        figures=[("Forward-group profiles", group_radar), ("Selected player profile", selected_radar), ("League context", market_fig)],
        tables=[("Sunderland forward output", sunderland_forwards[report_cols])],
        filename="sunderland_afc_forward_evidence_report.html",
        key="understat_forward_report_download",
    )
    st.stop()

    current = current_squad_strikers(players)
    if current.empty:
        st.warning("No current Sunderland AFC strikers were detected in the CSV. Run the Sunderland AFC ST scraper first, then refresh the app.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: display_metric("Current STs", f"{len(current)}")
        with c2: display_metric("Avg age", f"{current['Age'].mean():.1f}" if current['Age'].notna().any() else "-")
        with c3: display_metric("Best ST Fit", f"{current['Sunderland AFC ST Fit'].max():.1f}")
        with c4: display_metric("Best System Fit", f"{current['Best System Fit'].max():.1f}")

        st.markdown(
            f"""
            <div class="navy-card">
            <b>Interpretation:</b> this page turns the current Sunderland AFC striker group into a benchmark layer. The idea is to understand the present striker room first — who gives box threat, who connects play, who works as an aerial outlet, and who is closest to the ideal 4-2-3-1-led striker profile. The hidden-gem page should then be read against this current internal benchmark.
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_cols = ["Name","Club","Age","Country","Continent","Best Role","Best System","Sunderland AFC ST Fit","Hidden Gem Opportunity","Scouting Opportunity","Box Threat","Link-up Play","Channel Running","Wide Combination","Aerial Outlet","Defensive Work"]
        st.dataframe(report_table(current, current_cols, 20), width="stretch", hide_index=True)

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(multi_radar(current.head(5), "Current Sunderland AFC ST phase profiles"), width="stretch")
        with right:
            st.plotly_chart(difference_bars(current.head(5)), width="stretch")

        selected_current = st.selectbox("Inspect current striker", current["Name"].astype(str).tolist(), index=0, key="current_st_select")
        cur_row = current[current["Name"].astype(str) == selected_current].iloc[0]
        s1, s2 = st.columns([0.38, 0.62], vertical_alignment="center")
        with s1:
            render_player_image(cur_row, width=132)
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
            "Sunderland AFC — Current ST Assessment",
            "Internal striker benchmark before external recruitment",
            [
                "This report assesses the current Sunderland AFC striker group as an internal benchmark layer before moving to hidden-gem recruitment.",
                "The purpose is not to label current players as targets. It is to understand the reference profile already inside the squad: box threat, link-up play, channel running, wide combination, aerial outlet and defensive work.",
                "The external shortlist should be judged against this benchmark: does a candidate replace, complement or upgrade the current striker profile?",
            ],
            figures=[("Current ST radar comparison", multi_radar(current.head(5), "Current ST radar comparison")), ("Current ST phase differences", difference_bars(current.head(5)))],
            tables=[("Current striker assessment table", report_table(current, current_cols, 20))],
            filename="sunderland_afc_current_st_assessment_report.html",
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
            render_player_image(selected_row, width=128)
        with c_profile:
            st.markdown(f"### {selected_row['Name']}")
            st.markdown(f"**Club:** {selected_row['Club']}  ")
            st.markdown(f"**Age:** {int(selected_row['Age']) if pd.notna(selected_row['Age']) else '-'}  ")
            st.markdown(f"<span class='tag'>{selected_row['Scout Tag']}</span><span class='tag'>Opportunity {selected_row['Scouting Opportunity']:.1f}</span><span class='tag'>ST Fit {selected_row['Sunderland AFC ST Fit']:.1f}</span>", unsafe_allow_html=True)
        with c_radar:
            st.plotly_chart(radar_chart(selected_row, "Selected hidden-gem phase profile"), width="stretch")

    view = export_safe(df).head(500)
    st.dataframe(view, width="stretch", height=560, hide_index=True)
    hidden_report_pool = df if not df.empty else players.sort_values("Scouting Opportunity", ascending=False)
    hidden_cols = ["Name","Club","Age","Country","Continent","Best Role","Scout Tag","Scouting Opportunity","Sunderland AFC ST Fit","Hidden Gem Opportunity","Under-the-Radar Score","Best System"]
    page_report_download(
        "Download hidden gems report HTML",
        "Sunderland AFC — Hidden Gems Shortlist",
        "Filtered shortlist report with qualitative scouting context",
        [
            "This shortlist is ranked by Scouting Opportunity, which combines Sunderland AFC tactical fit, hidden-gem value, system fit and age upside.",
            "The page is designed as a first scouting filter: it does not make a final transfer decision, but highlights which players deserve video review and contextual validation.",
            "Obvious superstar or elite-club names are suppressed by default so the list stays useful for realistic recruitment discovery.",
        ],
        figures=[("Hidden-gem opportunity map", hidden_gem_scatter(hidden_report_pool))],
        tables=[("Top hidden-gem candidates", report_table(hidden_report_pool, hidden_cols, 50))],
        filename="sunderland_afc_hidden_gems_report.html",
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
        render_player_image(row, width=132)
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
        with m1: display_metric("ST Fit", f"{row['Sunderland AFC ST Fit']:.1f}")
        with m2: display_metric("Hidden Gem", f"{row['Hidden Gem Opportunity']:.1f}")
        m3, m4 = st.columns(2)
        with m3: display_metric("Opportunity", f"{row['Scouting Opportunity']:.1f}")
        with m4: display_metric("System", f"{row['Best System Fit']:.1f}")

    st.markdown("---")
    # Streamlit's inline HTML sanitizer can remove SVG polygons. The report
    # visual is deliberately isolated so the radar always renders unchanged.
    components.html(player_report_visual(row), height=920, scrolling=False)

    r3, r4 = st.columns([1, 1])
    with r3:
        st.plotly_chart(role_probability_chart(row), width="stretch")
    with r4:
        system_df = pd.DataFrame({
            "System": ["4-2-3-1", "3-4-3", "4-3-3"],
            "Fit": [row["4-2-3-1 Fit"], row["3-4-3 Fit"], row["4-3-3 Fit"]]
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
    st.caption("Compare role family, phase profile and the key difference between players.")
    pool = df if not df.empty else players.sort_values("Scouting Opportunity", ascending=False)
    default_names = pool.head(3)["Name"].tolist()
    selected_names = st.multiselect("Choose 2 to 5 players", pool["Name"].astype(str).tolist(), default=default_names, max_selections=5)
    if len(selected_names) < 2:
        st.warning("Choose at least two players to compare.")
    else:
        comp = pool[pool["Name"].isin(selected_names)].drop_duplicates("Name").head(5)
        st.plotly_chart(scout_canvas_figure(pool, comp, "Scout canvas · selected role families"), width="stretch")
        st.markdown('<div class="visual-caption">Grey points are the available pool. Selected players are highlighted. Hover a point for player details.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([0.68, 0.32], vertical_alignment="top")
        with c1:
            st.plotly_chart(profile_strip_chart(comp, "Selected profile fingerprints"), width="stretch")
        with c2:
            st.markdown(profile_reading(comp), unsafe_allow_html=True)
            st.markdown('<div class="small-note">Each tile uses the same 0–100 phase scale. Hover for the exact value.</div>', unsafe_allow_html=True)

        with st.expander("Open detailed attributes and exact comparison data"):
            if comp.apply(has_full_efem_attributes, axis=1).all():
                comparison_lens = st.selectbox("Attribute group", list(EFEM_ATTRIBUTE_GROUPS), key="comparison_attribute_lens")
                comparison_rows = []
                for _, candidate in comp.iterrows():
                    for attribute in EFEM_ATTRIBUTE_GROUPS[comparison_lens]:
                        comparison_rows.append({"Player": candidate["Name"], "Attribute": attribute, "Percentile": float(candidate[f"EFEM percentile · {attribute}"]), "Raw value": float(candidate[attribute])})
                comparison_detail = px.bar(
                    pd.DataFrame(comparison_rows), x="Percentile", y="Attribute", color="Player", barmode="group", orientation="h",
                    color_discrete_sequence=[GOLD, NAVY, BLUE, RED, "#7C3AED"],
                    hover_data={"Percentile": ":.0f", "Raw value": ":.0f"},
                )
                comparison_detail.add_vline(x=50, line_color="#AEB6C2", line_width=1.5, line_dash="dot")
                comparison_detail.update_layout(title=f"{comparison_lens} · attributes", xaxis=dict(range=[0, 105], gridcolor="#E5E7EB", title="Percentile across all strikers"), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_layout(comparison_detail, 470), width="stretch")
            else:
                st.info("Detailed attributes are not available for every selected player, so the comparison above uses the shared phase profile.")
            comp_cols = ["Name","Club","Age","Best Role","Scout Tag","Scouting Opportunity","Sunderland AFC ST Fit","Hidden Gem Opportunity","Under-the-Radar Score","Best System","Box Threat","Link-up Play","Channel Running","Wide Combination","Aerial Outlet","Defensive Work"]
            st.dataframe(comp[[c for c in comp_cols if c in comp.columns]].round(2), width="stretch", hide_index=True)

        page_report_download(
            "Download comparison report HTML",
            "Sunderland AFC — Player Comparison Report",
            "Side-by-side striker comparison aligned to Sunderland AFC DNA",
            [
                "This report compares selected strikers across the six Sunderland AFC striker phases and the main recruitment scores.",
                "The aim is to show profile differences clearly: who offers more box threat, who links play better, who attacks channels, and who fits the tactical system more naturally.",
                "This comparison should be used to support discussion, not as a final transfer decision without video and contextual scouting.",
            ],
            figures=[("Scout canvas", scout_canvas_figure(pool, comp, "Scout canvas · selected role families")), ("Profile fingerprints", profile_strip_chart(comp, "Selected profile fingerprints"))],
            tables=[("Comparison table", report_table(comp, comp_cols, 10))],
            filename="sunderland_afc_player_comparison_report.html",
            key="compare_report_download",
        )

elif page == "Sunderland AFC DNA":
    if understat_dna.empty:
        st.warning("The tactical DNA view is not available right now. Refresh the app and try again.")
        st.stop()

    visual_profile = make_dna_visual_profile(understat_dna, passmap_teams)
    season_label = f"{understat_status.get('season_start_year', '2025')}/{str(understat_status.get('season_start_year', '2025'))[-2:]}"
    st.markdown(
        f"""
        <div class="dna-hero">
            <div class="dna-kicker">Sunderland AFC · Tactical DNA</div>
            <div class="dna-hero-title">A visual read of how the team plays.</div>
            <div class="dna-hero-copy">Every visual is benchmarked against the available Premier League snapshot. No projected data and no empty pillars: only dimensions supported by the current performance view.</div>
            <div class="dna-strip"><span>Season {season_label}</span><span>League-relative profile</span><span>Current benchmark</span><span>Hover any visual for detail</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    identity_tab, match_tab, formation_tab = st.tabs(["Team identity", "Match rhythm", "Formation lens"])

    with identity_tab:
        left, right = st.columns([1.03, 0.97], gap="large")
        with left:
            radar_labels = visual_profile["Dimension"].tolist()
            radar_values = visual_profile["Percentile"].tolist()
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=radar_values + radar_values[:1],
                theta=radar_labels + radar_labels[:1],
                mode="lines+markers",
                line=dict(color=GOLD, width=3), marker=dict(color=GOLD, size=7),
                hovertemplate="<b>%{theta}</b><br>League percentile: %{r:.0f}<extra></extra>",
                name="Sunderland",
            ))
            radar_fig.update_layout(
                title="The Sunderland identity wheel",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0, 100], showticklabels=False, ticks="", gridcolor="#E5E7EB", linecolor="#E5E7EB"),
                    angularaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", tickfont=dict(size=13, color=NAVY)),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig_layout(radar_fig, 520, bottom_legend=False), width="stretch")
        with right:
            bars = go.Figure()
            for _, item in visual_profile.iloc[::-1].iterrows():
                bars.add_trace(go.Bar(
                    x=[item["Percentile"]], y=[item["Dimension"]], orientation="h",
                    marker_color=GOLD if item["Percentile"] >= 55 else "#8B8B8B",
                    text=[item["Band"]], textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color=WHITE, size=12),
                    hovertemplate=f"<b>{item['Dimension']}</b><br>League percentile: {item['Percentile']:.0f}<br>{item['Band']}<extra></extra>",
                    showlegend=False,
                ))
            bars.update_layout(
                title="What stands out at a glance",
                barmode="overlay",
                xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
                yaxis=dict(title="", tickfont=dict(size=13, color=NAVY), fixedrange=True),
                bargap=.38,
            )
            st.plotly_chart(fig_layout(bars, 520, bottom_legend=False), width="stretch")
        st.markdown('<div class="visual-caption">Red marks dimensions that sit above the league’s middle band. The wheel gives the complete profile; the bars make the strengths readable in seconds.</div>', unsafe_allow_html=True)

        if not passmap_teams.empty:
            pass_sunderland = passmap_teams[passmap_teams["Team"].astype(str).str.casefold().eq("sunderland")]
            if not pass_sunderland.empty:
                pass_row = pass_sunderland.iloc[0]
                lane_metrics = [
                    ("Pass volume", "Passes per 90"),
                    ("Pass accuracy", "Pass completion %"),
                    ("Progressive passing", "Progressive passes per 90"),
                    ("Chance-creation value", "xT per 90"),
                    ("Wide delivery", "Crosses per 90"),
                ]
                lanes = pd.DataFrame([
                    {"Lane": label, "Percentile": passmap_percentile(passmap_teams, pass_row, metric)}
                    for label, metric in lane_metrics
                ])
                lane_fig = go.Figure(go.Scatter(
                    x=lanes["Percentile"], y=lanes["Lane"], mode="lines+markers+text",
                    line=dict(color="#C8CDD4", width=4), marker=dict(size=19, color=GOLD, line=dict(color=WHITE, width=3)),
                    text=[relative_band(v).replace(" profile", "") for v in lanes["Percentile"]], textposition="middle right",
                    textfont=dict(color=NAVY, size=12),
                    hovertemplate="<b>%{y}</b><br>League percentile: %{x:.0f}<extra></extra>",
                ))
                lane_fig.add_vline(x=50, line_color="#AEB6C2", line_width=1.5)
                lane_fig.update_layout(
                    title="How Sunderland move the ball",
                    xaxis=dict(range=[0, 108], visible=False, fixedrange=True),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=13, color=NAVY), fixedrange=True),
                    showlegend=False,
                )
                st.plotly_chart(fig_layout(lane_fig, 390, bottom_legend=False), width="stretch")

    with match_tab:
        match_view = understat_matches.copy()
        match_view["Date"] = pd.to_datetime(match_view["Date"], errors="coerce")
        match_view = match_view.dropna(subset=["Date"]).sort_values("Date")
        match_view["xG rolling"] = pd.to_numeric(match_view["xG"], errors="coerce").rolling(5, min_periods=1).mean()
        match_view["xGA rolling"] = pd.to_numeric(match_view["xGA"], errors="coerce").rolling(5, min_periods=1).mean()
        result_colors = {"w": GOLD, "d": "#8B8B8B", "l": NAVY}
        rhythm = go.Figure()
        rhythm.add_trace(go.Scatter(
            x=match_view["Date"], y=match_view["xG rolling"], mode="lines",
            name="Attacking output", line=dict(color=GOLD, width=4),
            hovertemplate="<b>%{x|%d %b}</b><br>5-match attacking trend: %{y:.2f} xG<extra></extra>",
        ))
        rhythm.add_trace(go.Scatter(
            x=match_view["Date"], y=match_view["xGA rolling"], mode="lines",
            name="Defensive exposure", line=dict(color=NAVY, width=3),
            hovertemplate="<b>%{x|%d %b}</b><br>5-match defensive trend: %{y:.2f} xGA<extra></extra>",
        ))
        rhythm.add_trace(go.Scatter(
            x=match_view["Date"], y=[0.08] * len(match_view), mode="markers",
            name="Result", marker=dict(size=11, color=[result_colors.get(r, "#8B8B8B") for r in match_view["Result"]], line=dict(color=WHITE, width=1.5)),
            customdata=np.stack([match_view["Result"].str.upper(), match_view["Home/Away"].str.upper()], axis=-1),
            hovertemplate="<b>%{x|%d %b}</b><br>Result: %{customdata[0]} · %{customdata[1]}<extra></extra>",
        ))
        rhythm.add_hline(y=1, line_color="#D1D5DB", line_width=1.5, annotation_text="1.0 xG reference", annotation_font_color=MUTED)
        rhythm.update_layout(
            title="Five-match rhythm: attack versus defensive exposure",
            xaxis=dict(title="Match date", gridcolor="#EEF0F3"),
            yaxis=dict(title="Expected goals per match", rangemode="tozero", gridcolor="#EEF0F3"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_layout(rhythm, 510), width="stretch")
        st.markdown('<div class="visual-caption">The red line is the attacking trend; black is the xG conceded trend. The match-result dots along the baseline keep the story of each run visible without turning the page into a spreadsheet.</div>', unsafe_allow_html=True)

    with formation_tab:
        form = understat_formation.copy()
        form = form[pd.to_numeric(form["Minutes"], errors="coerce") >= 120].copy()
        form["Attack edge"] = pd.to_numeric(form["xG per 90"], errors="coerce") - pd.to_numeric(form["xGA per 90"], errors="coerce")
        formation_fig = px.scatter(
            form, x="xGA per 90", y="xG per 90", size="UsageShare%", color="Attack edge",
            text="Formation", color_continuous_scale=[[0, NAVY], [0.5, "#8B8B8B"], [1, GOLD]],
            size_max=72,
            hover_data={"UsageShare%": ":.1f", "xG per 90": ":.2f", "xGA per 90": ":.2f", "Attack edge": ":.2f"},
        )
        max_axis = max(float(form["xG per 90"].max()), float(form["xGA per 90"].max()), 1.0) + .35
        formation_fig.add_shape(type="line", x0=0, y0=0, x1=max_axis, y1=max_axis, line=dict(color="#AEB6C2", width=1.5, dash="dot"))
        formation_fig.add_annotation(x=max_axis * .14, y=max_axis * .87, text="More attacking<br>than exposed", showarrow=False, font=dict(color=GOLD, size=12))
        formation_fig.add_annotation(x=max_axis * .82, y=max_axis * .12, text="More exposed<br>than attacking", showarrow=False, font=dict(color=MUTED, size=12))
        formation_fig.update_traces(textposition="top center", marker=dict(line=dict(color=WHITE, width=2)))
        formation_fig.update_layout(
            title="Formation lens: output versus exposure",
            coloraxis_showscale=False,
            xaxis=dict(title="Defensive exposure · xGA per 90", range=[0, max_axis], gridcolor="#EEF0F3"),
            yaxis=dict(title="Attacking output · xG per 90", range=[0, max_axis], gridcolor="#EEF0F3"),
        )
        st.plotly_chart(fig_layout(formation_fig, 560), width="stretch")
        st.markdown('<div class="visual-caption">Each circle is a formation. Bigger circles mean it has been used more; the best territory is above the diagonal, where Sunderland create more than they concede.</div>', unsafe_allow_html=True)

    page_report_download(
        "Download DNA report HTML",
        "Sunderland AFC — Recruitment DNA Report",
        "Current team benchmark and recruitment context",
        [
            "This report retains the original formation output structure and includes only supported DNA dimensions.",
            "Unsupported pillars have been removed rather than calculated from unrelated statistics.",
            "The full Premier League player file supplies the actual performance layer required to rebuild the striker-role model.",
        ],
        figures=[("Sunderland identity wheel", radar_fig), ("Match rhythm", rhythm), ("Formation lens", formation_fig)],
        tables=[("DNA coverage", understat_dna), ("Formation summary", understat_formation), ("Sunderland match inputs", understat_matches)],
        filename="sunderland_afc_understat_dna_source_report.html",
        key="understat_dna_report_download",
    )
    st.stop()

    st.markdown("### Provisional striker recruitment hypothesis")
    req = pd.DataFrame({
        "Striker Requirement": PHASES,
        "Why it matters for Sunderland AFC": [
            "The No.9 must convert central occupation into reliable shot and box presence.",
            "Sunderland need a forward who can connect with the eights, wingers and attacking midfielder.",
            "Vertical depth runs and attacking the centre-back/full-back channel are central to the model.",
            "The striker must combine with wide runners and attack crosses and cutbacks.",
            "Aerial value helps against direct play and at set pieces, without defining the whole profile.",
            "Le Bris' front line needs an active first defender who can counter-press and screen passes.",
        ],
        "Model Weight": [22, 18, 22, 15, 8, 15],
    })
    fig = px.bar(req, x="Model Weight", y="Striker Requirement", orientation="h", text="Model Weight", color="Model Weight", color_continuous_scale=[[0, LIGHT_NAVY], [1, GOLD]])
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 30], gridcolor="#E5E7EB")
    fig.update_layout(title="Sunderland AFC striker model weights", coloraxis_showscale=False)
    st.plotly_chart(fig_layout(fig, 430), width="stretch")
    st.dataframe(req, width="stretch", hide_index=True)

    if not dna_by_season.empty:
        st.markdown("### Team DNA evolution")
        long = dna_by_season.melt(id_vars=["Coach Season"], var_name="Phase", value_name="Score")
        long["Phase"] = long["Phase"].str.replace("DNA ", "", regex=False)
        fig = px.line(long, x="Coach Season", y="Score", color="Phase", markers=True, color_discrete_sequence=[NAVY, GOLD, BLUE, RED, "#6B7280", "#7C3AED", "#A16207", "#334155", "#93C5FD"])
        fig.update_yaxes(range=[0, 100], gridcolor="#E5E7EB")
        fig.update_layout(title="Sunderland AFC playing-phase DNA by season", legend_title_text="")
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
        """Sunderland AFC ST Fit =
0.22 Box Threat
+ 0.18 Link-up Play
+ 0.22 Channel Running
+ 0.15 Wide Combination
+ 0.08 Aerial Outlet
+ 0.15 Defensive Work

Scouting Opportunity =
0.42 Sunderland AFC ST Fit
+ 0.25 Hidden Gem Opportunity
+ 0.18 Best System Fit
+ 0.15 Age Upside
- superstar / elite-club visibility penalty

Hidden Gem Opportunity =
Sunderland AFC tactical fit
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
        "Sunderland AFC — Tactical DNA Report",
        "Club striker model and recruitment logic",
        [
            "The Sunderland recruitment model is provisional: public match data establishes a baseline, while video and authorised event data are needed to validate the tactical interpretation.",
            "The striker model is not built for a poacher-only profile. It rewards a forward who can attack the box, link the front line, run channels, offer an aerial outlet and lead the first defensive action.",
            "The three recruitment contexts used in the model are 4-3-3, 4-2-3-1 and 3-4-3, covering the principal striker roles expected in the Sunderland structure.",
        ],
        figures=[],
        tables=dna_tables,
        filename="sunderland_afc_tactical_dna_report.html",
        key="dna_report_download",
    )

elif page == "Similarity Scouting":
    st.markdown('<div class="section-header">Similarity Scouting</div>', unsafe_allow_html=True)
    st.caption("Choose a benchmark player. The tool returns the closest profiles in the current pool.")

    benchmark_pool = players.sort_values("Sunderland AFC ST Fit", ascending=False).copy()
    benchmark_names = benchmark_pool["Name"].astype(str).tolist()
    benchmark_name = st.selectbox("Benchmark player", benchmark_names, index=0, key="similarity_benchmark")
    benchmark_row = benchmark_pool[benchmark_pool["Name"].astype(str) == benchmark_name].iloc[0]

    c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
    with c1:
        top_n = st.slider("Number of similar players", 5, 30, 12, 1)
    with c2:
        apply_current_filters = st.checkbox("Apply current sidebar filters to candidates", value=True)
    with c3:
        exclude_current_club = st.checkbox("Exclude current Sunderland AFC players", value=True)

    candidate_pool = df.copy() if apply_current_filters else players.copy()
    if exclude_current_club and "Club" in candidate_pool.columns:
        candidate_pool = candidate_pool[~candidate_pool["Club"].astype(str).str.contains("sunderland", case=False, na=False)]

    sims = similar_players(benchmark_row, candidate_pool, top_n=top_n)

    b1, b2 = st.columns([0.30, 0.70], vertical_alignment="center")
    with b1:
        render_player_image(benchmark_row, width=132)
        st.markdown(f"### Benchmark: {benchmark_row['Name']}")
        st.markdown(f"**{player_identity_line(benchmark_row)}**")
        st.markdown(f"<span class='tag'>{benchmark_row['Best Role']}</span><span class='tag'>{benchmark_row['Best System']}</span><span class='tag'>ST Fit {benchmark_row['Sunderland AFC ST Fit']:.1f}</span>", unsafe_allow_html=True)
    with b2:
        st.plotly_chart(profile_strip_chart(benchmark_row.to_frame().T, "Benchmark fingerprint"), width="stretch")

    if sims.empty:
        st.warning("No similar candidates found under the current filters. Try removing some filters or allowing current Sunderland AFC players.")
    else:
        sim_cols = ["Name","Club","Age","Country","Continent","Best Role","Scout Tag","Similarity Score","ST Fit Gap vs Benchmark","Scouting Opportunity","Sunderland AFC ST Fit","Hidden Gem Opportunity","Best System"]
        sim_comp = pd.concat([benchmark_row.to_frame().T, sims.head(4)], ignore_index=True)
        st.markdown("### Similarity canvas")
        st.plotly_chart(scout_canvas_figure(candidate_pool, sim_comp, "Benchmark and closest role families", benchmark_name=str(benchmark_row["Name"])), width="stretch")
        st.markdown('<div class="visual-caption">The map shows role context. Hover a player to inspect the profile.</div>', unsafe_allow_html=True)

        alignment_left, alignment_right = st.columns([0.54, 0.46], vertical_alignment="top")
        with alignment_left:
            st.plotly_chart(profile_strip_chart(sim_comp, "Benchmark vs closest profile fingerprints"), width="stretch")
        with alignment_right:
            st.plotly_chart(similarity_alignment_chart(benchmark_row, sims.head(4)), width="stretch")

        top_match = sims.iloc[0]
        st.markdown(
            f"""
            <div class="navy-card">
            <b>Closest match:</b> <b>{top_match['Name']}</b> · <b>{top_match['Similarity Score']:.1f}%</b><br>
            {similarity_trait_reading(benchmark_row, top_match)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Recommended similar profiles")
        cards = sims.head(min(top_n, 12)).reset_index(drop=True)
        for start in range(0, len(cards), 2):
            columns = st.columns(2)
            for offset, column in enumerate(columns):
                card_index = start + offset
                if card_index >= len(cards):
                    continue
                candidate = cards.iloc[card_index]
                with column:
                    photo_col, copy_col = st.columns([0.20, 0.80], vertical_alignment="center")
                    with photo_col:
                        render_player_image(candidate, width=66)
                    with copy_col:
                        st.markdown(
                            f"""
                            <div class="similarity-card">
                                <div class="similarity-rank">#{card_index + 1} · closest profile</div>
                                <div class="similarity-name">{html.escape(str(candidate['Name']))}</div>
                                <div class="similarity-meta">{html.escape(player_identity_line(candidate))} · {html.escape(str(candidate['Best Role']))}</div>
                                <div style="margin-top:8px"><span class="similarity-score">{float(candidate['Similarity Score']):.1f}%</span></div>
                                <div class="similarity-trait">{similarity_trait_reading(benchmark_row, candidate)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        with st.expander("Open exact ranked data"):
            st.dataframe(report_table(sims, sim_cols, top_n), width="stretch", hide_index=True)

        page_report_download(
            "Download similarity scouting report HTML",
            "Sunderland AFC — Similarity Scouting Report",
            f"Benchmark: {benchmark_row['Name']}",
            [
                f"This report uses {benchmark_row['Name']} as the benchmark profile and searches for similar strikers based on role scores, Sunderland AFC striker phases and system-fit behaviour.",
                "Similarity is not a final quality ranking. It is a scouting workflow that helps answer: if the club likes this benchmark profile, which players in the database behave most similarly?",
                "The top similar players should be treated as video-review recommendations and checked for league context, physical profile, contract situation and tactical role before any recruitment decision.",
            ],
            figures=[("Similarity canvas", scout_canvas_figure(candidate_pool, sim_comp, "Benchmark and closest role families", benchmark_name=str(benchmark_row["Name"]))), ("Benchmark vs closest fingerprints", profile_strip_chart(sim_comp, "Benchmark vs closest profile fingerprints")), ("Alignment differences", similarity_alignment_chart(benchmark_row, sims.head(4)))],
            tables=[("Similarity recommendations", report_table(sims, sim_cols, top_n))],
            filename="sunderland_afc_similarity_scouting_report.html",
            key="similarity_report_download",
        )
