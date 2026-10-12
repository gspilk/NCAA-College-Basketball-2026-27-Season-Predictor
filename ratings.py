"""
Steps 2 & 3: Game prediction model + validation.

The model:
    expected margin = (Team A SRS - Team B SRS) + home court edge
    win probability = normal CDF(expected margin / SD)

SRS is "points per game better than an average team, adjusted for schedule",
so the difference between two teams' SRS is how many points one should beat
the other by on a neutral floor.

Running this file checks the model against every game from last season and
fits the two numbers it needs (home court edge and SD) from real results.

Usage:
    python ratings.py                 # men's 2026
    python ratings.py --gender women
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from statistics import NormalDist

DATA_DIR = Path("data")

# Fitted from 5,752 D1 games in 2025-26 (run validate() to refit for another season)
HOME_COURT = 2.3
GAME_SD = 10.6


# ---------------------------------------------------------------------------
# Name cleanup
# ---------------------------------------------------------------------------

# Schools whose game-log name differs from the ratings-page name
ALIASES = {
    "Maryland-Baltimore County": "UMBC",
    "Texas A&M-Corpus Christi": "TAMUCC",
    "Southern Illinois-Edwardsville": "SIU Edwardsville",
    "Virginia Military Institute": "VMI",
}


def clean_name(name: str) -> str:
    """
    Fix three problems in scraped school names:
      - garbled dashes ("Nevadaâ\x80\x93Las Vegas" -> "Nevada-Las Vegas")
      - AP rank stuck on the end ("Michigan (3)" -> "Michigan")
      - abbreviations that differ between pages (ALIASES)
    """
    if not isinstance(name, str):
        return name
    try:
        name = name.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    name = name.replace("\xa0", " ").replace("–", "-")
    name = re.sub(r"\s*\(\d+\)\s*$", "", name).strip()
    return ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ratings(season: int = 2026, gender: str = "men") -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / gender / str(season) / "team_ratings.csv")
    df = df.rename(columns={"SRS_SRS": "srs", "SRS_OSRS": "osrs", "SRS_DSRS": "dsrs",
                            "Adjusted_ORtg": "ortg", "Adjusted_DRtg": "drtg",
                            "Adjusted_NRtg": "nrtg"})
    df["School"] = df["School"].map(clean_name)
    keep = ["School", "Conf", "slug", "W", "L", "srs", "osrs", "dsrs", "ortg", "drtg", "nrtg"]
    return df[keep]


def load_games(season: int = 2026, gender: str = "men") -> pd.DataFrame:
    """
    One row per game (not per team). Each game appears in both teams' logs,
    so keep only the copy from the home team's side (or, at a neutral site,
    the alphabetically first team's side).
    """
    raw = pd.read_csv(DATA_DIR / gender / str(season) / "games_raw.csv")
    ratings = load_ratings(season, gender)
    name_to_slug = dict(zip(ratings["School"], ratings["slug"]))

    g = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="%a, %b %d, %Y", errors="coerce"),
        "type": raw["Type"],                      # REG, CTOURN, NCAA, NIT, CBC
        "loc": raw["Unnamed: 4"].fillna("H"),    # H = home, @ = away, N = neutral
        "team": raw["team_slug"],
        "opp": raw["Opponent"].map(clean_name).map(name_to_slug),  # NaN = non-Division I opponent
        "pts": raw["Tm"],
        "opp_pts": raw["Opp"],
    })

    g = g.dropna(subset=["opp", "pts", "opp_pts"])      # drop non-D1 and unplayed games
    is_home = g["loc"] == "H"
    is_neutral_first = (g["loc"] == "N") & (g["team"] < g["opp"])
    g = g[is_home | is_neutral_first].copy()
    g["neutral"] = g["loc"] == "N"
    g["margin"] = g["pts"] - g["opp_pts"]
    return g.drop_duplicates(subset=["date", "team", "opp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def expected_margin(srs_a: float, srs_b: float, a_is_home: bool = False,
                    neutral: bool = True, home_court: float = HOME_COURT) -> float:
    edge = 0.0 if neutral else (home_court if a_is_home else -home_court)
    return srs_a - srs_b + edge


def win_prob(srs_a: float, srs_b: float, a_is_home: bool = False,
             neutral: bool = True, home_court: float = HOME_COURT,
             sd: float = GAME_SD) -> float:
    """Probability team A beats team B."""
    margin = expected_margin(srs_a, srs_b, a_is_home, neutral, home_court)
    return NormalDist(0, sd).cdf(margin)


# ---------------------------------------------------------------------------
# Validation against last season
# ---------------------------------------------------------------------------

def validate(season: int = 2026, gender: str = "men"):
    ratings = load_ratings(season, gender)
    games = load_games(season, gender)
    srs = dict(zip(ratings["slug"], ratings["srs"]))

    games["srs_diff"] = games["team"].map(srs) - games["opp"].map(srs)

    # Fit home court: the average amount home teams beat their rating gap by
    non_neutral = games[~games["neutral"]]
    home_court = (non_neutral["margin"] - non_neutral["srs_diff"]).mean()

    games["pred"] = games["srs_diff"] + np.where(games["neutral"], 0, home_court)
    sd = (games["margin"] - games["pred"]).std()

    games["p_win"] = [NormalDist(0, sd).cdf(m) for m in games["pred"]]
    games["won"] = (games["margin"] > 0).astype(int)

    picked_right = ((games["p_win"] > 0.5) == (games["won"] == 1)).mean()
    brier = ((games["p_win"] - games["won"]) ** 2).mean()

    print(f"\n{gender.title()}'s {season - 1}-{str(season)[2:]}: {len(games)} D1 vs D1 games\n")
    print(f"  Fitted home court edge : {home_court:.2f} points")
    print(f"  Fitted game SD         : {sd:.2f} points")
    print(f"  Picked the winner      : {picked_right:.1%}")
    print(f"  Brier score            : {brier:.3f}   (lower is better; 0.25 = coin flip)")

    # Calibration: when the model says 70%, does that team win ~70%?
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fav_p = np.maximum(games["p_win"], 1 - games["p_win"])
    fav_won = np.where(games["p_win"] >= 0.5, games["won"], 1 - games["won"])
    cal = pd.DataFrame({"bucket": pd.cut(fav_p, bins), "p": fav_p, "won": fav_won})
    table = cal.groupby("bucket", observed=True).agg(games=("won", "size"),
                                                     predicted=("p", "mean"),
                                                     actual=("won", "mean"))
    print("\n  Calibration (favorite's predicted vs actual win rate):")
    for bucket, row in table.iterrows():
        print(f"    {str(bucket):12} {int(row.games):5d} games   "
              f"predicted {row.predicted:.0%}   actual {row.actual:.0%}")

    print("\n  Note: SRS was calculated from these same games, so this is a")
    print("  best case. Real predictions for a new season will be less accurate.")
    print(f"\n  Paste these into ratings.py:  HOME_COURT = {home_court:.1f}   GAME_SD = {sd:.1f}")
    return home_court, sd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--gender", choices=["men", "women"], default="men")
    args = parser.parse_args()
    validate(args.season, args.gender)