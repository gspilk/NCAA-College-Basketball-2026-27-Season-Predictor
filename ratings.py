"""
ratings.py
Step 2: team ratings from game results, plus the win-probability model.

THE RATING
Every team gets one number: how many points better it is than an average
Division I team on a neutral floor. Ratings are the values that best
explain every game's final margin:

    margin = rating(team) - rating(opponent) + home_court * location

where location is +1 at home, -1 away, 0 neutral. Home court is estimated
from the same games. This is least squares with a small ridge penalty
(RIDGE), which keeps ratings stable early in a season and centers them on 0.

WIN PROBABILITY
    P(team A wins) = NormalCDF(predicted margin / GAME_SD)
GAME_SD (how much real margins scatter around the prediction) is measured
on games the model did NOT see, so the probabilities are honest.

VALIDATION (python ratings.py)
For each past season: rate teams on regular-season games only, then
predict that season's NCAA tournament games. Reports accuracy, Brier score,
log loss and calibration, compared with "the better seed wins".

Usage:
    from ratings import fit_ratings, season_ratings, win_prob
    r = season_ratings(data, 2026)             # one season's ratings table
    win_prob(r.loc[1181, "Rating"], r.loc[1242, "Rating"], loc=0)

    python ratings.py                 # men's validation + top 25 for 2026
    python ratings.py --gender W      # women's
"""

import argparse
import os
from math import erf, sqrt, log

import numpy as np
import pandas as pd

from kaggle_data import load_all

RIDGE = 1.0          # penalty strength; ~ "one game's worth" of shrinkage toward 0
MARGIN_CAP = None    # cap blowout margins (e.g. 25); None = use real margins
GAME_SD = 11.4       # measured on 2006-2025 men's tournaments (validate() re-measures it)
LOC_SIGN = {"H": 1, "A": -1, "N": 0}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ratings")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_ratings(games: pd.DataFrame, ridge: float = RIDGE, margin_cap=MARGIN_CAP):
    """
    games: one row per game (WTeamID, LTeamID, WScore, LScore, WLoc).
    Returns (ratings Series indexed by TeamID, home_court float).
    """
    teams = np.unique(np.concatenate([games["WTeamID"].values, games["LTeamID"].values]))
    idx = {t: i for i, t in enumerate(teams)}
    n_teams, n_games = len(teams), len(games)

    margin = (games["WScore"] - games["LScore"]).to_numpy(dtype=float)
    if margin_cap is not None:
        margin = np.minimum(margin, margin_cap)
    loc = games["WLoc"].map(LOC_SIGN).to_numpy(dtype=float)

    # columns: one per team, last column = home court
    w = games["WTeamID"].map(idx).to_numpy()
    l = games["LTeamID"].map(idx).to_numpy()
    rows = np.arange(n_games)
    X = np.zeros((n_games, n_teams + 1))
    X[rows, w] = 1.0
    X[rows, l] = -1.0
    X[:, -1] = loc

    penalty = np.full(n_teams + 1, ridge)
    penalty[-1] = 0.0                          # don't shrink home court
    A = X.T @ X + np.diag(penalty)
    b = X.T @ margin
    coef = np.linalg.solve(A, b)

    ratings = pd.Series(coef[:-1], index=teams, name="Rating")
    ratings -= ratings.mean()
    return ratings, float(coef[-1])


def predict_margin(r_team: float, r_opp: float, loc: int, home_court: float) -> float:
    """Expected margin for team vs opp. loc: +1 home, -1 away, 0 neutral."""
    return r_team - r_opp + home_court * loc


def _norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def win_prob(r_team: float, r_opp: float, loc: int = 0, home_court: float = 0.0,
             sd: float = GAME_SD) -> float:
    """Probability team beats opp."""
    return _norm_cdf(predict_margin(r_team, r_opp, loc, home_court) / sd)


# ---------------------------------------------------------------------------
# Season ratings table
# ---------------------------------------------------------------------------

def season_ratings(data: dict, season: int, through_day: int = None,
                   include_tourney: bool = False) -> pd.DataFrame:
    """
    Ratings table for one season: TeamName, Conf, W, L, Rating, Rank.
    through_day: only use games up to this DayNum (e.g. 132 = end of regular season).
    """
    g = data["games"]
    g = g[g["Season"] == season]
    if not include_tourney:
        g = g[~g["Tourney"]]
    if through_day is not None:
        g = g[g["DayNum"] <= through_day]
    ratings, hca = fit_ratings(g)

    wins = g["WTeamID"].value_counts()
    losses = g["LTeamID"].value_counts()
    names = data["teams"].set_index("TeamID")["TeamName"]
    conf = data["conferences"]
    conf = conf[conf["Season"] == season].set_index("TeamID")["ConfAbbrev"] if not conf.empty else pd.Series(dtype=str)

    table = pd.DataFrame({"Rating": ratings.round(2)})
    table.index.name = "TeamID"
    table["TeamName"] = table.index.map(names)
    table["Conf"] = table.index.map(conf)
    table["W"] = table.index.map(wins).fillna(0).astype(int)
    table["L"] = table.index.map(losses).fillna(0).astype(int)
    table = table.sort_values("Rating", ascending=False)
    table["Rank"] = range(1, len(table) + 1)
    table.attrs["home_court"] = hca
    return table[["Rank", "TeamName", "Conf", "W", "L", "Rating"]]


# ---------------------------------------------------------------------------
# Validation on NCAA tournaments the model never saw
# ---------------------------------------------------------------------------

def _tourney_predictions(data: dict, seasons) -> pd.DataFrame:
    games, seeds = data["games"], data["seeds"]
    rows = []
    for season in seasons:
        reg = games[(games["Season"] == season) & ~games["Tourney"]]
        tour = games[(games["Season"] == season) & games["Tourney"]]
        if reg.empty or tour.empty:
            continue
        ratings, hca = fit_ratings(reg)
        seed_num = seeds[seeds["Season"] == season].set_index("TeamID")["SeedNum"] if not seeds.empty else {}
        for g in tour.itertuples():
            # look at each game from the lower TeamID's side so "won" isn't always True
            a, b = sorted((g.WTeamID, g.LTeamID))
            a_won = a == g.WTeamID
            loc = LOC_SIGN[g.WLoc] * (1 if a_won else -1)
            pred = predict_margin(ratings.get(a, 0.0), ratings.get(b, 0.0), loc, hca)
            actual = (g.WScore - g.LScore) * (1 if a_won else -1)
            rows.append({"Season": season, "pred": pred, "actual": actual, "a_won": a_won,
                         "seed_a": seed_num.get(a, np.nan), "seed_b": seed_num.get(b, np.nan)})
    return pd.DataFrame(rows)


def validate(data: dict, first_season: int = None, last_season: int = None, verbose: bool = True):
    """Out-of-sample check on NCAA tournament games. Returns the measured game SD."""
    games = data["games"]
    tour_seasons = sorted(games.loc[games["Tourney"], "Season"].unique())
    first = first_season or max(tour_seasons[0], tour_seasons[-1] - 19)
    last = last_season or tour_seasons[-1]
    seasons = [s for s in tour_seasons if first <= s <= last]
    df = _tourney_predictions(data, seasons)

    sd = float(np.sqrt(np.mean((df["actual"] - df["pred"]) ** 2)))
    p = np.array([_norm_cdf(x / sd) for x in df["pred"]])
    y = df["a_won"].to_numpy(dtype=float)
    picked = np.mean((p > 0.5) == (y == 1))
    brier = np.mean((p - y) ** 2)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logloss = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))

    # baseline: better (lower) seed wins; same-seed games (First Four) count as coin flips
    s = df.dropna(subset=["seed_a", "seed_b"])
    diff = s["seed_b"] - s["seed_a"]
    seed_hits = np.where(diff == 0, 0.5, (diff > 0) == s["a_won"])
    seed_acc = float(np.mean(seed_hits)) if len(s) else float("nan")

    if verbose:
        print(f"\nOut-of-sample check: NCAA tournaments {seasons[0]}-{seasons[-1]} "
              f"({len(df)} games, ratings from regular season only)\n")
        print(f"  Game SD (measured)      : {sd:.2f} points")
        print(f"  Picked the winner       : {picked:.1%}")
        print(f"  Better seed wins        : {seed_acc:.1%}   (baseline to beat)")
        print(f"  Brier score             : {brier:.3f}   (0.25 = coin flip, lower is better)")
        print(f"  Log loss                : {logloss:.3f}   (0.693 = coin flip)")

        fav = np.maximum(p, 1 - p)
        fav_won = np.where(p >= 0.5, y, 1 - y)
        cal = pd.DataFrame({"bucket": pd.cut(fav, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                            "p": fav, "won": fav_won})
        table = cal.groupby("bucket", observed=True).agg(games=("won", "size"),
                                                         predicted=("p", "mean"),
                                                         actual=("won", "mean"))
        print("\n  Calibration (favorite's predicted vs actual win rate):")
        for bucket, row in table.iterrows():
            print(f"    {str(bucket):12} {int(row.games):5d} games   "
                  f"predicted {row.predicted:.0%}   actual {row.actual:.0%}")
    return sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=["M", "W"], default="M")
    parser.add_argument("--season", type=int, default=None, help="season to rate (default: latest)")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    data = load_all(args.gender)
    label = "Men's" if args.gender == "M" else "Women's"
    sd = validate(data)

    season = args.season or int(data["games"]["Season"].max())
    table = season_ratings(data, season)
    print(f"\n{label} {season - 1}-{str(season)[2:]} ratings (regular season), "
          f"home court = {table.attrs['home_court']:.2f} points\n")
    print(table.head(args.top).to_string())

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.gender}_ratings_{season}.csv")
    table.to_csv(out)
    print(f"\nSaved all {len(table)} teams to {out}")
    print(f"Suggested GAME_SD for ratings.py: {sd:.1f}")