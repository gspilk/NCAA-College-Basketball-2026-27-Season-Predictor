"""
preseason.py
Step 4: preseason ratings for the coming season, learned from history.

Kaggle has no rosters, so the forecast uses what history says about how
much of a team's strength carries over from one season to the next:

    preseason = a + b1 * last season + b2 * two seasons ago
                  + b3 * three seasons ago + b4 * conference average

All ratings are end-of-regular-season ratings from ratings.py. The weights
(a, b1..b4) are fit by least squares on past seasons in the training
window, so the data decides how much carries over. "Conference average" is
last season's average rating of the team's NEW conference, which handles
realignment (see conference_changes.csv).

TESTED HONESTLY (leave-one-season-out): each season's preseason ratings are
built with weights fit on the OTHER seasons, then used to predict every
game of that season. Compared with two baselines: "same as last season"
and "every team equal".

UNCERTAINTY: TEAM_SD = how far a team's real season typically lands from
its preseason rating. The season simulator draws each team's true strength
once per simulated season from Normal(preseason, TEAM_SD), so a team that's
better than expected is better in every game, the way real seasons work.

Usage:
    python preseason.py                  # men's preseason ratings for next season
    python preseason.py --gender W
    python preseason.py --since 2008     # change the training window

    from preseason import build_preseason
    table, info = build_preseason(data, 2027)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

try:
    from scipy.special import ndtr as norm_cdf
except ImportError:                                   # scipy not installed
    from math import erf, sqrt
    norm_cdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))

from kaggle_data import load_all
from ratings import fit_ratings, GAME_SD, LOC_SIGN

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data", "ratings")
CHANGES_CSV = os.path.join(ROOT, "conference_changes.csv")
DEFAULT_SINCE = 2012           # first target season used to fit the weights
FEATURES = ["last", "two_ago", "three_ago", "conf_avg"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def all_season_ratings(data: dict) -> pd.DataFrame:
    """End-of-regular-season rating for every team in every season."""
    g = data["games"]
    rows, hca = [], {}
    for season, games in g[~g["Tourney"]].groupby("Season"):
        r, h = fit_ratings(games)
        hca[season] = h
        rows.append(pd.DataFrame({"Season": season, "TeamID": r.index, "Rating": r.values}))
    out = pd.concat(rows, ignore_index=True)
    out.attrs["home_court"] = hca
    return out


def conference_changes() -> pd.DataFrame:
    """
    Realignment you enter by hand (Kaggle's conference list stops at the
    last completed season). Columns: TeamName, NewConf  -- NewConf uses
    Kaggle's conference abbreviations (e.g. big_ten, acc, wcc).
    """
    if not os.path.exists(CHANGES_CSV):
        pd.DataFrame(columns=["TeamName", "NewConf"]).to_csv(CHANGES_CSV, index=False)
        print(f"  [info] created {os.path.basename(CHANGES_CSV)} -- add any conference "
              f"changes for the new season there")
    return pd.read_csv(CHANGES_CSV)


def membership(data: dict, season: int) -> pd.Series:
    """TeamID -> conference for a season (latest known + your changes for a future season)."""
    conf = data["conferences"]
    have = sorted(conf["Season"].unique())
    if season in have:
        return conf[conf["Season"] == season].set_index("TeamID")["ConfAbbrev"]
    m = conf[conf["Season"] == have[-1]].set_index("TeamID")["ConfAbbrev"].copy()
    changes = conference_changes()
    if not changes.empty:
        ids = data["teams"].set_index("TeamName")["TeamID"]
        for r in changes.itertuples():
            if r.TeamName in ids.index:
                m.loc[ids[r.TeamName]] = r.NewConf
            else:
                print(f"  [warn] conference_changes.csv: unknown team '{r.TeamName}' "
                      f"(use the exact name from MTeams.csv)")
    return m


def features_for(ratings: pd.DataFrame, data: dict, target: int) -> pd.DataFrame:
    """One row per team in the target season (or last season, for a future one)."""
    by_season = {s: g.set_index("TeamID")["Rating"] for s, g in ratings.groupby("Season")}
    last = by_season.get(target - 1)
    if last is None:
        return pd.DataFrame()
    conf = membership(data, target)
    teams = conf.index.union(last.index)
    df = pd.DataFrame(index=teams)
    df.index.name = "TeamID"
    df["last"] = last
    df["two_ago"] = by_season.get(target - 2, pd.Series(dtype=float))
    df["three_ago"] = by_season.get(target - 3, pd.Series(dtype=float))
    df["conf"] = conf
    conf_avg = last.groupby(conf.reindex(last.index)).mean()
    df["conf_avg"] = df["conf"].map(conf_avg)
    # teams new to Division I (no last season): start them at the bottom of their conference
    low = last.groupby(conf.reindex(last.index)).quantile(0.1)
    df["last"] = df["last"].fillna(df["conf"].map(low)).fillna(last.quantile(0.05))
    df["two_ago"] = df["two_ago"].fillna(df["last"])
    df["three_ago"] = df["three_ago"].fillna(df["two_ago"])
    df["conf_avg"] = df["conf_avg"].fillna(last.mean())
    return df


def training_rows(ratings: pd.DataFrame, data: dict, targets) -> pd.DataFrame:
    """Features + the actual end-of-season rating, for past seasons."""
    by_season = {s: g.set_index("TeamID")["Rating"] for s, g in ratings.groupby("Season")}
    rows = []
    for t in targets:
        f = features_for(ratings, data, t)
        if f.empty or t not in by_season:
            continue
        f = f.join(by_season[t].rename("actual"), how="inner")
        f["Season"] = t
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)


def fit_weights(rows: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(rows))] + [rows[c] for c in FEATURES])
    coef, *_ = np.linalg.lstsq(X, rows["actual"].to_numpy(), rcond=None)
    return coef


def apply_weights(f: pd.DataFrame, coef: np.ndarray) -> pd.Series:
    X = np.column_stack([np.ones(len(f))] + [f[c] for c in FEATURES])
    return pd.Series(X @ coef, index=f.index)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def _game_scores(games: pd.DataFrame, rating: pd.Series, hca: float, sd: float) -> dict:
    a = games["WTeamID"].map(rating).fillna(rating.min()).to_numpy()
    b = games["LTeamID"].map(rating).fillna(rating.min()).to_numpy()
    loc = games["WLoc"].map(LOC_SIGN).to_numpy()
    pred = a - b + hca * loc                       # from the winner's side
    actual = (games["WScore"] - games["LScore"]).to_numpy()
    p = norm_cdf(pred / sd)                        # chance the actual winner wins
    return {"pred": pred, "actual": actual, "p": p}


def backtest(data: dict, ratings: pd.DataFrame, since: int, verbose: bool = True) -> dict:
    """Leave-one-season-out test of preseason ratings on every regular-season game."""
    last_season = int(ratings["Season"].max())
    targets = list(range(since, last_season + 1))
    rows = training_rows(ratings, data, targets)
    hca_by = ratings.attrs["home_court"]
    g = data["games"]

    models = {"Preseason model": [], "Same as last season": [], "Every team equal": []}
    errors = []
    for t in sorted(rows["Season"].unique()):
        coef = fit_weights(rows[rows["Season"] != t])
        f = rows[rows["Season"] == t].set_index("TeamID")
        pre = apply_weights(f, coef)
        errors.append(f["actual"] - pre)
        season_games = g[(g["Season"] == t) & ~g["Tourney"]]
        hca = hca_by.get(t - 1, 3.0)
        models["Preseason model"].append(_game_scores(season_games, pre, hca, 1.0))
        models["Same as last season"].append(_game_scores(season_games, f["last"], hca, 1.0))
        models["Every team equal"].append(_game_scores(season_games, f["last"] * 0, hca, 1.0))

    errors = pd.concat(errors)
    team_sd = float(np.sqrt(np.mean(errors ** 2)))
    results = {}
    for name, parts in models.items():
        pred = np.concatenate([x["pred"] for x in parts])
        actual = np.concatenate([x["actual"] for x in parts])
        sd = float(np.sqrt(np.mean((actual - pred) ** 2)))
        p = norm_cdf(pred / sd)
        results[name] = {"picked": float(np.mean(p > 0.5) + 0.5 * np.mean(p == 0.5)),
                         "brier": float(np.mean((1 - p) ** 2)), "sd": sd}

    if verbose:
        n_games = len(np.concatenate([x["pred"] for x in models["Preseason model"]]))
        print(f"\nBacktest: seasons {targets[0]}-{last_season}, each predicted with weights "
              f"fit on the other seasons ({n_games:,} regular-season games)\n")
        print(f"  {'':22} {'Picked winner':>14} {'Brier':>8} {'Game SD':>9}")
        for name, r in results.items():
            print(f"  {name:22} {r['picked']:>13.1%} {r['brier']:>8.3f} {r['sd']:>8.1f}")
        print(f"\n  Team uncertainty (TEAM_SD): {team_sd:.1f} points -- how far a team's real "
              f"season\n  typically lands from its preseason rating")
    return {"team_sd": team_sd, "results": results}


def carryover_by_era(data: dict, ratings: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """How much of last season's rating carried over, in different eras."""
    last_season = int(ratings["Season"].max())
    rows = training_rows(ratings, data, range(1988, last_season + 1))
    eras = [(1988, 2000), (2001, 2010), (2011, 2020), (2021, last_season)]
    out = []
    for lo, hi in eras:
        r = rows[(rows["Season"] >= lo) & (rows["Season"] <= hi)]
        if len(r) < 50:
            continue
        slope = np.polyfit(r["last"], r["actual"], 1)[0]
        out.append({"Seasons": f"{lo}-{hi}", "Carryover": slope})
    out = pd.DataFrame(out)
    if verbose and not out.empty:
        print("\nHow much of last season's rating carries over (on its own):")
        for r in out.itertuples():
            print(f"  {r.Seasons}: {r.Carryover:.0%}")
    return out


# ---------------------------------------------------------------------------
# Build next season's ratings
# ---------------------------------------------------------------------------

def build_preseason(data: dict, target: int, since: int = DEFAULT_SINCE, ratings=None):
    ratings = all_season_ratings(data) if ratings is None else ratings
    last_season = int(ratings["Season"].max())
    rows = training_rows(ratings, data, range(since, last_season + 1))
    coef = fit_weights(rows)
    f = features_for(ratings, data, target)
    pre = apply_weights(f, coef)
    pre -= pre.mean()                                 # keep the league average at 0

    names = data["teams"].set_index("TeamID")["TeamName"]
    table = pd.DataFrame({
        "TeamName": f.index.map(names),
        "Conf": f["conf"],
        "LastSeason": f["last"].round(2),
        "Preseason": pre.round(2),
    }, index=f.index)
    table = table.dropna(subset=["Conf"])             # only teams in a conference this season
    table = table.sort_values("Preseason", ascending=False)
    table.insert(0, "Rank", range(1, len(table) + 1))
    info = {"target_season": target, "since": since,
            "weights": dict(zip(["intercept"] + FEATURES, map(float, coef))),
            "home_court": float(ratings.attrs["home_court"].get(last_season, 3.0))}
    return table, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=["M", "W"], default="M")
    parser.add_argument("--since", type=int, default=DEFAULT_SINCE)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    data = load_all(args.gender)
    label = "Men's" if args.gender == "M" else "Women's"
    print(f"Rating every {label.lower()} season ...")
    ratings = all_season_ratings(data)
    target = int(ratings["Season"].max()) + 1

    carryover_by_era(data, ratings)
    bt = backtest(data, ratings, args.since)
    table, info = build_preseason(data, target, args.since, ratings)
    info["team_sd"] = bt["team_sd"]
    info["game_sd"] = GAME_SD

    w = info["weights"]
    print(f"\nWeights (fit on {args.since}-{target - 1}):")
    print(f"  last season {w['last']:.2f}, two ago {w['two_ago']:.2f}, three ago "
          f"{w['three_ago']:.2f}, conference average {w['conf_avg']:.2f}")

    print(f"\n{label} {target - 1}-{str(target)[2:]} preseason ratings\n")
    print(table.head(args.top).to_string())

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.gender}_preseason_{target}.csv")
    table.to_csv(out)
    with open(out.replace(".csv", ".json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"\nSaved {len(table)} teams to {out} (+ settings in the .json next to it)")