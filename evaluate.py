"""
evaluate.py
Walk-forward test: replays a season day by day, the way it happens in real
life. Before each game day, ratings are updated using ONLY the games already
played; that day's games are predicted; then the real results are added and
the next day is predicted. Every game is predicted without the model having
seen its result.

Compares two versions:
  Updated daily     -- the real model (preseason rating + results so far)
  Preseason only    -- never updates (shows how much in-season learning helps)

Reports: % of winners picked, Brier score, average miss on the margin,
accuracy by month, calibration, and the biggest upsets.

Usage:
    python evaluate.py                    # men's, last completed season in the Kaggle data
    python evaluate.py --season 2025      # a different past season
    python evaluate.py --current          # this season, using the games you've entered
    python evaluate.py --gender W
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    from scipy.special import ndtr
except ImportError:                                   # scipy not installed
    from math import erf, sqrt
    ndtr = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))

from kaggle_data import load_all
from ratings import update_ratings, GAME_SD

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data", "evaluation")
MONTHS = ["~Nov", "~Dec", "~Jan", "~Feb", "~Mar"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def past_season_inputs(data: dict, season: int):
    """Preseason ratings built ONLY from earlier seasons, plus that season's real games."""
    from preseason import all_season_ratings, build_preseason, backtest as pre_backtest
    ratings = all_season_ratings(data)
    before = ratings[ratings["Season"] < season]
    before.attrs = ratings.attrs
    table, info = build_preseason(data, season, ratings=before)
    team_sd = pre_backtest(data, before, info["since"], verbose=False)["team_sd"]

    g = data["games"]
    g = g[(g["Season"] == season) & ~g["Tourney"]]
    away_won = g["WLoc"].eq("A").to_numpy()
    games = pd.DataFrame({
        "DayNum": g["DayNum"].to_numpy(),
        "HomeID": np.where(away_won, g["LTeamID"], g["WTeamID"]),
        "AwayID": np.where(away_won, g["WTeamID"], g["LTeamID"]),
        "Neutral": g["WLoc"].eq("N").to_numpy(),
        "HomeScore": np.where(away_won, g["LScore"], g["WScore"]),
        "AwayScore": np.where(away_won, g["WScore"], g["LScore"]),
    })
    return table["Preseason"], team_sd, info["home_court"], games


def current_season_inputs(data: dict, gender: str):
    """This season's preseason ratings plus the real games you've entered scores for."""
    from season import load_inputs
    season, table, info, conf, sched = load_inputs(data, gender)
    games = sched[(sched["Source"] == "real") & sched["HomeScore"].notna()]
    games = games[["DayNum", "HomeID", "AwayID", "Neutral", "HomeScore", "AwayScore"]]
    return season, table["Preseason"], info["team_sd"], info["home_court"], games


# ---------------------------------------------------------------------------
# Walk-forward replay
# ---------------------------------------------------------------------------

def walk_forward(pre: pd.Series, team_sd: float, home_court: float, games: pd.DataFrame,
                 game_sd: float = GAME_SD) -> pd.DataFrame:
    teams = set(pre.index)
    skipped = (~(games["HomeID"].isin(teams) & games["AwayID"].isin(teams))).sum()
    games = games[games["HomeID"].isin(teams) & games["AwayID"].isin(teams)].copy()
    games[["HomeScore", "AwayScore"]] = games[["HomeScore", "AwayScore"]].astype(float)
    games["Neutral"] = games["Neutral"].astype(bool)
    if skipped:
        print(f"  [info] skipped {skipped} games involving teams without a preseason rating")

    pre_sd = np.sqrt(game_sd ** 2 + 2 * team_sd ** 2)
    out = []
    days = sorted(games["DayNum"].unique())
    for i, day in enumerate(days):
        played = games[games["DayNum"] < day]               # ONLY games before today
        rating, sd = update_ratings(pre, played, team_sd, home_court, game_sd)
        today = games[games["DayNum"] == day]
        h, a = today["HomeID"].to_numpy(), today["AwayID"].to_numpy()
        edge = np.where(today["Neutral"], 0.0, home_court)

        pred = rating.reindex(h).to_numpy() - rating.reindex(a).to_numpy() + edge
        spread_sd = np.sqrt(game_sd ** 2 + sd.reindex(h).to_numpy() ** 2 + sd.reindex(a).to_numpy() ** 2)
        pre_pred = pre.reindex(h).to_numpy() - pre.reindex(a).to_numpy() + edge
        out.append(today.assign(Pred=pred, P_home=ndtr(pred / spread_sd),
                                PrePred=pre_pred, PreP=ndtr(pre_pred / pre_sd)))
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(days)} game days")
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _scores(df, pred_col, p_col):
    y = (df["HomeScore"] > df["AwayScore"]).to_numpy(dtype=float)
    p = df[p_col].to_numpy(dtype=float)
    actual = (df["HomeScore"] - df["AwayScore"]).to_numpy(dtype=float)
    picked = np.mean((p > 0.5) == (y == 1))
    brier = np.mean((p - y) ** 2)
    miss = np.mean(np.abs(actual - df[pred_col].to_numpy(dtype=float)))
    return picked, brier, miss


def report(df: pd.DataFrame, names: pd.Series, title: str):
    print(f"\n{title}  ({len(df):,} games, each predicted before it was played)\n")
    print(f"  {'':32} {'Picked winner':>14} {'Brier':>7} {'Avg margin miss':>16}")
    for label, pc, ppc in [("Updated daily (the model)", "Pred", "P_home"),
                           ("Preseason only (never updates)", "PrePred", "PreP")]:
        picked, brier, miss = _scores(df, pc, ppc)
        print(f"  {label:32} {picked:>13.1%} {brier:>7.3f} {miss:>13.1f} pts")
    print("  (Brier: 0.25 = coin flip, lower is better)")

    print("\n  By month (model):")
    block = (df["DayNum"] // 30).clip(upper=len(MONTHS) - 1)
    for b, grp in df.groupby(block):
        picked, brier, _ = _scores(grp, "Pred", "P_home")
        print(f"    {MONTHS[int(b)]:6} {len(grp):5,} games   picked {picked:.1%}   Brier {brier:.3f}")

    p = df["P_home"].to_numpy()
    y = (df["HomeScore"] > df["AwayScore"]).to_numpy(dtype=float)
    fav = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    cal = pd.DataFrame({"bucket": pd.cut(fav, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), "p": fav, "won": fav_won})
    table = cal.groupby("bucket", observed=True).agg(games=("won", "size"), predicted=("p", "mean"),
                                                     actual=("won", "mean"))
    print("\n  Calibration (when the model's favorite had this chance, how often did it win?):")
    for bucket, row in table.iterrows():
        print(f"    {str(bucket):12} {int(row.games):5,} games   predicted {row.predicted:.0%}   "
              f"actual {row.actual:.0%}")

    winner_p = np.where(y == 1, p, 1 - p)
    up = df.assign(WinnerP=winner_p).nsmallest(10, "WinnerP")
    print("\n  Biggest upsets (winner's predicted chance):")
    for r in up.itertuples():
        hw = r.HomeScore > r.AwayScore
        w, l = (r.HomeID, r.AwayID) if hw else (r.AwayID, r.HomeID)
        ws, ls = (r.HomeScore, r.AwayScore) if hw else (r.AwayScore, r.HomeScore)
        print(f"    {names.get(w, w):18} {int(ws):3}-{int(ls):<3} over {names.get(l, l):18} "
              f"({r.WinnerP:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=["M", "W"], default="M")
    parser.add_argument("--season", type=int, default=None, help="past season to replay (default: latest)")
    parser.add_argument("--current", action="store_true", help="this season, games you've entered so far")
    args = parser.parse_args()

    data = load_all(args.gender)
    label = "Men's" if args.gender == "M" else "Women's"
    if args.current:
        season, pre, team_sd, hca, games = current_season_inputs(data, args.gender)
        tag = "current"
    else:
        season = args.season or int(data["games"]["Season"].max())
        print(f"Building {label.lower()} {season - 1}-{str(season)[2:]} preseason ratings from earlier seasons only ...")
        pre, team_sd, hca, games = past_season_inputs(data, season)
        tag = "replay"

    print(f"Replaying {games['DayNum'].nunique()} game days ...")
    df = walk_forward(pre, team_sd, hca, games)
    names = data["teams"].set_index("TeamID")["TeamName"]
    report(df, names, f"{label} {season - 1}-{str(season)[2:]} walk-forward test")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.gender}_walkforward_{season}_{tag}.csv")
    df.assign(Home=df["HomeID"].map(names), Away=df["AwayID"].map(names)).to_csv(out, index=False)
    print(f"\nSaved every prediction to {out}")