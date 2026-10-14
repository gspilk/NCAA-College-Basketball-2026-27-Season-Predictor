"""
season.py
Step 6: simulates the regular season thousands of times.

Each simulated season:
  1. draws every team's true strength once:  preseason + Normal(0, TEAM_SD)
     (a team that's better than expected is better in EVERY game, like real
     surprise teams)
  2. plays every scheduled game:  margin = strength(home) - strength(away)
     + home court (unless neutral) + Normal(0, GAME_SD)
  3. records overall and conference wins and losses

Conference standings rank teams by conference win %, ties broken by true
strength. The per-simulation results are kept for the next steps
(conference tournaments, selection, the NCAA bracket).

TESTED (python season.py --backtest 2026): builds preseason ratings using
only seasons before 2025-26, plays the REAL 2025-26 schedule, and compares
with what actually happened -- error in projected wins and whether ~80% of
teams land inside their 80% win range.

Usage:
    python season.py                      # men's 2026-27 projections
    python season.py --backtest 2026      # test on 2025-26
    python season.py --gender W --sims 20000
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from kaggle_data import load_all, DATA_DIR as KAGGLE_DIR
from ratings import GAME_SD

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data", "sims")
CHUNK = 1000                     # simulations processed at once (keeps memory low)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_season(schedule: pd.DataFrame, preseason: pd.Series, conferences: pd.Series,
                    team_sd: float, home_court: float, sims: int = 10000,
                    game_sd: float = GAME_SD, seed: int = 0) -> dict:
    """
    schedule: HomeID, AwayID, Neutral, ConfGame.   preseason: TeamID -> rating.
    Returns per-simulation arrays (sims x teams) plus the team list.
    """
    teams = preseason.index.to_numpy()
    idx = {t: i for i, t in enumerate(teams)}
    sched = schedule[schedule["HomeID"].isin(idx) & schedule["AwayID"].isin(idx)]
    h = sched["HomeID"].map(idx).to_numpy()
    a = sched["AwayID"].map(idx).to_numpy()
    edge = np.where(sched["Neutral"].astype(bool), 0.0, home_court)
    conf_game = sched["ConfGame"].astype(bool).to_numpy()
    n_t, n_g = len(teams), len(sched)
    rng = np.random.default_rng(seed)

    strength = preseason.to_numpy()[None, :] + rng.normal(0, team_sd, (sims, n_t))
    wins = np.zeros((sims, n_t), dtype=np.int16)
    losses = np.zeros((sims, n_t), dtype=np.int16)
    cwins = np.zeros((sims, n_t), dtype=np.int16)
    closses = np.zeros((sims, n_t), dtype=np.int16)

    # one-hot game -> team matrices, so wins/losses per team are matrix products
    H = np.zeros((n_g, n_t), np.float32); H[np.arange(n_g), h] = 1
    A = np.zeros((n_g, n_t), np.float32); A[np.arange(n_g), a] = 1
    Hc, Ac = H * conf_game[:, None], A * conf_game[:, None]

    for start in range(0, sims, CHUNK):
        s = slice(start, min(start + CHUNK, sims))
        k = s.stop - s.start
        margin = strength[s][:, h] - strength[s][:, a] + edge + rng.normal(0, game_sd, (k, n_g))
        hw = (margin > 0).astype(np.float32)
        aw = 1.0 - hw
        wins[s] = hw @ H + aw @ A
        losses[s] = aw @ H + hw @ A
        cwins[s] = hw @ Hc + aw @ Ac
        closses[s] = aw @ Hc + hw @ Ac

    conf = conferences.reindex(teams).to_numpy()
    standing = conference_standings(cwins, closses, strength, conf)
    return {"teams": teams, "conf": conf, "strength": strength, "wins": wins, "losses": losses,
            "conf_wins": cwins, "conf_losses": closses, "conf_rank": standing}


def conference_standings(cwins, closses, strength, conf) -> np.ndarray:
    """Rank within each conference (1 = regular-season champion), per simulation."""
    games = (cwins + closses).astype(float)
    pct = np.divide(cwins, games, out=np.zeros_like(games), where=games > 0)
    key = pct + strength * 1e-4                     # ties broken by true strength
    rank = np.zeros_like(cwins, dtype=np.int16)
    for c in pd.unique(conf[pd.notna(conf)]):
        cols = np.where(conf == c)[0]
        order = np.argsort(-key[:, cols], axis=1)
        r = np.empty_like(order)
        np.put_along_axis(r, order, np.arange(1, len(cols) + 1)[None, :].repeat(len(order), 0), axis=1)
        rank[:, cols] = r
    return rank


def summarize(res: dict, names: pd.Series, preseason: pd.Series) -> pd.DataFrame:
    w, l = res["wins"], res["losses"]
    out = pd.DataFrame({
        "TeamName": [names.get(t, t) for t in res["teams"]],
        "Conf": res["conf"],
        "Preseason": preseason.reindex(res["teams"]).round(2).to_numpy(),
        "W": w.mean(0).round(1), "L": l.mean(0).round(1),
        "ConfW": res["conf_wins"].mean(0).round(1), "ConfL": res["conf_losses"].mean(0).round(1),
        "W_10th": np.percentile(w, 10, axis=0).astype(int),
        "W_90th": np.percentile(w, 90, axis=0).astype(int),
        "ConfTitle": (res["conf_rank"] == 1).mean(0).round(3),
    }, index=pd.Index(res["teams"], name="TeamID"))
    return out.sort_values("Preseason", ascending=False)


# ---------------------------------------------------------------------------
# Backtest on a past season
# ---------------------------------------------------------------------------

def real_schedule(data: dict, season: int) -> pd.DataFrame:
    """A past season's actual regular-season games, without conference tournaments."""
    g = data["games"]
    g = g[(g["Season"] == season) & ~g["Tourney"]].copy()
    conf = data["conferences"]
    conf = conf[conf["Season"] == season].set_index("TeamID")["ConfAbbrev"]
    gender = "W" if g["WTeamID"].min() >= 3000 else "M"
    ct_path = os.path.join(KAGGLE_DIR, f"{gender}ConferenceTourneyGames.csv")
    if os.path.exists(ct_path):
        ct = pd.read_csv(ct_path)
        ct = ct[ct["Season"] == season]
        drop = set(zip(ct["DayNum"], ct["WTeamID"], ct["LTeamID"]))
        g = g[[(d, w, l) not in drop for d, w, l in zip(g["DayNum"], g["WTeamID"], g["LTeamID"])]]
    home = np.where(g["WLoc"] == "A", g["LTeamID"], g["WTeamID"])
    away = np.where(g["WLoc"] == "A", g["WTeamID"], g["LTeamID"])
    s = pd.DataFrame({"HomeID": home, "AwayID": away, "Neutral": g["WLoc"].eq("N").to_numpy(),
                      "WTeamID": g["WTeamID"].to_numpy()})
    s["ConfGame"] = s["HomeID"].map(conf).eq(s["AwayID"].map(conf)) & s["HomeID"].map(conf).notna()
    return s


def backtest(data: dict, season: int, sims: int = 5000, seed: int = 0):
    from preseason import all_season_ratings, build_preseason, backtest as pre_backtest, membership

    ratings = all_season_ratings(data)
    before = ratings[ratings["Season"] < season]
    before.attrs = ratings.attrs
    table, info = build_preseason(data, season, ratings=before)
    team_sd = pre_backtest(data, before, info["since"], verbose=False)["team_sd"]
    conf = membership(data, season).reindex(table.index)
    sched = real_schedule(data, season)

    res = simulate_season(sched, table["Preseason"], conf, team_sd, info["home_court"], sims,
                          seed=seed)
    teams = res["teams"]
    actual_w = sched["WTeamID"].value_counts().reindex(teams).fillna(0).to_numpy()
    played = pd.concat([sched["HomeID"], sched["AwayID"]]).value_counts().reindex(teams).fillna(0).to_numpy()
    proj = res["wins"].mean(0)
    lo, hi = np.percentile(res["wins"], 10, axis=0), np.percentile(res["wins"], 90, axis=0)
    inside = np.mean((actual_w >= lo) & (actual_w <= hi))
    naive = played / 2                                 # baseline: everyone goes .500

    print(f"\nBacktest: {season - 1}-{str(season)[2:]} season, preseason ratings built only from "
          f"earlier seasons,\nplayed on the real schedule ({len(sched):,} games), {sims:,} simulations\n")
    print(f"  Average miss in projected wins   : {np.mean(np.abs(proj - actual_w)):.1f} wins   "
          f"(everyone .500 would miss by {np.mean(np.abs(naive - actual_w)):.1f})")
    print(f"  Correlation, projected vs actual : {np.corrcoef(proj, actual_w)[0, 1]:.2f}")
    print(f"  Teams inside their 80% range     : {inside:.0%}   (should be about 80%)")

    names = data["teams"].set_index("TeamID")["TeamName"]
    df = pd.DataFrame({"Team": [names.get(t, t) for t in teams], "Projected": proj.round(1),
                       "Actual": actual_w.astype(int), "Miss": (actual_w - proj).round(1)})
    print("\n  Biggest surprises (actual minus projected wins):")
    for r in pd.concat([df.nlargest(5, "Miss"), df.nsmallest(5, "Miss")]).itertuples():
        print(f"    {r.Team:22} projected {r.Projected:5.1f}  actual {r.Actual:3d}  ({r.Miss:+.1f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_inputs(data: dict, gender: str, seed: int = 0):
    """Preseason ratings, settings, conferences and schedule for the coming season."""
    from preseason import build_preseason, all_season_ratings, backtest as pre_backtest, membership
    from schedule import build_schedule

    season = int(data["games"]["Season"].max()) + 1
    ratings_dir = os.path.join(ROOT, "data", "ratings")
    csv = os.path.join(ratings_dir, f"{gender}_preseason_{season}.csv")
    js = csv.replace(".csv", ".json")
    if os.path.exists(csv) and os.path.exists(js):
        table = pd.read_csv(csv, index_col="TeamID")
        info = json.load(open(js))
    else:
        print("  [info] no saved preseason ratings -- building them (run preseason.py to see details)")
        ratings = all_season_ratings(data)
        table, info = build_preseason(data, season, ratings=ratings)
        info["team_sd"] = pre_backtest(data, ratings, info["since"], verbose=False)["team_sd"]
    conf = membership(data, season).reindex(table.index)
    sched = build_schedule(data, conf, table["Preseason"], season, gender, seed)
    return season, table, info, conf, sched


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=["M", "W"], default="M")
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--backtest", type=int, default=None)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = load_all(args.gender)

    if args.backtest:
        backtest(data, args.backtest, min(args.sims, 5000), args.seed)
    else:
        season, table, info, conf, sched = load_inputs(data, args.gender, args.seed)
        res = simulate_season(sched, table["Preseason"], conf, info["team_sd"],
                              info["home_court"], args.sims, seed=args.seed)
        names = data["teams"].set_index("TeamID")["TeamName"]
        summary = summarize(res, names, table["Preseason"])
        real = (sched["Source"] == "real").sum()
        label = "Men's" if args.gender == "M" else "Women's"
        print(f"\n{label} {season - 1}-{str(season)[2:]} regular season -- {args.sims:,} simulations "
              f"({len(sched):,} games, {real:,} from your real schedule)\n")
        show = summary.head(args.top).copy()
        show["Record"] = show["W"].map("{:.1f}".format) + "-" + show["L"].map("{:.1f}".format)
        show["Conf rec"] = show["ConfW"].map("{:.1f}".format) + "-" + show["ConfL"].map("{:.1f}".format)
        show["80% range"] = show["W_10th"].astype(str) + "-" + show["W_90th"].astype(str) + " W"
        show["Conf title"] = (show["ConfTitle"] * 100).map("{:.0f}%".format)
        print(show[["TeamName", "Conf", "Preseason", "Record", "80% range", "Conf rec", "Conf title"]]
              .to_string(index=False))
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, f"{args.gender}_season_{season}.csv")
        summary.to_csv(out)
        print(f"\nSaved projections for all {len(summary)} teams to {out}")