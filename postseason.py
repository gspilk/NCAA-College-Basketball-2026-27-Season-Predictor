"""
postseason.py
Step 7: conference tournaments, the selection committee, and the NCAA
tournament -- played inside every simulated season, so a team that's
better than expected in the regular season is better in March too.

1. CONFERENCE TOURNAMENTS
   Single elimination, seeded by that simulation's conference standings,
   byes for top seeds when the field isn't a power of 2, neutral sites.
   The champion gets the automatic bid.

2. SELECTION COMMITTEE (learned from history)
   The committee sees results, not true strength, so each team gets
       selection score = end-of-season rating + C * win %
   where the rating is what a rating system would measure (true strength
   plus measurement noise) and C is fit on 2012-2026: the value that best
   matches who actually received at-large bids. The same score sets the
   seeds (S-curve). `python postseason.py --fit` shows how well it matches.

3. NCAA TOURNAMENT
   31 automatic bids + at-large bids = 68 teams. First Four: the last four
   at-large teams and the four lowest-rated automatic qualifiers. Seeds go
   into four regions by S-curve. Men's: all neutral sites. Women's: seeds
   1-4 host the first two rounds.

Usage:
    python postseason.py                 # men's 2026-27 championship odds
    python postseason.py --gender W
    python postseason.py --fit           # refit + test the committee model
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
INELIGIBLE_CSV = os.path.join(ROOT, "ineligible_teams.csv")
FIELD_SIZE = 68
OBS_SD = 2.0               # how far a measured end-of-season rating is from true strength
HOST_LINES = {"M": 0, "W": 4}
REGION_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
ROUNDS = ["R64", "R32", "S16", "E8", "F4", "Final", "Champion"]


def _bracket_order(size: int) -> list:
    """Standard seeded bracket order, e.g. size 8 -> [1, 8, 4, 5, 2, 7, 3, 6]."""
    order = [1]
    while len(order) < size:
        n = len(order) * 2
        order = [x for s in order for x in (s, n + 1 - s)]
    return order


def _play(a, b, strength, rng, edge=0.0, game_sd=GAME_SD):
    """Vectorized games between team-index arrays a and b (-1 = bye). Returns winners, losers."""
    sims = np.arange(a.shape[0])[:, None] if a.ndim == 2 else np.arange(a.shape[0])
    sa = strength[sims, np.where(a < 0, 0, a)]
    sb = strength[sims, np.where(b < 0, 0, b)]
    a_wins = rng.random(a.shape) < _cdf((sa - sb + edge) / game_sd)
    a_wins = np.where(b < 0, True, np.where(a < 0, False, a_wins))
    winner = np.where(a_wins, a, b)
    loser = np.where(a_wins, b, a)
    loser = np.where((a < 0) | (b < 0), -1, loser)
    return winner, loser


try:
    from scipy.special import ndtr as _cdf
except ImportError:
    from math import erf, sqrt
    _cdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))


# ---------------------------------------------------------------------------
# 1. Conference tournaments
# ---------------------------------------------------------------------------

def conference_tournaments(res: dict, rng) -> np.ndarray:
    """Returns auto-bid winner index per (sim, conference) and updates W/L in res."""
    strength, rank, conf = res["strength"], res["conf_rank"], res["conf"]
    sims = strength.shape[0]
    rows = np.arange(sims)[:, None]
    confs = [c for c in pd.unique(conf[pd.notna(conf)])]
    champs = np.zeros((sims, len(confs)), dtype=int)
    for ci, c in enumerate(confs):
        cols = np.where(conf == c)[0]
        n = len(cols)
        if n == 1:
            champs[:, ci] = cols[0]
            continue
        seeded = cols[np.argsort(rank[:, cols], axis=1)]          # seed 1 first
        size = 1 << (n - 1).bit_length()
        slots = np.array([s - 1 if s <= n else -1 for s in _bracket_order(size)])
        field = np.where(slots >= 0, seeded[:, np.clip(slots, 0, None)], -1)
        while field.shape[1] > 1:
            w, l = _play(field[:, 0::2], field[:, 1::2], strength, rng)
            for arr, who in ((res["wins"], w), (res["losses"], l)):
                ok = who >= 0
                np.add.at(arr, (np.broadcast_to(rows, who.shape)[ok], who[ok]), 1)
            field = w
        champs[:, ci] = field[:, 0]
    res["conf_list"] = confs
    return champs


# ---------------------------------------------------------------------------
# 2. Selection committee
# ---------------------------------------------------------------------------

def ineligible_ids(data: dict) -> set:
    """Teams that can't receive an NCAA bid (e.g. still transitioning to Division I)."""
    if not os.path.exists(INELIGIBLE_CSV):
        pd.DataFrame(columns=["TeamName"]).to_csv(INELIGIBLE_CSV, index=False)
        print(f"  [info] created {os.path.basename(INELIGIBLE_CSV)} -- list teams not eligible "
              f"for the NCAA tournament (e.g. reclassifying from Division II)")
        return set()
    from schedule import name_lookup, _norm
    names = pd.read_csv(INELIGIBLE_CSV)["TeamName"].dropna()
    gender = "W" if data["teams"]["TeamID"].min() >= 3000 else "M"
    lookup = name_lookup(data, gender)
    ids = set()
    for n in names:
        t = lookup.get(_norm(n))
        if t is None:
            print(f"  [warn] ineligible_teams.csv: '{n}' not found -- skipped")
        else:
            ids.add(t)
    return ids


def _past_auto_bids(data: dict, season: int, seeds: pd.DataFrame) -> set:
    gender = "W" if data["teams"]["TeamID"].min() >= 3000 else "M"
    path = os.path.join(KAGGLE_DIR, f"{gender}ConferenceTourneyGames.csv")
    conf = data["conferences"]
    conf = conf[conf["Season"] == season].set_index("TeamID")["ConfAbbrev"]
    autos = set()
    if os.path.exists(path):
        ct = pd.read_csv(path)
        ct = ct[ct["Season"] == season].sort_values("DayNum")
        autos = set(ct.groupby("ConfAbbrev")["WTeamID"].last())
    # conferences with no tournament game in the file: best seed from that conference
    field = seeds.assign(Conf=seeds["TeamID"].map(conf))
    covered = set(conf.reindex(list(autos)))
    for c, grp in field.groupby("Conf"):
        if c not in covered:
            autos.add(grp.sort_values("SeedNum")["TeamID"].iloc[0])
    return autos & set(seeds["TeamID"])


def fit_committee(data: dict, first: int = 2012, verbose: bool = True) -> dict:
    """Find C (win % weight) that best reproduces past at-large selections."""
    from preseason import all_season_ratings
    ratings = all_season_ratings(data)
    g = data["games"]
    seasons = sorted(set(data["seeds"]["Season"]) & set(ratings["Season"]))
    seasons = [s for s in seasons if s >= first and s != 2021]   # 2021: COVID-shortened season
    rows = []
    for s in seasons:
        reg = g[(g["Season"] == s) & ~g["Tourney"]]
        w = reg["WTeamID"].value_counts()
        l = reg["LTeamID"].value_counts()
        r = ratings[ratings["Season"] == s].set_index("TeamID")["Rating"]
        seeds = data["seeds"][data["seeds"]["Season"] == s]
        autos = _past_auto_bids(data, s, seeds)
        df = pd.DataFrame({"rating": r})
        df["wpct"] = w.reindex(df.index).fillna(0) / (w.reindex(df.index).fillna(0) + l.reindex(df.index).fillna(0))
        df["in_field"] = df.index.isin(seeds["TeamID"])
        df["auto"] = df.index.isin(autos)
        df["line"] = df.index.map(seeds.set_index("TeamID")["SeedNum"])
        df["Season"] = s
        rows.append(df)
    hist = pd.concat(rows)

    best = None
    for c in np.arange(0, 41, 2.5):
        hits, total, line_err = 0, 0, []
        for s, df in hist.groupby("Season"):
            score = df["rating"] + c * df["wpct"]
            pool = df[~df["auto"]]
            k = int(pool["in_field"].sum())
            picked = score[pool.index].nlargest(k).index
            hits += pool.loc[picked, "in_field"].sum()
            total += k
            field = df[df["in_field"]]
            pred_line = (score[field.index].rank(ascending=False, method="first") - 1) // 4 + 1
            line_err.append(np.abs(pred_line.clip(upper=16) - field["line"]).mean())
        acc = hits / total
        if best is None or acc > best["at_large_accuracy"] + 1e-9:
            best = {"C": float(c), "at_large_accuracy": float(acc),
                    "avg_seed_line_error": float(np.mean(line_err)), "seasons": f"{seasons[0]}-{seasons[-1]}"}
    if verbose:
        print(f"\nCommittee model (fit on {best['seasons']}, skipping 2021):")
        print(f"  selection score = rating + {best['C']:.1f} x win %")
        print(f"  At-large bids picked correctly : {best['at_large_accuracy']:.1%}")
        print(f"  Average seed-line miss         : {best['avg_seed_line_error']:.2f} lines")
    return best


def committee_settings(data: dict, gender: str, refit: bool = False) -> dict:
    path = os.path.join(ROOT, "data", "ratings", f"{gender}_committee.json")
    if os.path.exists(path) and not refit:
        return json.load(open(path))
    best = fit_committee(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(best, open(path, "w"), indent=2)
    return best


def select_field(res: dict, champs: np.ndarray, C: float, ineligible: set, rng):
    """Returns (score, field order sims x 68, at-large flag sims x 68)."""
    strength, teams = res["strength"], res["teams"]
    sims, n_t = strength.shape
    games = (res["wins"] + res["losses"]).astype(float)
    wpct = np.divide(res["wins"], games, out=np.zeros_like(games), where=games > 0)
    score = strength + rng.normal(0, OBS_SD, strength.shape) + C * wpct

    eligible = ~np.isin(teams, list(ineligible))
    auto = np.zeros((sims, n_t), bool)
    auto[np.arange(sims)[:, None], champs] = True
    auto &= eligible[None, :]               # an ineligible champion's bid is simply lost here
    n_at_large = FIELD_SIZE - auto.sum(axis=1)

    pool = np.where(auto | ~eligible[None, :], -np.inf, score)
    order = np.argsort(-pool, axis=1)
    rank_in_pool = np.empty_like(order)
    np.put_along_axis(rank_in_pool, order, np.arange(n_t)[None, :].repeat(sims, 0), axis=1)
    at_large = rank_in_pool < n_at_large[:, None]
    field = auto | at_large

    fscore = np.where(field, score, -np.inf)
    forder = np.argsort(-fscore, axis=1)[:, :FIELD_SIZE]
    al_flag = np.take_along_axis(at_large, forder, axis=1)
    return score, forder, al_flag


# ---------------------------------------------------------------------------
# 3. NCAA tournament
# ---------------------------------------------------------------------------

def build_brackets(forder: np.ndarray, al_flag: np.ndarray, score: np.ndarray):
    """
    Returns pos_a, pos_b (sims x 64): team at each bracket position, with pos_b
    the second team when that position is a First Four game (else -1), and the
    seed line of each position.
    """
    sims = forder.shape[0]
    line_of_pos = np.array([REGION_ORDER] * 4).ravel()
    pos_a = np.full((sims, 64), -1)
    pos_b = np.full((sims, 64), -1)
    for s in range(sims):
        teams, al = forder[s], al_flag[s]
        last_al = [i for i in range(FIELD_SIZE - 1, -1, -1) if al[i]][:4]
        last_auto = [i for i in range(FIELD_SIZE - 1, -1, -1) if not al[i]][:4]
        pair_of = {}
        for group in (sorted(last_al), sorted(last_auto)):
            if len(group) == 4:
                pair_of[group[0]] = group[1]
                pair_of[group[2]] = group[3]
        skip = set(pair_of.values())
        entries = [i for i in range(FIELD_SIZE) if i not in skip][:64]
        for r, i in enumerate(entries):
            line, k = r // 4 + 1, r % 4
            region = k if line % 2 == 1 else 3 - k
            pos = region * 16 + REGION_ORDER.index(line)
            pos_a[s, pos] = teams[i]
            if i in pair_of:
                pos_b[s, pos] = teams[pair_of[i]]
    return pos_a, pos_b, line_of_pos


def play_tournament(pos_a, pos_b, line_of_pos, strength, rng, home_court: float,
                    host_lines: int, n_teams: int) -> np.ndarray:
    """Returns reach counts: teams x 7 rounds (R64 ... Champion)."""
    sims = pos_a.shape[0]
    rows = np.arange(sims)[:, None]
    reach = np.zeros((n_teams, 7))
    # First Four
    ff = pos_b >= 0
    w, _ = _play(pos_a, np.where(ff, pos_b, -1), strength, rng)
    field = w
    np.add.at(reach[:, 0], field.ravel(), 1)
    lines = np.broadcast_to(line_of_pos, field.shape)
    for rnd in range(1, 7):
        a, b = field[:, 0::2], field[:, 1::2]
        la, lb = lines[:, 0::2], lines[:, 1::2]
        edge = 0.0
        if host_lines and rnd in (1, 2):
            edge = home_court * (((la <= host_lines) & (lb > host_lines)).astype(float)
                                 - ((lb <= host_lines) & (la > host_lines)).astype(float))
        field, _ = _play(a, b, strength, rng, edge)
        lines = np.where(field == a, la, lb)
        np.add.at(reach[:, rnd], field.ravel(), 1)
    return reach


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run(data: dict, gender: str, sims: int, seed: int = 0, refit: bool = False):
    from season import load_inputs, simulate_season
    season, table, info, conf, sched = load_inputs(data, gender, seed)
    committee = committee_settings(data, gender, refit)
    res = simulate_season(sched, table["Preseason"], conf, info["team_sd"], info["home_court"],
                          sims, seed=seed)
    rng = np.random.default_rng(seed + 1)
    champs = conference_tournaments(res, rng)
    score, forder, al_flag = select_field(res, champs, committee["C"], ineligible_ids(data), rng)
    pos_a, pos_b, line_of_pos = build_brackets(forder, al_flag, score)
    reach = play_tournament(pos_a, pos_b, line_of_pos, res["strength"], rng,
                            info["home_court"], HOST_LINES[gender], len(res["teams"]))

    teams = res["teams"]
    n_t = len(teams)
    in_field = np.zeros(n_t)
    np.add.at(in_field, forder.ravel(), 1)
    auto = np.zeros(n_t)
    np.add.at(auto, champs.ravel(), 1)
    line_sum = np.zeros(n_t)
    line_cnt = np.zeros(n_t)
    one_seed = np.zeros(n_t)
    for arr in (pos_a, pos_b):
        ok = arr >= 0
        lines = np.broadcast_to(line_of_pos, arr.shape)[ok]
        np.add.at(line_sum, arr[ok], lines)
        np.add.at(line_cnt, arr[ok], 1)
        np.add.at(one_seed, arr[ok][lines == 1], 1)

    names = data["teams"].set_index("TeamID")["TeamName"]
    out = pd.DataFrame({
        "Team": [names.get(t, t) for t in teams], "Conf": res["conf"],
        "Preseason": table["Preseason"].reindex(teams).round(2).to_numpy(),
        "W": res["wins"].mean(0).round(1), "L": res["losses"].mean(0).round(1),
        "AutoBid": auto / sims, "MakeTourney": in_field / sims,
        "AvgSeed": np.divide(line_sum, line_cnt, out=np.full(n_t, np.nan), where=line_cnt > 0).round(1),
        "OneSeed": one_seed / sims,
    }, index=pd.Index(teams, name="TeamID"))
    for k, name in enumerate(ROUNDS):
        out[name] = reach[:, k] / sims
    out = out.sort_values(["Champion", "Final", "F4"], ascending=False)
    return season, out, committee


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=["M", "W"], default="M")
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit", action="store_true", help="refit and test the committee model")
    args = parser.parse_args()
    data = load_all(args.gender)

    if args.fit:
        committee_settings(data, args.gender, refit=True)
    else:
        season, out, committee = run(data, args.gender, args.sims, args.seed)
        label = "Men's" if args.gender == "M" else "Women's"
        print(f"\n{label} {season - 1}-{str(season)[2:]} -- {args.sims:,} full seasons "
              f"(regular season, conference tournaments, selection, NCAA tournament)\n")
        show = out.head(args.top).copy()
        show["Record"] = show["W"].map("{:.1f}".format) + "-" + show["L"].map("{:.1f}".format)
        for c in ["MakeTourney", "OneSeed", "S16", "F4", "Champion"]:
            show[c] = (show[c] * 100).map("{:.1f}%".format)
        show = show.rename(columns={"MakeTourney": "Make NCAA", "OneSeed": "1 seed",
                                    "AvgSeed": "Avg seed", "S16": "Sweet 16", "F4": "Final Four"})
        print(show[["Team", "Conf", "Record", "Make NCAA", "Avg seed", "1 seed", "Sweet 16",
                    "Final Four", "Champion"]].to_string(index=False))
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"{args.gender}_championship_odds_{season}.csv")
        out.to_csv(path)
        print(f"\nSaved odds for all {len(out)} teams to {path}")