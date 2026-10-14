"""
bracket.py
Step 3: simulates an NCAA tournament bracket thousands of times and reports
each team's chance of reaching every round and winning the title.

It uses the bracket structure straight from Kaggle (MNCAATourneySeeds.csv +
MNCAATourneySlots.csv), so the First Four, regions and Final Four pairings
are exactly the real ones. Every game's win probability comes from the
team ratings in ratings.py.

Location:
  Men's:   every game is at a neutral site.
  Women's: the top 16 seeds (1-4 in each region) host the first two rounds,
           so they get home court in those games (HOST_SEEDS below).

Usage:
    python bracket.py                    # men's, latest season with seeds
    python bracket.py --gender W
    python bracket.py --season 2025      # a past bracket (shows how the real champion was rated)
    python bracket.py --sims 50000

    from bracket import simulate_bracket
    odds, picks = simulate_bracket(data, ratings, season, sims=20000)
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

try:
    from scipy.special import ndtr as _norm_cdf
except ImportError:                                   # scipy not installed
    from math import erf, sqrt
    _norm_cdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))

from kaggle_data import load_all
from ratings import season_ratings, GAME_SD

HOST_SEEDS = {"M": 0, "W": 4}          # seeds 1..N host rounds 1-2
HOST_ROUNDS = {1, 2}

ROUND_COLS = {0: "R64", 1: "R32", 2: "S16", 3: "E8", 4: "F4", 5: "Final", 6: "Champion"}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sims")


def _slot_round(slot: str) -> int:
    """'R1W1' -> 1, 'R6CH' -> 6; First Four slots like 'W16' -> 0."""
    m = re.match(r"R(\d)", slot)
    return int(m.group(1)) if m else 0


def _ordered_slots(slots: pd.DataFrame) -> list:
    """Slots in an order where every game's two inputs are already decided."""
    remaining = slots.copy()
    done, order = set(), []
    while not remaining.empty:
        pending = set(remaining["Slot"])
        ready = remaining[remaining.apply(
            lambda r: all(x in done or x not in pending for x in (r["StrongSeed"], r["WeakSeed"])),
            axis=1)]
        if ready.empty:
            raise ValueError("bracket slots have a cycle or a missing input")
        for r in ready.itertuples():
            order.append((r.Slot, r.StrongSeed, r.WeakSeed))
            done.add(r.Slot)
        remaining = remaining[~remaining["Slot"].isin(done)]
    return order


def simulate_bracket(data: dict, ratings: pd.DataFrame, season: int, sims: int = 20000,
                     gender: str = "M", sd: float = GAME_SD, home_court: float = None,
                     seed: int = 0):
    """
    Monte Carlo the bracket. ratings: table from season_ratings() (index TeamID,
    column Rating). Returns (odds DataFrame, model's pick for every slot).
    """
    seeds = data["seeds"][data["seeds"]["Season"] == season]
    slots = data["slots"][data["slots"]["Season"] == season]
    if seeds.empty or slots.empty:
        raise ValueError(f"no seeds/slots for season {season}")
    hca = ratings.attrs.get("home_court", 3.0) if home_court is None else home_court

    teams = seeds["TeamID"].to_numpy()
    idx = {t: i for i, t in enumerate(teams)}
    rating = np.array([ratings["Rating"].get(t, ratings["Rating"].min()) for t in teams])
    seed_num = seeds["SeedNum"].to_numpy()
    by_label = dict(zip(seeds["Seed"], seeds["TeamID"]))
    host_max = HOST_SEEDS.get(gender, 0)

    rng = np.random.default_rng(seed)
    winners = {}                                   # slot -> array of team indexes (one per sim)
    picks = {}                                     # slot -> model's pick (higher rated team)
    reach = np.zeros((len(teams), 7))              # team x round reached
    order = _ordered_slots(slots)

    def entrant(label):
        if label in winners:
            return winners[label], picks[label]
        i = idx[by_label[label]]
        return np.full(sims, i), i

    # every seeded team without a First Four game starts in the round of 64
    first_four_labels = {x for s in order if _slot_round(s[0]) == 0 for x in s[1:]}
    for label, t in by_label.items():
        if label not in first_four_labels:
            reach[idx[t], 0] = sims

    for slot, strong, weak in order:
        rnd = _slot_round(slot)
        a, pa = entrant(strong)
        b, pb = entrant(weak)
        margin = rating[a] - rating[b]
        if host_max and rnd in HOST_ROUNDS:
            a_host = (seed_num[a] <= host_max) & (seed_num[b] > host_max)
            b_host = (seed_num[b] <= host_max) & (seed_num[a] > host_max)
            margin = margin + hca * (a_host.astype(float) - b_host.astype(float))
        p_a = _norm_cdf(margin / sd)
        a_wins = rng.random(sims) < p_a
        w = np.where(a_wins, a, b)
        winners[slot] = w
        picks[slot] = pa if rating[pa] >= rating[pb] else pb
        np.add.at(reach[:, rnd], w, 1)          # First Four winner -> R64, R1 winner -> R32 ...

    names = data["teams"].set_index("TeamID")["TeamName"]
    odds = pd.DataFrame(reach / sims, columns=[ROUND_COLS[k] for k in range(7)])
    odds.insert(0, "TeamID", teams)
    odds.insert(1, "Team", [names.get(t, t) for t in teams])
    odds.insert(2, "Seed", seeds["Seed"].to_numpy())
    odds.insert(3, "Rating", rating.round(2))
    odds = odds.sort_values(["Champion", "Final", "F4"], ascending=False).reset_index(drop=True)

    pick_table = pd.DataFrame(
        [{"Slot": s, "Round": _slot_round(s), "Pick": names.get(teams[picks[s]], teams[picks[s]]),
          "Seed": seeds["Seed"].to_numpy()[picks[s]]} for s, _, _ in order])
    return odds, pick_table


def actual_champion(data: dict, season: int):
    """Name of the real champion, if that tournament's results are in the data."""
    g = data["games"]
    t = g[(g["Season"] == season) & g["Tourney"]]
    if t.empty:
        return None
    final = t.sort_values("DayNum").iloc[-1]
    return int(final["WTeamID"])


def print_report(odds: pd.DataFrame, picks: pd.DataFrame, label: str, season: int,
                 sims: int, top: int = 25, champ_id=None, names=None):
    print(f"\n{label} {season - 1}-{str(season)[2:]} NCAA tournament -- {sims:,} simulations\n")
    show = odds.head(top).copy()
    for c in ["R32", "S16", "E8", "F4", "Final", "Champion"]:
        show[c] = (show[c] * 100).map(lambda v: f"{v:5.1f}%")
    print(show[["Team", "Seed", "Rating", "R32", "S16", "E8", "F4", "Final", "Champion"]]
          .to_string(index=False))

    f4 = picks[picks["Round"] == 4]
    final = picks[picks["Round"] == 5]
    champ = picks[picks["Round"] == 6]
    print("\nModel's bracket (higher-rated team every game):")
    print(f"  Final Four:  {', '.join(f'{r.Pick} ({r.Seed})' for r in f4.itertuples())}")
    print(f"  Final:       {', '.join(f'{r.Pick}' for r in final.itertuples())}")
    print(f"  Champion:    {champ.iloc[0].Pick}")

    if champ_id is not None:
        row = odds[odds["TeamID"] == champ_id]
        if not row.empty:
            r = row.iloc[0]
            rank = int(row.index[0]) + 1
            print(f"\nActual champion: {r.Team} ({r.Seed}) -- model gave them "
                  f"{r.Champion:.1%} to win it all (#{rank} most likely)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=["M", "W"], default="M")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    data = load_all(args.gender)
    season = args.season or int(data["seeds"]["Season"].max())
    ratings = season_ratings(data, season)
    odds, picks = simulate_bracket(data, ratings, season, args.sims, args.gender)

    label = "Men's" if args.gender == "M" else "Women's"
    print_report(odds, picks, label, season, args.sims, args.top, actual_champion(data, season))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.gender}_bracket_odds_{season}.csv")
    odds.to_csv(out, index=False)
    print(f"\nSaved round-by-round odds for all {len(odds)} teams to {out}")