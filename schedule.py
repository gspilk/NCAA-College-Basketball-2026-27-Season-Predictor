"""
schedule.py
Step 5: the regular-season schedule the season simulator plays out.

Two sources:

  REAL (for the actual prediction) -- a CSV you fill in from schedules you
  collect (conference composite schedules, school athletics sites):
      data/schedule/M_schedule_2027.csv      (W_ for women's)
      Date,Team,Opponent,Site                one row per team per game
      11-23,Duke,North Carolina,H            Site: H home, A away, N neutral
      11-23,North Carolina,Duke,A
  (or Date,Home,Away,Neutral, one row per game). The same game entered from
  both sides is counted once. Dates may leave off the year. Team names are
  matched with Kaggle's spellings file, so "UConn", "Connecticut" and
  "St. John's" all work. Unrecognized opponents are listed and skipped
  (usually non-Division I teams).

  GENERATED (for testing, and to fill gaps) -- a realistic schedule built
  from conference memberships:
      conference games  double round robin if that's 20 games or fewer,
                        otherwise play everyone once plus extra rematches
                        up to CONF_GAMES_LARGE
      non-conference    the rest of NON_CONF_TOTAL games: half against teams
                        of similar strength (often neutral-site events), half
                        against random opponents with the stronger team
                        usually hosting (like real "buy" games)

  MIXED -- if the real CSV only covers some teams, conferences whose real
  conference schedule is incomplete get a generated slate, and teams short
  of non-conference games get generated ones.

Output (both): one row per game -- Season, DayNum, HomeID, AwayID, Neutral,
ConfGame, Source ("real" / "generated").

Usage:
    python schedule.py --check 2026      # generate last season, compare with the real one
    python schedule.py                   # build next season's schedule (real CSV + fill)
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

from kaggle_data import load_all, DATA_DIR as KAGGLE_DIR

ROOT = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_DIR = os.path.join(ROOT, "data", "schedule")

# Tuned so a generated 2025-26 schedule matches the real one (python schedule.py --check 2026)
TOTAL_GAMES = 30              # Division I games per team before conference tournaments
CONF_GAMES_LARGE = 20         # conference games for big conferences (generated schedules)
MIN_COMPLETE_CONF_GAMES = 16  # a real conference slate counts as complete at this many games per team
NEUTRAL_SHARE = 0.33          # share of similar-strength non-conference games at neutral sites
SIMILAR_NOISE = 14.0          # how loosely "similar strength" opponents are matched (points)
CONF_START_DAY = 60           # roughly early January in Kaggle's DayNum


# ---------------------------------------------------------------------------
# Team name matching
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", str(name).lower().replace("&", " and ").replace("'", ""))
    return re.sub(r"\s+", " ", text).strip()


def _variants(name: str) -> list:
    """Common alternate spellings to try when a name isn't found as typed."""
    n = _norm(name)
    out = [n]
    for a, b in [(r"\bstate\b", "st"), (r"\bsaint\b", "st"), (r"\bst\b", "saint"),
                 (r"\bmount\b", "mt"), (r"^the ", ""), (r"\buniversity\b", ""),
                 (r"\bnorth carolina\b", "nc"), (r"\bsouth carolina\b", "sc")]:
        m = re.sub(r"\s+", " ", re.sub(a, b, n)).strip()
        if m not in out:
            out.append(m)
    both = re.sub(r"\bsaint\b", "st", re.sub(r"\bstate\b", "st", n))
    out.append(re.sub(r"\bmount\b", "mt", both))
    return out


def find_team(lookup: dict, name):
    for v in _variants(name):
        if v in lookup:
            return lookup[v]
    return None


def name_lookup(data: dict, gender: str = "M") -> dict:
    """Every known spelling -> TeamID (Kaggle's spellings file + official names)."""
    lookup = {_norm(n): t for n, t in zip(data["teams"]["TeamName"], data["teams"]["TeamID"])}
    path = os.path.join(KAGGLE_DIR, f"{gender}TeamSpellings.csv")
    if os.path.exists(path):
        sp = pd.read_csv(path, encoding="latin-1")
        for n, t in zip(sp.iloc[:, 0], sp.iloc[:, 1]):
            lookup.setdefault(_norm(n), int(t))
    lookup.setdefault("miami", lookup.get("miami fl"))      # Miami alone = Miami (FL)
    lookup = {k: v for k, v in lookup.items() if v is not None}
    return lookup


# ---------------------------------------------------------------------------
# Generated schedule
# ---------------------------------------------------------------------------

def _random_matching(ids: list, rng) -> list:
    ids = list(ids)
    rng.shuffle(ids)
    return [(ids[i], ids[i + 1]) for i in range(0, len(ids) - 1, 2)]


def conference_games(teams: list, rng, conf_games_large: int = CONF_GAMES_LARGE) -> list:
    """List of (home, away) for one conference."""
    n = len(teams)
    games = []
    if n < 2:
        return games
    if 2 * (n - 1) <= conf_games_large:                # double round robin
        for i in range(n):
            for j in range(n):
                if i != j:
                    games.append((teams[i], teams[j]))
        return games
    for i in range(n):                                 # everyone once
        for j in range(i + 1, n):
            games.append((teams[i], teams[j]) if rng.random() < 0.5 else (teams[j], teams[i]))
    extra = conf_games_large - (n - 1)
    played = {}
    for _ in range(extra):                             # rematches, swapping home court
        for a, b in _random_matching(teams, rng):
            key = frozenset((a, b))
            if key in played:
                continue
            played[key] = True
            first = next(g for g in games if set(g) == {a, b})
            games.append((first[1], first[0]))
    return games


def nonconf_games(need: pd.Series, strength: pd.Series, rng) -> list:
    """
    need: TeamID -> number of non-conference games still to schedule.
    Returns (home, away, neutral) tuples.
    """
    need = need.copy()
    games = []
    for rnd in range(int(need.max()) * 3):
        open_teams = need[need > 0].index.to_list()
        if len(open_teams) < 2:
            break
        if rnd % 2 == 0:                               # similar-strength round
            noisy = strength.reindex(open_teams).fillna(strength.min()) + rng.normal(0, SIMILAR_NOISE, len(open_teams))
            ordered = noisy.sort_values().index.to_list()
            pairs = [(ordered[i], ordered[i + 1]) for i in range(0, len(ordered) - 1, 2)]
        else:                                          # random-opponent round
            pairs = _random_matching(open_teams, rng)
        for a, b in pairs:
            sa, sb = strength.get(a, -20), strength.get(b, -20)
            strong, weak = (a, b) if sa >= sb else (b, a)
            if rnd % 2 == 0 and rng.random() < NEUTRAL_SHARE:
                games.append((strong, weak, True))     # neutral-site event
            else:
                home_strong = rng.random() < 0.8
                games.append((strong, weak, False) if home_strong else (weak, strong, False))
            need[a] -= 1
            need[b] -= 1
    # anyone still short (the odd team out) plays random opponents, who go one over
    all_teams = need.index.to_list()
    for t in need[need > 0].index:
        for _ in range(int(need[t])):
            o = t
            while o == t:
                o = all_teams[rng.integers(len(all_teams))]
            games.append((t, o, False) if rng.random() < 0.5 else (o, t, False))
    return games


def generate_schedule(conferences: pd.Series, strength: pd.Series, season: int,
                      seed: int = 0, fixed: pd.DataFrame = None) -> pd.DataFrame:
    """
    conferences: TeamID -> conference. strength: TeamID -> rating (used only to
    shape non-conference matchups). fixed: real games to keep (from the CSV).
    """
    rng = np.random.default_rng(seed)
    rows = []
    fixed = fixed if fixed is not None else pd.DataFrame(
        columns=["HomeID", "AwayID", "Neutral", "ConfGame", "DayNum"])

    # conference games: keep a conference's real slate only if it's complete
    for conf, members in conferences.groupby(conferences):
        teams = members.index.to_list()
        # "complete" = every team has at least 16 conference games (real slates run 16-20),
        # or the full double round robin for small conferences (e.g. 8 teams -> 14)
        expected = min(2 * (len(teams) - 1), MIN_COMPLETE_CONF_GAMES)
        real = fixed[fixed["ConfGame"] & fixed["HomeID"].isin(teams) & fixed["AwayID"].isin(teams)]
        counts = pd.concat([real["HomeID"], real["AwayID"]]).value_counts().reindex(teams).fillna(0)
        if len(real) and (counts >= expected).all():
            continue                                    # real slate is complete
        if len(real):
            fixed = fixed.drop(real.index)              # incomplete: replace with generated
        for h, a in conference_games(teams, rng):
            rows.append((h, a, False, True))

    # non-conference: top every team up to TOTAL_GAMES
    tmp = pd.DataFrame(rows, columns=["HomeID", "AwayID", "Neutral", "ConfGame"])
    allg = pd.concat([tmp, fixed[["HomeID", "AwayID", "Neutral", "ConfGame"]]], ignore_index=True)
    played = pd.concat([allg["HomeID"], allg["AwayID"]]).value_counts().reindex(conferences.index).fillna(0)
    need = (TOTAL_GAMES - played).clip(lower=0).astype(int)
    for h, a, neutral in nonconf_games(need, strength, rng):
        rows.append((h, a, neutral, False))

    gen = pd.DataFrame(rows, columns=["HomeID", "AwayID", "Neutral", "ConfGame"])
    gen["DayNum"] = np.where(gen["ConfGame"], rng.integers(CONF_START_DAY, 128, len(gen)),
                             rng.integers(7, CONF_START_DAY, len(gen)))
    gen["Source"] = "generated"
    real = fixed.assign(Source="real")
    out = pd.concat([real, gen], ignore_index=True)
    out.insert(0, "Season", season)
    return out.sort_values("DayNum").reset_index(drop=True)[
        ["Season", "DayNum", "HomeID", "AwayID", "Neutral", "ConfGame", "Source"]]


# ---------------------------------------------------------------------------
# Real schedule CSV
# ---------------------------------------------------------------------------

def _parse_date(value, season: int):
    """'2026-11-23', '11/23/2026', or no year at all ('11-23', '11/23'):
    July-December belong to the first calendar year of the season, January-June to the second."""
    text = str(value).strip()
    parts = [x for x in re.split(r"[-/]", text) if x]
    if len(parts) == 2 and all(x.isdigit() for x in parts):
        month, day = int(parts[0]), int(parts[1])
        year = season - 1 if month >= 7 else season
        return pd.Timestamp(year=year, month=month, day=day)
    if len(parts) == 3 and all(x.isdigit() for x in parts) and len(parts[0]) <= 2:
        month, day, year = map(int, parts)
        return pd.Timestamp(year=year + 2000 if year < 100 else year, month=month, day=day)
    return pd.to_datetime(text, errors="coerce")


def read_real_schedule(data: dict, conferences: pd.Series, season: int, gender: str = "M",
                       path: str = None) -> pd.DataFrame:
    """
    Two layouts work (one per file):
      Date,Home,Away,Neutral      -- one row per game; Neutral = Y/N
      Date,Team,Opponent,Site     -- one row per team per game, copied from each
                                     team's schedule; Site = H (home), A (away), N (neutral)
    Dates can leave off the year (11-23). The same game entered from both teams'
    sides is counted once; if the two rows disagree about the location, you're warned.
    """
    path = path or os.path.join(SCHEDULE_DIR, f"{gender}_schedule_{season}.csv")
    cols = ["HomeID", "AwayID", "Neutral", "ConfGame", "DayNum"]
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(columns=["Date", "Team", "Opponent", "Site"]).to_csv(path, index=False)
        print(f"  [info] created {path} -- add real games there (Date,Team,Opponent,Site)")
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(path, dtype=str, sep=None, engine="python").dropna(how="all")   # commas or tabs
    raw.columns = [c.strip() for c in raw.columns]
    if raw.empty:
        print(f"  [info] {os.path.basename(path)} is empty -- using a fully generated schedule")
        return pd.DataFrame(columns=cols)

    lookup = name_lookup(data, gender)
    ident = lambda n: find_team(lookup, n)

    if {"Team", "Opponent", "Site"} <= set(raw.columns):          # one row per team per game
        raw["TeamID"] = raw["Team"].map(ident)
        raw["OppID"] = raw["Opponent"].map(ident)
        bad_team = sorted(set(raw.loc[raw["TeamID"].isna(), "Team"].dropna()))
        if bad_team:
            print(f"  [warn] not recognized in the Team column (check the spelling): {', '.join(bad_team)}")
        site = raw["Site"].astype(str).str.strip().str.upper().str[0]
        bad_site = raw.loc[~site.isin(["H", "A", "N"]), "Site"].unique()
        if len(bad_site):
            print(f"  [warn] Site must be H, A or N -- skipped rows with: {', '.join(map(str, bad_site))}")
        raw = raw[site.isin(["H", "A", "N"])].copy()
        site = site[raw.index]
        away = site.eq("A")
        raw["HomeID"] = np.where(away, raw["OppID"], raw["TeamID"])
        raw["AwayID"] = np.where(away, raw["TeamID"], raw["OppID"])
        raw["Neutral"] = site.eq("N")
        unknown_opp = sorted(set(raw.loc[raw["OppID"].isna() & raw["TeamID"].notna(), "Opponent"].dropna()))
    elif {"Home", "Away"} <= set(raw.columns):                      # one row per game
        raw["HomeID"] = raw["Home"].map(ident)
        raw["AwayID"] = raw["Away"].map(ident)
        raw["Neutral"] = raw.get("Neutral", "N").astype(str).str.strip().str.upper().str.startswith("Y")
        unknown_opp = sorted(set(raw.loc[raw["HomeID"].isna(), "Home"].dropna())
                             | set(raw.loc[raw["AwayID"].isna(), "Away"].dropna()))
    else:
        print(f"  [warn] {os.path.basename(path)} needs columns Date,Team,Opponent,Site "
              f"(or Date,Home,Away,Neutral) -- found {list(raw.columns)}")
        return pd.DataFrame(columns=cols)

    if unknown_opp:
        print(f"  [info] skipped games vs {len(unknown_opp)} unrecognized opponents (usually non-Division I): "
              + ", ".join(map(str, unknown_opp[:10])) + (" ..." if len(unknown_opp) > 10 else ""))
    raw = raw.dropna(subset=["HomeID", "AwayID"]).copy()
    raw["HomeID"] = raw["HomeID"].astype(int)
    raw["AwayID"] = raw["AwayID"].astype(int)
    raw["_date"] = pd.to_datetime(pd.Series([_parse_date(d, season) for d in raw["Date"]],
                                            index=raw.index, dtype="object"), errors="coerce")
    bad_dates = raw.loc[raw["_date"].isna(), "Date"].unique()
    if len(bad_dates):
        print(f"  [warn] couldn't read these dates (rows skipped): {', '.join(map(str, bad_dates[:10]))}")
    raw = raw.dropna(subset=["_date"])

    # the same game from both teams' sides: keep it once, warn if they disagree on location
    raw["_pair"] = [tuple(sorted(x)) for x in zip(raw["HomeID"], raw["AwayID"])]
    names = data["teams"].set_index("TeamID")["TeamName"]
    for (date, pair), grp in raw.groupby(["_date", "_pair"]):
        views = set(zip(grp["HomeID"], grp["Neutral"]))
        if len(grp) > 1 and len(views) > 1 and not grp["Neutral"].all():
            print(f"  [warn] {date:%Y-%m-%d} {names.get(pair[0])} vs {names.get(pair[1])}: "
                  f"the two rows disagree on home/away/neutral -- the first row is used")
    raw = raw.drop_duplicates(subset=["_date", "_pair"])

    raw["ConfGame"] = raw["HomeID"].map(conferences).eq(raw["AwayID"].map(conferences)) & \
        raw["HomeID"].map(conferences).notna()
    start = raw["_date"].min()
    raw["DayNum"] = ((raw["_date"] - start).dt.days + 7).astype(int)
    print(f"  [ok] {len(raw)} real games for "
          f"{len(set(raw['HomeID']) | set(raw['AwayID']))} teams")
    return raw[cols].reset_index(drop=True)


def build_schedule(data: dict, conferences: pd.Series, strength: pd.Series, season: int,
                   gender: str = "M", seed: int = 0) -> pd.DataFrame:
    """Real games where you have them, generated games everywhere else."""
    real = read_real_schedule(data, conferences, season, gender)
    return generate_schedule(conferences, strength, season, seed, fixed=real)


# ---------------------------------------------------------------------------
# Comparison: generated vs real for a past season
# ---------------------------------------------------------------------------

def schedule_summary(games: pd.DataFrame, conferences: pd.Series, strength: pd.Series) -> dict:
    teams = conferences.index
    per_team = pd.concat([games["HomeID"], games["AwayID"]]).value_counts().reindex(teams).fillna(0)
    conf = games[games["ConfGame"]]
    conf_per = pd.concat([conf["HomeID"], conf["AwayID"]]).value_counts().reindex(teams).fillna(0)
    opp = pd.concat([
        pd.DataFrame({"team": games["HomeID"], "opp": games["AwayID"]}),
        pd.DataFrame({"team": games["AwayID"], "opp": games["HomeID"]})])
    opp["r"] = opp["opp"].map(strength)
    sos = opp.groupby("team")["r"].mean().reindex(teams)
    top = strength.reindex(teams).nlargest(50).index
    return {
        "Games per team": f"{per_team.mean():.1f}",
        "Conference games per team": f"{conf_per.mean():.1f}",
        "Neutral-site share": f"{games['Neutral'].mean():.1%}",
        "Avg opponent rating, top-50 teams": f"{sos.reindex(top).mean():+.1f}",
        "Avg opponent rating, all teams": f"{sos.mean():+.1f}",
        "Correlation of SOS with team rating": f"{np.corrcoef(sos.fillna(0), strength.reindex(teams).fillna(0))[0, 1]:.2f}",
    }


def check(data: dict, season: int, seed: int = 0):
    from ratings import season_ratings
    g = data["games"]
    g = g[(g["Season"] == season) & ~g["Tourney"] & (g["DayNum"] <= 132)]
    conf = data["conferences"]
    conf = conf[conf["Season"] == season].set_index("TeamID")["ConfAbbrev"]
    strength = season_ratings(data, season)["Rating"]

    # real schedule in the same shape (conference tournament games are excluded
    # by keeping only games between conference members that aren't neutral-site in March)
    home = np.where(g["WLoc"] == "A", g["LTeamID"], g["WTeamID"])
    away = np.where(g["WLoc"] == "A", g["WTeamID"], g["LTeamID"])
    real = pd.DataFrame({"HomeID": home, "AwayID": away, "Neutral": g["WLoc"].eq("N").to_numpy(),
                         "DayNum": g["DayNum"].to_numpy()})
    real["ConfGame"] = real["HomeID"].map(conf).eq(real["AwayID"].map(conf)).to_numpy()
    conf_tourney = real["ConfGame"] & real["Neutral"] & (real["DayNum"] >= 118)
    real = real[~conf_tourney]

    gen = generate_schedule(conf, strength, season, seed)
    a, b = schedule_summary(real, conf, strength), schedule_summary(gen, conf, strength)
    print(f"\nGenerated vs real {season - 1}-{str(season)[2:]} schedule\n")
    print(f"  {'':40} {'Real':>8} {'Generated':>10}")
    for k in a:
        print(f"  {k:40} {a[k]:>8} {b[k]:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=["M", "W"], default="M")
    parser.add_argument("--check", type=int, default=None,
                        help="generate this past season and compare with the real one")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = load_all(args.gender)

    if args.check:
        check(data, args.check, args.seed)
    else:
        from preseason import build_preseason, membership
        season = int(data["games"]["Season"].max()) + 1
        pre, _ = build_preseason(data, season)
        conf = membership(data, season).reindex(pre.index).dropna()
        sched = build_schedule(data, conf, pre["Preseason"], season, args.gender, args.seed)
        os.makedirs(SCHEDULE_DIR, exist_ok=True)
        out = os.path.join(SCHEDULE_DIR, f"{args.gender}_full_schedule_{season}.csv")
        sched.to_csv(out, index=False)
        print(f"\n{season - 1}-{str(season)[2:]} schedule: {len(sched):,} games "
              f"({(sched['Source'] == 'real').sum():,} real, "
              f"{(sched['Source'] == 'generated').sum():,} generated)")
        print(f"Saved to {out}")