"""
kaggle_data.py
Step 1: loads the Kaggle "March Machine Learning Mania" data into clean
tables for the rest of the project.

Put the unzipped Kaggle CSVs in data/kaggle/ (git-ignored -- the
competition rules don't allow republishing the data itself).

Main tables:
    teams        TeamID, TeamName
    games        one row per game: Season, DayNum, WTeamID, LTeamID, WScore,
                 LScore, WLoc, NumOT, plus Tourney (True for NCAA tournament)
    team_games   two rows per game, one from each team's view:
                 Season, DayNum, TeamID, OppID, Pts, OppPts, Margin, Loc
                 (H/A/N for THIS team), Won, NumOT, Tourney
    conferences  Season, TeamID, ConfAbbrev
    seeds        Season, Seed ("W01", "X16a"...), TeamID, SeedNum, Region
    slots        Season, Slot, StrongSeed, WeakSeed  (bracket structure)

Usage:
    from kaggle_data import load_all
    data = load_all()                 # men's
    data = load_all(gender="W")       # women's

    python kaggle_data.py             # prints a summary to check the files
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kaggle")


def _read(name: str, required: bool = True) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"{path} not found -- download the Kaggle March Machine Learning Mania "
                f"data and put the CSVs in data/kaggle/")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_teams(gender: str = "M") -> pd.DataFrame:
    return _read(f"{gender}Teams.csv")[["TeamID", "TeamName"]]


def load_games(gender: str = "M") -> pd.DataFrame:
    """Regular season + NCAA tournament games, one row per game."""
    reg = _read(f"{gender}RegularSeasonCompactResults.csv")
    reg["Tourney"] = False
    tour = _read(f"{gender}NCAATourneyCompactResults.csv", required=False)
    if not tour.empty:
        tour["Tourney"] = True
    games = pd.concat([reg, tour], ignore_index=True)
    return games.sort_values(["Season", "DayNum"]).reset_index(drop=True)


FLIP_LOC = {"H": "A", "A": "H", "N": "N"}


def to_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one from the winner's view, one from the loser's."""
    win = pd.DataFrame({
        "Season": games["Season"], "DayNum": games["DayNum"],
        "TeamID": games["WTeamID"], "OppID": games["LTeamID"],
        "Pts": games["WScore"], "OppPts": games["LScore"],
        "Loc": games["WLoc"], "Won": True,
        "NumOT": games["NumOT"], "Tourney": games["Tourney"],
    })
    lose = pd.DataFrame({
        "Season": games["Season"], "DayNum": games["DayNum"],
        "TeamID": games["LTeamID"], "OppID": games["WTeamID"],
        "Pts": games["LScore"], "OppPts": games["WScore"],
        "Loc": games["WLoc"].map(FLIP_LOC), "Won": False,
        "NumOT": games["NumOT"], "Tourney": games["Tourney"],
    })
    tg = pd.concat([win, lose], ignore_index=True)
    tg["Margin"] = tg["Pts"] - tg["OppPts"]
    return tg.sort_values(["Season", "DayNum", "TeamID"]).reset_index(drop=True)


def load_seeds(gender: str = "M") -> pd.DataFrame:
    seeds = _read(f"{gender}NCAATourneySeeds.csv", required=False)
    if seeds.empty:
        return seeds
    seeds["Region"] = seeds["Seed"].str[0]
    seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)
    return seeds


def load_all(gender: str = "M") -> dict:
    """Every table the project uses, for men ("M") or women ("W")."""
    games = load_games(gender)
    return {
        "teams": load_teams(gender),
        "games": games,
        "team_games": to_team_games(games),
        "conferences": _read(f"{gender}TeamConferences.csv", required=False),
        "seeds": load_seeds(gender),
        "slots": _read(f"{gender}NCAATourneySlots.csv", required=False),
    }


def summarize(data: dict):
    games = data["games"]
    reg = games[~games["Tourney"]]
    tour = games[games["Tourney"]]
    last = int(games["Season"].max())
    print(f"Seasons:            {int(games['Season'].min())}-{last}")
    print(f"Teams:              {len(data['teams'])}")
    print(f"Regular season:     {len(reg):,} games")
    print(f"NCAA tournament:    {len(tour):,} games")
    print(f"Season {last}:        {len(reg[reg['Season'] == last]):,} regular season games, "
          f"{len(tour[tour['Season'] == last])} tournament games")
    seeds = data["seeds"]
    if not seeds.empty:
        print(f"Seeds available through {int(seeds['Season'].max())}")
    conf = data["conferences"]
    if not conf.empty:
        latest = conf[conf["Season"] == conf["Season"].max()]
        print(f"Conferences ({int(conf['Season'].max())}): {latest['ConfAbbrev'].nunique()}, "
              f"{len(latest)} teams")
    print(f"Home/away/neutral (winner's view): {reg['WLoc'].value_counts().to_dict()}")


if __name__ == "__main__":
    summarize(load_all())