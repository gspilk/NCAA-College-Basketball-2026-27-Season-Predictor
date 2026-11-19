"""
standings.py
Current standings from games already played, next to the model's projections.
"""
import os
from kaggle_data import load_all
from preseason import membership
from schedule import read_real_schedule
import numpy as np
import pandas as pd

data = load_all("M")
season = int(data["games"]["Season"].max()) +1
conf = membership(data, season)
games = read_real_schedule(data, conf, season, "M")

played = games[games["HomeScore"].notna()]
print(len(played), "games played so far")

home_won = played["HomeScore"] > played["AwayScore"]
winners = pd.Series(np.where(home_won, played["HomeID"], played["AwayID"]))
losers = pd.Series(np.where(home_won, played["AwayID"], played["HomeID"]))

names = data["teams"].set_index("TeamID")["TeamName"]
table = pd.DataFrame({
    "Team": names.reindex(conf.index),
    "Conf": conf,
    "W": winners.value_counts().reindex(conf.index).fillna(0).astype(int),
    "L": losers.value_counts().reindex(conf.index).fillna(0).astype(int),
})
cg = played[played["ConfGame"]]
c_home_won = cg["HomeScore"] > cg["AwayScore"]
c_winners = pd.Series(np.where(c_home_won, cg["HomeID"], cg["AwayID"]))
c_losers = pd.Series(np.where(c_home_won, cg["AwayID"], cg["HomeID"]))
table["CW"] = c_winners.value_counts().reindex(conf.index).fillna(0).astype(int)
table["CL"] = c_losers.value_counts().reindex(conf.index).fillna(0).astype(int)

root = os.path.dirname(os.path.abspath(__file__))
proj = pd.read_csv(os.path.join(root, "data", "sims", "M_season_2027.csv"), index_col="TeamID")
table["Proj"] = proj["W"].round(1).astype(str) + "-" + proj["L"].round(1).astype(str)
table["Title"] = (proj["ConfTitle"] * 100).round().astype(int).astype(str) + "%"

table = table.sort_values(["Conf", "CW", "CL", "W", "L"], ascending=[True, False, True, False, True])
for c, grp in table.groupby("Conf"):
    print(f"\n{c}")
    print(grp[["Team", "W", "L", "CW", "CL", "Proj", "Title"]].to_string(index=False))