"""
rankings.py
Our own rankings, 1-364: this season's ratings next to the preseason ones.
"""
import pandas as pd
import numpy as np
from kaggle_data import load_all
from season import load_inputs
from ratings import update_ratings


data = load_all("M")
season, table, info, conf, sched = load_inputs(data, "M")
played = sched[sched["HomeScore"].notna()]
rating, sd = update_ratings(table["Preseason"], played, info["team_sd"], info["home_court"])

pre = table["Preseason"].reindex(rating.index)
names = data["teams"].set_index("TeamID")["TeamName"]
ranks = pd.DataFrame({
    "Team": names.reindex(rating.index),
    "Conf": conf.reindex(rating.index),
    "Pre": pre.round(1),
    "Now": rating.round(1),
    "PreRank": pre.rank(ascending=False, method="min").astype(int),
    "Rank": rating.rank(ascending=False, method="min").astype(int),
})
ranks["Move"] = ranks["PreRank"] - ranks["Rank"]
ranks = ranks.sort_values("Rank")

home_won = played["HomeScore"] > played["AwayScore"]
winners = pd.Series(np.where(home_won, played["HomeID"], played["AwayID"]))
losers = pd.Series(np.where(home_won, played["AwayID"], played["HomeID"]))
w = winners.value_counts().reindex(ranks.index).fillna(0).astype(int)
l = losers.value_counts().reindex(ranks.index).fillna(0).astype(int)
ranks["Record"] = w.astype(str) + "-" + l.astype(str)
ranks = ranks[["Rank", "Team", "Conf", "Record", "Pre", "Now", "PreRank", "Move"]]

print(ranks.head(25).to_string(index=False))
print("\nBiggest risers since preseason:")
print(ranks.nlargest(10, "Move").to_string(index=False))
print("\nBiggest fallers since preseason:")
print(ranks.nsmallest(10, "Move").to_string(index=False))