"""
conf_leaders.py
The projected winner of every conference.
"""
import os
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
proj = pd.read_csv(os.path.join(root, "data", "sims", "M_season_2027.csv"), index_col="TeamID")

ranked = proj.sort_values("ConfTitle", ascending=False)
ranked["Rank"] = ranked.groupby("Conf").cumcount() + 1
first = ranked[ranked["Rank"] == 1].set_index("Conf")
second = ranked[ranked["Rank"] == 2].set_index("Conf")

out = pd.DataFrame({
    "Favorite": first["TeamName"],
    "Fav %": (first["ConfTitle"] * 100).round().astype(int).astype(str) + "%",
    "Runner-up": second["TeamName"],
    "R-up %": (second["ConfTitle"] * 100).round().astype(int).astype(str) + "%",
}).sort_index()
print(out.to_string())
print(len(out), "conferences")