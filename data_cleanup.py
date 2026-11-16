# inspo: https://www.geeksforgeeks.org/python/how-to-count-the-number-of-lines-in-a-csv-file-in-python/
import pandas as pd
from schedule import _parse_date

df = pd.read_csv('NCAA-College-Basketball-2026-27-Season-Predictor/data/schedule/M_schedule_2027.csv', sep=None, engine="python")
df["Home"] = df["Home"].str.strip()
df["Away"] = df["Away"].str.strip()
df = df.replace("St. John's", "St John's")
df["Date"] = df["Date"].apply(lambda d: _parse_date(d, 2027))
df["Pair"] = df.apply(lambda r: tuple(sorted([r["Home"], r["Away"]])), axis=1)
print(df[["Home", "Away", "Pair"]].head())
print(len(df))
print(df["Date"].head())
df = df.drop_duplicates(subset=["Date", "Pair"])
print("rows after:", len(df))
df = df.sort_values("Date")
df = df.drop(columns=["Pair"])
df.to_csv("NCAA-College-Basketball-2026-27-Season-Predictor/data/schedule/M_schedule_2027_clean.csv", index=False)
print("saved!")