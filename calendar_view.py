"""
calendar_view.py
Turns your typed schedule into a season calendar, like the schedule pages on
ESPN or a school site, built entirely from your own files: no web requests.

It reads data/schedule/M_schedule_2027.csv (the games you typed), adds the
model's preseason prediction for every game, and writes one self-contained
web page:

    data/calendar/M_calendar_2027.html

Open it in any browser to:
  - flip through the season month by month and see every game on a day
  - pick a team to see its full schedule, its win chance in each game,
    and its projected record
  - filter to one conference
  - download a team's schedule as a .ics file for Google/Apple/Outlook calendars

Only games from your typed file appear, because the generated fill-in games
(schedule.py) have no real dates.

Usage:
    python calendar_view.py
    python calendar_view.py --gender W
"""

import argparse
import json
import os
from math import erf, sqrt

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data", "calendar")
TEMPLATE_MARK = "/*__GAMES__*/null"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def game_table(real: pd.DataFrame, start: pd.Timestamp, rating: pd.Series, conf: pd.Series,
               names: pd.Series, home_court: float, sd: float) -> pd.DataFrame:
    """One row per dated game, with the home team's win chance and the point spread."""
    g = real.copy()
    g["Date"] = (start + pd.to_timedelta(g["DayNum"] - 7, unit="D")).dt.strftime("%Y-%m-%d")
    rh = g["HomeID"].map(rating).fillna(rating.min())
    ra = g["AwayID"].map(rating).fillna(rating.min())
    margin = rh - ra + home_court * (~g["Neutral"].astype(bool))
    g["Spread"] = margin.round(1)                       # positive = home team favored
    g["PHome"] = [round(_norm_cdf(m / sd), 3) for m in margin]
    g["Home"] = g["HomeID"].map(names)
    g["Away"] = g["AwayID"].map(names)
    g["HomeConf"] = g["HomeID"].map(conf).fillna("")
    g["AwayConf"] = g["AwayID"].map(conf).fillna("")
    return g.sort_values(["Date", "Home"])[
        ["Date", "Home", "Away", "Neutral", "ConfGame", "HomeConf", "AwayConf", "Spread", "PHome"]]


def write_html(games: pd.DataFrame, ratings: dict, title: str, path: str):
    payload = {
        "title": title,
        "games": [[r.Date, r.Home, r.Away, int(bool(r.Neutral)), int(bool(r.ConfGame)),
                   r.HomeConf, r.AwayConf, float(r.Spread), float(r.PHome)]
                  for r in games.itertuples()],
        "ratings": ratings,
    }
    with open(os.path.join(ROOT, "calendar_template.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace(TEMPLATE_MARK, json.dumps(payload, separators=(",", ":")))
    html = html.replace("<title>Season Calendar</title>", f"<title>{title}</title>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    from kaggle_data import load_all
    from season import load_inputs
    from schedule import read_real_schedule, _parse_date, SCHEDULE_DIR
    from ratings import GAME_SD

    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=["M", "W"], default="M")
    args = parser.parse_args()

    data = load_all(args.gender)
    season, table, info, conf, _ = load_inputs(data, args.gender)
    real = read_real_schedule(data, conf, season, args.gender)
    if real.empty:
        raise SystemExit("No real games yet -- type some into data/schedule/ first.")

    # read_real_schedule numbers days from the first date in the file (that day = DayNum 7)
    raw = pd.read_csv(os.path.join(SCHEDULE_DIR, f"{args.gender}_schedule_{season}.csv"),
                      dtype=str, sep=None, engine="python").dropna(how="all")
    start = min(d for d in (_parse_date(x, season) for x in raw["Date"]) if pd.notna(d))

    # one game's uncertainty = game-to-game noise + how far off each preseason rating might be
    team_sd = info.get("team_sd", 0.0)
    sd = sqrt(GAME_SD ** 2 + 2 * team_sd ** 2)
    names = data["teams"].set_index("TeamID")["TeamName"]
    games = game_table(real, start, table["Preseason"], conf, names, info["home_court"], sd)

    ratings = {names[t]: [round(float(table.at[t, "Preseason"]), 2), str(conf.get(t, ""))]
               for t in table.index if t in names.index}
    label = "Men's" if args.gender == "M" else "Women's"
    title = f"{label} {season - 1}-{str(season)[2:]} Season Calendar"
    out = os.path.join(OUT_DIR, f"{args.gender}_calendar_{season}.html")
    write_html(games, ratings, title, out)
    print(f"\n{len(games):,} games from {games['Date'].min()} to {games['Date'].max()}")
    print(f"Saved to {out} -- open it in your browser")