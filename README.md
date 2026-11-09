# NCAA College Basketball 2026–27 Season Predictor

  > **Status: work in progress.** The rating, preseason, bracket, and simulation code is built and validated. I'm entering the 2026–27 schedules conference by conference; next is tracking the projections against real results during the season.

A Python project that projects the 2026–27 NCAA Division I men's and women's basketball seasons, from preseason team ratings through the regular season, conference tournaments, tournament selection, and the NCAA Tournament. Every step runs inside the same simulated season, so a team that's better than expected in November stays better in March, the way real seasons work.

## How it works

The pipeline runs in seven steps, one script each:

| Step | Script | What it does |
|---|---|---|
| 1 | `kaggle_data.py` | Loads and cleans the historical game, team, conference, and bracket data. |
| 2 | `ratings.py` | Rates every team as points better or worse than an average D-I team on a neutral floor, using ridge-regularized least squares on game margins with an estimated home-court edge. Converts rating gaps to win probabilities. |
| 3 | `bracket.py` | Simulates an NCAA Tournament bracket thousands of times and reports each team's odds of reaching every round. |
| 4 | `preseason.py` | Builds preseason ratings from the past three seasons plus the average strength of each team's new conference, which handles realignment. The weights are fit on history rather than chosen by hand. |
| 5 | `schedule.py` | Builds the season schedule from real schedules I've entered, filling any gaps with generated games based on conference membership. |
| 6 | `season.py` | Simulates the regular season thousands of times and produces projected records and conference standings. |
| 7 | `postseason.py` | Plays conference tournaments for automatic bids, models the selection committee from past at-large decisions, seeds the field, and simulates the NCAA Tournament. |

## Validation

Each part of the model is tested against seasons it didn't see:

- **Ratings:** For each past season, teams are rated on regular-season games only, then used to predict that season's NCAA Tournament. The script reports accuracy, Brier score, log loss, and calibration, compared with a "better seed wins" baseline.
- **Preseason ratings:** Leave-one-season-out testing. Each season's preseason ratings are built from weights fit on the *other* seasons, then compared with "same as last season" and "every team equal" baselines.
- **Season simulation:** A backtest builds 2025–26 preseason ratings using only earlier seasons, plays the real 2025–26 schedule, and checks projected wins against what happened, including whether about 80% of teams land inside their 80% win range.
- **Schedule generator:** A generated 2025–26 schedule is compared with the real one.
- **Selection committee:** The at-large selection model is fit on 2012–2026 and checked against which teams actually received bids.

## Setup

1. Install the requirements:

   ```
   pip install -r requirements.txt
   ```

2. Download the [Kaggle March Machine Learning Mania](https://www.kaggle.com/competitions) competition data and put the unzipped CSVs in `data/kaggle/`.

3. For the real 2026–27 schedule, add schedule files to `data/schedule/` (format described at the top of `schedule.py`). Without them, the project generates a realistic schedule from conference memberships.

## Usage

```
python kaggle_data.py              # check that the data files load
python ratings.py                  # validate the ratings, show the top 25
python preseason.py                # preseason ratings for 2026-27
python schedule.py                 # build the 2026-27 schedule
python season.py                   # regular-season projections
python season.py --backtest 2026   # test the simulator on 2025-26
python postseason.py               # conference tournaments + NCAA Tournament odds
```

Add `--gender W` to any script for the women's season.

## Data and sources

- Historical results, teams, conferences, and brackets come from Kaggle's March Machine Learning Mania competition. The competition rules don't allow republishing the data, so the `data/` folder is git-ignored and not included here.
- The 2026–27 schedules are entered by hand from conference and school athletics schedules.
- `conference_changes.csv` lists teams changing conferences for 2026–27.

## Roadmap

- [ ] Finish entering 2026–27 schedules
- [ ] Update the tournament to the 2027 format of 76 teams with an expanded opening round (currently built for 68 teams and the First Four)
- [ ] Publish preseason projections before the season tips off
- [ ] Track the projections against real results during the season
