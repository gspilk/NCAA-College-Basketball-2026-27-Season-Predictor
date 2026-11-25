# NCAA College Basketball 2026–27 Season Predictor

> **Status: in season.** The rating, preseason, bracket, and simulation code is built and validated. The men's projections now update with real 2026–27 results as games are played; the women's schedule is still being entered.

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
| 7 | `postseason.py` | Plays conference tournaments for automatic bids, models the selection committee from past at-large decisions, seeds the 76-team field, and simulates the NCAA Tournament. |

During the season, a few more scripts use the results entered so far (men's for now):

| Script | What it does |
|---|---|
| `standings.py` | Current records from games already played, next to the model's projections. |
| `rankings.py` | Updates every team's rating with this season's results and ranks all D-I teams, showing how far each has moved since the preseason. |
| `conf_leaders.py` | The projected favorite and runner-up in every conference, plus the likeliest automatic-bid winner. |
| `run_all.py` | Runs the season, postseason, conference leaders, and rankings steps in order. |

## 2026–27 projections

Each table comes from 10,000 simulated seasons. **Rating** is the preseason rating: points better than an average D-I team on a neutral floor. Projected records include conference tournament games but not the NCAA Tournament.

### Men's, updated with results through November 23, 2026

| Team | Conf | Rating | Proj. record | Make tourney | Final Four | Title |
|:--|:--|--:|:--|--:|--:|--:|
| Illinois | Big Ten | 19.0 | 32-5 | 100% | 50.3% | 24.6% |
| Houston | Big 12 | 20.4 | 30-5 | 100% | 34.0% | 12.1% |
| Duke | ACC | 22.4 | 27-9 | 99% | 18.0% | 4.7% |
| Florida | SEC | 20.6 | 26-8 | 99% | 15.6% | 3.9% |
| Kansas | Big 12 | 16.1 | 26-9 | 99% | 15.9% | 3.8% |
| Arizona | Big 12 | 21.4 | 23-9 | 97% | 14.4% | 3.8% |
| UCLA | Big Ten | 13.7 | 27-8 | 99% | 15.6% | 3.8% |
| Connecticut | Big East | 16.5 | 26-9 | 97% | 13.4% | 3.5% |
| Texas Tech | Big 12 | 16.8 | 24-8 | 97% | 14.7% | 3.4% |
| St John's | Big East | 16.0 | 26-7 | 97% | 13.1% | 2.9% |

### Men's preseason (before any games)

| Team | Conf | Rating | Proj. record | Make tourney | Final Four | Title |
|:--|:--|--:|:--|--:|--:|--:|
| Duke | ACC | 22.4 | 29-7 | 98% | 28.6% | 10.4% |
| Arizona | Big 12 | 21.4 | 25-8 | 96% | 23.5% | 8.1% |
| Florida | SEC | 20.6 | 25-8 | 96% | 20.7% | 6.3% |
| Michigan | Big Ten | 20.5 | 26-8 | 95% | 19.7% | 6.0% |
| Houston | Big 12 | 20.3 | 25-8 | 95% | 19.5% | 5.9% |
| Iowa St | Big 12 | 20.0 | 25-8 | 95% | 18.7% | 5.3% |
| Purdue | Big Ten | 18.8 | 26-10 | 91% | 14.2% | 4.1% |
| Illinois | Big Ten | 19.0 | 26-10 | 92% | 14.8% | 4.1% |
| Alabama | SEC | 18.0 | 22-11 | 87% | 12.1% | 3.3% |
| Tennessee | SEC | 17.8 | 22-10 | 87% | 11.4% | 3.1% |

Illinois's title odds rose from 4.1% to 24.6% after its early-season results, while Duke's fell from 10.4% to 4.7%.

### Women's preseason

| Team | Conf | Rating | Proj. record | Make tourney | Final Four | Title |
|:--|:--|--:|:--|--:|--:|--:|
| South Carolina | SEC | 39.2 | 28-5 | 100% | 51.8% | 19.7% |
| Connecticut | Big East | 38.1 | 33-1 | 100% | 49.6% | 15.2% |
| Texas | SEC | 37.8 | 28-5 | 100% | 46.2% | 14.8% |
| UCLA | Big Ten | 37.6 | 30-4 | 100% | 45.8% | 14.7% |
| LSU | SEC | 37.2 | 28-5 | 100% | 42.7% | 13.1% |
| Michigan | Big Ten | 30.2 | 26-8 | 98% | 16.1% | 2.9% |
| Oklahoma | SEC | 29.0 | 24-8 | 98% | 12.2% | 2.2% |
| Vanderbilt | SEC | 28.3 | 20-12 | 94% | 9.4% | 1.6% |
| Iowa | Big Ten | 27.0 | 24-10 | 95% | 8.1% | 1.4% |
| Ohio St | Big Ten | 27.1 | 24-10 | 94% | 8.1% | 1.3% |

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
python calendar_view.py            # build the season calendar (HTML) with win probabilities
python standings.py                # current records next to projections
python rankings.py                 # in-season ratings and rankings
python conf_leaders.py             # projected conference favorites
python run_all.py                  # season, postseason, conference leaders, and rankings in one go
```

Add `--gender W` to the step scripts for the women's season. The in-season scripts (`standings.py`, `rankings.py`, `conf_leaders.py`, `run_all.py`) cover the men's season for now.

## Data and sources

- Historical results, teams, conferences, and brackets come from Kaggle's March Machine Learning Mania competition. The competition rules don't allow republishing the data, so the `data/` folder is git-ignored and not included here.
- The 2026–27 schedules are entered by hand from conference and school athletics schedules.
- `conference_changes.csv` lists teams changing conferences for 2026–27.

## Roadmap

- [x] Update the tournament to the 2027 format of 76 teams with an expanded opening round
- [x] Publish preseason projections
- [x] Update the men's projections with real results during the season
- [ ] Finish entering the women's 2026–27 schedule
- [ ] Extend the in-season scripts to the women's season
- [ ] Compare the final projections with how the season actually played out