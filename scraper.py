"""
Step 1: Scraper for Sports-Reference College Basketball.

Pulls last season's team ratings and every team's game results, and caches
everything locally so you only ever hit the site once per page.

Usage:
    python scraper.py                  # men's, 2026 season (i.e. 2025-26)
    python scraper.py --gender women
    python scraper.py --season 2025

Install first:
    pip install requests pandas lxml
"""

import argparse
import html as html_lib
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.sports-reference.com/cbb"
DELAY_SECONDS = 4  # Sports-Reference blocks fast scrapers (~20 requests/min). Don't lower this.
HEADERS = {"User-Agent": "Mozilla/5.0 (personal NCAA season simulator project)"}

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "html_cache"


# ---------------------------------------------------------------------------
# Fetching with a local cache
# ---------------------------------------------------------------------------

def fetch(url: str) -> str:
    """Return a page's HTML, using the local cache when possible."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE, "")) + ".html")

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    print(f"  fetching {url}")
    time.sleep(DELAY_SECONDS)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 429:
        raise RuntimeError("Rate limited (429). Wait an hour before trying again.")
    resp.raise_for_status()

    cache_file.write_text(resp.text, encoding="utf-8")
    return resp.text


def read_table(html: str, table_id: str) -> pd.DataFrame:
    """
    Pull one table out of a Sports-Reference page by its HTML id.
    Sports-Reference hides some tables inside HTML comments, so strip those first.
    """
    html = html.replace("<!--", "").replace("-->", "")
    tables = pd.read_html(StringIO(html), attrs={"id": table_id})
    df = tables[0]

    # Flatten two-row headers like ("Overall", "W") -> "Overall_W"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(p) for p in col if not str(p).startswith("Unnamed")).strip("_")
            for col in df.columns
        ]

    # Drop the repeated header rows that appear every ~20 rows
    first_col = df.columns[0]
    df = df[df[first_col].astype(str) != first_col]
    return df.reset_index(drop=True)


def school_slugs(html: str, gender: str) -> dict:
    """Map each school's display name to its URL slug, e.g. 'Duke' -> 'duke'."""
    pattern = rf'href="/cbb/schools/([^/]+)/{gender}/\d{{4}}\.html">([^<]+)</a>'
    # html.unescape turns "Texas A&amp;M" into "Texas A&M" so it matches the table
    return {html_lib.unescape(name): slug for slug, name in re.findall(pattern, html)}


# ---------------------------------------------------------------------------
# What we scrape
# ---------------------------------------------------------------------------

def scrape_ratings(season: int, gender: str) -> pd.DataFrame:
    """Team ratings page: conference, record, SRS, SOS, adjusted ORtg/DRtg."""
    url = f"{BASE}/seasons/{gender}/{season}-ratings.html"
    html = fetch(url)
    df = read_table(html, "ratings")

    slugs = school_slugs(html, gender)
    df["slug"] = df["School"].map(slugs)
    df = df.dropna(subset=["School"])
    return df


def scrape_team_games(slug: str, season: int, gender: str) -> pd.DataFrame:
    """One team's full schedule and results."""
    url = f"{BASE}/schools/{slug}/{gender}/{season}-schedule.html"
    html = fetch(url)
    df = read_table(html, "schedule")
    df["team_slug"] = slug
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026,
                        help="Season END year (2026 = the 2025-26 season)")
    parser.add_argument("--gender", choices=["men", "women"], default="men")
    parser.add_argument("--skip-games", action="store_true",
                        help="Only grab team ratings (1 request instead of ~365)")
    args = parser.parse_args()

    out_dir = DATA_DIR / args.gender / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scraping {args.gender}'s {args.season - 1}-{str(args.season)[2:]} season")

    # 1. Team ratings (one page)
    ratings = scrape_ratings(args.season, args.gender)
    ratings.to_csv(out_dir / "team_ratings.csv", index=False)
    print(f"Saved ratings for {len(ratings)} teams")

    if args.skip_games:
        return

    # 2. Every team's game log (~365 pages; about 25 minutes the first time,
    #    instant after that because of the cache)
    all_games = []
    slugs = ratings["slug"].dropna().unique()
    for i, slug in enumerate(slugs, 1):
        print(f"[{i}/{len(slugs)}] {slug}")
        try:
            all_games.append(scrape_team_games(slug, args.season, args.gender))
        except Exception as e:
            print(f"  skipped {slug}: {e}")

    games = pd.concat(all_games, ignore_index=True)
    games.to_csv(out_dir / "games_raw.csv", index=False)
    print(f"Saved {len(games)} game rows to {out_dir / 'games_raw.csv'}")


if __name__ == "__main__":
    main()