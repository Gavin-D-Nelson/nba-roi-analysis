"""Scrape player salaries from ESPN's NBA salary pages."""

import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config

ESPN_SALARY_URL = "https://www.espn.com/nba/salaries/_/year/{year}/page/{page}"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def parse_salary_rows(html):
    """Extract (player, team, salary) tuples from one ESPN salary page.

    Player rows are the table rows with exactly 4 cells: rank, name, team,
    salary. Values are kept as raw strings; cleaning happens later.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) == 4:  # only player rows
            rows.append((cols[1].text.strip(), cols[2].text.strip(), cols[3].text.strip()))
    return rows


def scrape_espn_salaries(year=config.ESPN_SALARY_YEAR, pages=config.ESPN_SALARY_PAGES,
                         pause=1.0, verbose=True):
    """Scrape all salary pages and return a raw Player/Team/Salary DataFrame."""
    rows = []
    for page in range(1, pages + 1):
        if verbose:
            print(f"Scraping page {page}...")
        response = requests.get(ESPN_SALARY_URL.format(year=year, page=page),
                                headers=HEADERS, timeout=30)
        response.raise_for_status()
        rows.extend(parse_salary_rows(response.text))
        time.sleep(pause)  # polite; avoids getting blocked
    return pd.DataFrame(rows, columns=["Player", "Team", "Salary"])


def main(output_file=config.SALARIES_FILE):
    df = scrape_espn_salaries()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_file, index=False)  # raw, no conversion
    print(f"Saved {len(df)} rows -> {output_file}")
    return df


if __name__ == "__main__":
    main()
