"""Download the raw data: ESPN salaries + NBA Stats API game logs and tracking stats.

Usage:
    python scripts/fetch_data.py               # everything (~10 minutes)
    python scripts/fetch_data.py --salaries    # ESPN salaries only
    python scripts/fetch_data.py --stats       # NBA API stats only

Output goes to data/raw/.
"""

import argparse

from nba_roi import collect_stats, scrape_salaries


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salaries", action="store_true", help="only scrape ESPN salaries")
    parser.add_argument("--stats", action="store_true", help="only pull NBA API stats")
    args = parser.parse_args()
    run_all = not (args.salaries or args.stats)

    if run_all or args.salaries:
        print("── ESPN salaries ──")
        scrape_salaries.main()
    if run_all or args.stats:
        print("── NBA Stats API ──")
        collect_stats.main()


if __name__ == "__main__":
    main()
