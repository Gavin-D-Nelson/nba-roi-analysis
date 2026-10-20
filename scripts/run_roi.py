"""Compute player ROI from the raw data and save the results workbook.

Usage:
    python scripts/run_roi.py

Reads data/raw/*.xlsx and writes data/processed/player_roi_2024-25.xlsx.
"""

from nba_roi import config
from nba_roi.roi import load_raw_data, run_roi_model, write_workbook


def main():
    logs, adv, sal = load_raw_data()
    print(f"Game logs: {len(logs):,} rows | Advanced: {len(adv)} players | Salaries: {len(sal)} rows")

    run = run_roi_model(logs, adv, sal)
    p, results = run.params, run.results

    print(f"Players matched to salary: {len(run.player_season)} "
          f"(no salary match: {len(run.unmatched)})")
    print(f"SGV per game: ${p['sgv']:,.0f} | m_bar: {p['m_bar']:.2f} | "
          f"replacement share: {p['replacement_share']:.4%}")

    cols = ["PLAYER_NAME", "Team", "SALARY_CLEAN", "GAMES_PLAYED", "ROI"]
    print("\n=== TOP 10 ROI ===")
    print(results[cols].head(10).to_string(index=False))
    print("\n=== BOTTOM 10 ROI ===")
    print(results[cols].tail(10).to_string(index=False))

    print("\n=== SUMMARY ===")
    print(f"Players analysed: {len(results)}")
    print(f"Mean ROI:   {results['ROI'].mean():.3f}x")
    print(f"Median ROI: {results['ROI'].median():.3f}x")
    print(f"Above break-even (ROI > 1.0): {(results['ROI'] > 1.0).sum()}")
    print(f"Below break-even (ROI < 1.0): {(results['ROI'] < 1.0).sum()}")

    path = write_workbook(run)
    print(f"\nSaved: {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
