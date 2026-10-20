"""Tests for the ROI formulas and data-cleaning helpers."""

import math

import numpy as np
import pandas as pd
import pytest

from nba_roi import config
from nba_roi.cleaning import add_shooting_splits, clean_salaries, normalize_name
from nba_roi.roi import (
    average_players_per_game,
    compute_delta,
    compute_roi,
    compute_wrms,
    run_roi_model,
    single_game_value,
    summarize_players,
)
from nba_roi.scrape_salaries import parse_salary_rows


# ── Name matching ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("Nikola Jokić", "NIKOLA JOKIC"),
    ("Nikola Jokic, C", "NIKOLA JOKIC"),
    ("  Luka   Dončić, PG ", "LUKA DONCIC"),
    ("Karl-Anthony Towns, F", "KARL-ANTHONY TOWNS"),
    ("De'Aaron Fox", "DE'AARON FOX"),
    (None, ""),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_espn_and_nba_api_names_match():
    assert normalize_name("Jusuf Nurkić") == normalize_name("Jusuf Nurkic, C")


# ── Salary cleaning ───────────────────────────────────────────────────────────

def test_clean_salaries_parses_drops_headers_and_dedupes():
    sal = pd.DataFrame({
        "Player": ["NAME", "Stephen Curry, G", "Stephen Curry, G", "Rookie Guy, F"],
        "Team": ["TEAM", "Golden State Warriors", "Golden State Warriors", "Utah Jazz"],
        "Salary": ["SALARY", "$55,761,216", "$1,000,000", "$1,157,153"],
    })
    out = clean_salaries(sal)
    assert len(out) == 2  # header row dropped, duplicate Curry removed
    curry = out.loc[out["PLAYER_NAME_CLEAN"] == "STEPHEN CURRY", "SALARY_CLEAN"]
    assert curry.item() == 55_761_216  # highest salary kept


# ── Shooting splits ───────────────────────────────────────────────────────────

def test_add_shooting_splits():
    df = pd.DataFrame({"FGM": [8], "FGA": [17], "FG3M": [3], "FG3A": [7], "FTM": [4], "FTA": [5]})
    add_shooting_splits(df)
    row = df.iloc[0]
    assert row["FG2O"] == 5   # 8 made - 3 threes
    assert row["FG2A"] == 10  # 17 - 7
    assert row["FG2X"] == 5   # 10 attempted - 5 made
    assert row["FG3O"] == 3
    assert row["FG3X"] == 4
    assert row["FTMO"] == 4
    assert row["FTMX"] == 1


# ── Performance score (Delta) ─────────────────────────────────────────────────

def test_delta_is_weighted_sum_of_centered_stats():
    df = pd.DataFrame({"A": [1.0, 3.0], "B": [10.0, 20.0]})
    beta = {"A": 2.0, "B": -0.5}
    delta, means = compute_delta(df, beta)
    assert means == {"A": 2.0, "B": 15.0}
    # row 0: 2*(1-2) + -0.5*(10-15) = -2 + 2.5 = 0.5
    assert delta.tolist() == pytest.approx([0.5, -0.5])


def test_delta_average_game_scores_zero():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({v: rng.poisson(3, 500) for v in config.BETA})
    delta, _ = compute_delta(df, config.BETA)
    assert delta.mean() == pytest.approx(0, abs=1e-10)


def test_delta_skips_missing_variables():
    df = pd.DataFrame({"A": [1.0, 3.0]})
    delta, means = compute_delta(df, {"A": 1.0, "NOT_THERE": 5.0})
    assert list(means) == ["A"]
    assert delta.tolist() == pytest.approx([-1.0, 1.0])


# ── WRMS ──────────────────────────────────────────────────────────────────────

def test_wrms_mean_and_std_equal_replacement_share():
    rng = np.random.default_rng(1)
    delta = pd.Series(rng.normal(0, 1.4, 2000))
    m_bar = 21.39
    wrms, d_bar, d_std = compute_wrms(delta, m_bar)
    assert wrms.mean() == pytest.approx(1 / m_bar)
    assert wrms.std(ddof=1) == pytest.approx(1 / m_bar)
    assert d_bar == pytest.approx(delta.mean())


def test_wrms_average_performance_earns_replacement_share():
    delta = pd.Series([-1.0, 0.0, 1.0])
    wrms, _, _ = compute_wrms(delta, m_bar=20)
    assert wrms[1] == pytest.approx(1 / 20)
    assert wrms[0] < 1 / 20 < wrms[2]


def test_average_players_per_game():
    df = pd.DataFrame({"Game_ID": [1, 1, 1, 2, 2], "PLAYER_ID": [10, 11, 12, 10, 13]})
    assert average_players_per_game(df) == 2.5


# ── SGV, cash flow, ROI ───────────────────────────────────────────────────────

def test_single_game_value_uses_team_games():
    assert single_game_value(2_460_000, n_teams=30, games_per_season=82) == 1_000


def test_roi_ratio():
    assert compute_roi(15_000_000, 10_000_000) == 1.5
    assert compute_roi(-1_000_000, 10_000_000) == -0.1


def test_missed_games_count_against_player():
    logs = pd.DataFrame({
        "PLAYER_ID": [1] * 70, "PLAYER_NAME": ["A"] * 70, "PLAYER_NAME_CLEAN": ["A"] * 70,
        "CF": [100.0] * 70, "DELTA": [0.0] * 70, "WRMS": [0.05] * 70,
    })
    season = summarize_players(logs, games_per_season=82)
    assert season.loc[0, "GAMES_PLAYED"] == 70
    assert season.loc[0, "GAMES_MISSED"] == 12
    assert season.loc[0, "TOTAL_CF"] == 7_000


# ── ESPN scraper parsing ──────────────────────────────────────────────────────

def test_parse_salary_rows_keeps_only_four_cell_rows():
    html = """
    <table>
      <tr><td colspan="4">NBA Player Salaries</td></tr>
      <tr><td>RK</td><td>NAME</td><td>TEAM</td><td>SALARY</td></tr>
      <tr><td>1</td><td>Stephen Curry, G</td><td>Golden State Warriors</td><td>$55,761,216</td></tr>
      <tr><td>2</td><td>Joel Embiid, C</td><td>Philadelphia 76ers</td><td>$51,415,938</td></tr>
    </table>"""
    rows = parse_salary_rows(html)
    assert rows[1] == ("Stephen Curry, G", "Golden State Warriors", "$55,761,216")
    assert len(rows) == 3  # header row is kept here and removed later by clean_salaries


# ── End-to-end on synthetic data ──────────────────────────────────────────────

def _synthetic_inputs(n_games=40, players_per_team=5, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        for p in range(players_per_team * 2):
            fga, fg3a, fta = rng.integers(5, 20), rng.integers(0, 8), rng.integers(0, 8)
            fg3m = rng.integers(0, fg3a + 1)
            fgm = fg3m + rng.integers(0, fga - fg3a + 1) if fga > fg3a else fg3m
            rows.append({
                "Game_ID": g, "PLAYER_ID": p, "PLAYER_NAME": f"Player {p}",
                "GAME_DATE": f"G{g}", "MATCHUP": "X vs. Y", "WL": "W", "MIN": 30,
                "FGM": fgm, "FGA": max(fga, fg3a), "FG3M": fg3m, "FG3A": fg3a,
                "FTM": rng.integers(0, fta + 1), "FTA": fta,
                "PF": rng.integers(0, 5), "STL": rng.integers(0, 3), "BLK": rng.integers(0, 3),
                "TOV": rng.integers(0, 4), "PTS": 0, "AST": 0, "REB": 0,
            })
    logs = pd.DataFrame(rows)
    adv = pd.DataFrame({"PLAYER_ID": range(players_per_team * 2),
                        "PLAYER_NAME": [f"Player {p}" for p in range(players_per_team * 2)]})
    for v in config.ADVANCED_VARS:
        adv[v] = rng.uniform(0, 3, len(adv))
    sal = pd.DataFrame({
        "Player": [f"Player {p}, G" for p in range(players_per_team * 2)],
        "Team": ["Team"] * (players_per_team * 2),
        "Salary": [f"${s:,}" for s in rng.integers(1_000_000, 40_000_000, players_per_team * 2)],
    })
    return logs, adv, sal


def test_pipeline_end_to_end():
    logs, adv, sal = _synthetic_inputs()
    run = run_roi_model(logs, adv, sal)
    p = run.params

    assert len(run.results) == 10
    assert p["m_bar"] == 10
    assert run.logs["WRMS"].mean() == pytest.approx(1 / p["m_bar"])
    # SGV is defined per team-game, while WRMS shares sum to 1 per game, so the
    # cash flows distributed equal SGV x number of games (see README limitations).
    assert run.logs["CF"].sum() == pytest.approx(p["sgv"] * logs["Game_ID"].nunique())
    # ROI column is exactly Total_CF / Salary and is sorted descending
    assert np.allclose(run.results["ROI"], run.results["TOTAL_CF"] / run.results["SALARY_CLEAN"])
    assert run.results["ROI"].is_monotonic_decreasing


# ── Regression check against the published 2024-25 results ────────────────────

raw_files_present = all(p.exists() for p in
                        (config.GAMELOGS_FILE, config.ADVANCED_FILE, config.SALARIES_FILE))


@pytest.mark.skipif(not raw_files_present, reason="raw data not downloaded (scripts/fetch_data.py)")
def test_reproduces_original_2024_25_results():
    """Only meaningful on the original data snapshot; a fresh download may differ."""
    from nba_roi.roi import load_raw_data
    run = run_roi_model(*load_raw_data())
    r, p = run.results, run.params

    assert len(r) == 468
    assert len(run.unmatched) == 101
    assert p["total_salary_pool"] == 5_169_349_665
    assert round(p["sgv"]) == 2_101_362
    assert p["m_bar"] == pytest.approx(21.39, abs=0.005)
    assert r["ROI"].mean() == pytest.approx(1.583, abs=5e-4)
    assert r["ROI"].median() == pytest.approx(0.811, abs=5e-4)
    assert (r["ROI"] > 1.0).sum() == 212
    assert r.iloc[0]["PLAYER_NAME"] == "Isaac Jones"
    assert r.iloc[0]["ROI"] == pytest.approx(29.955, abs=5e-4)
    assert r.iloc[-1]["PLAYER_NAME"] == "Keyonte George"
    assert math.isclose(r.iloc[-1]["ROI"], -0.258, abs_tol=5e-4)
