"""ROI model: performance score -> WRMS -> single game value -> cash flow -> ROI.

Based on the framework of Lautier et al., simplified to a linear performance
score (no logistic transform) and a simple cash-flow ratio (no IRR solve).

    Delta_gm = sum_j beta_j * (x_gm_j - x_bar_j)
    WRMS_gm  = ((Delta_gm - Delta_bar) / SD(Delta)) * (1 / m_bar) + (1 / m_bar)
    SGV      = total salary pool / total team-games
    CF_gm    = SGV * WRMS_gm
    ROI      = sum_g CF_gm / salary
"""

from dataclasses import dataclass, field

import pandas as pd

from . import config
from .cleaning import add_shooting_splits, clean_salaries, normalize_name


# ── Step-by-step building blocks ──────────────────────────────────────────────

def attach_advanced_stats(logs, adv, advanced_vars=config.ADVANCED_VARS):
    """Attach season-average tracking stats to every game-log row.

    Players with no tracking data get 0. If the game logs have no PFD
    (fouls drawn) column it is set to 0, which matches the original analysis.
    """
    adv_cols = [v for v in advanced_vars if v in adv.columns]
    logs_merged = logs.merge(adv[["PLAYER_ID"] + adv_cols], on="PLAYER_ID", how="left")

    if "PFD" not in logs_merged.columns:
        logs_merged["PFD"] = 0  # conservative fallback (see README: Known limitations)

    fill_cols = [v for v in advanced_vars if v in logs_merged.columns]
    logs_merged[fill_cols] = logs_merged[fill_cols].fillna(0)
    return logs_merged


def compute_delta(df, beta=config.BETA):
    """Linear performance score for every player-game row.

    Each variable is centered on its mean across all player-game rows, so an
    exactly average game scores 0. Returns (delta Series, dict of column means).
    Variables in ``beta`` that are missing from ``df`` are skipped.
    """
    col_means = {var: df[var].mean() for var in beta if var in df.columns}

    delta = pd.Series(0.0, index=df.index)
    for var, coef in beta.items():
        if var in col_means:
            delta += coef * (df[var] - col_means[var])
    return delta, col_means


def average_players_per_game(df, game_col="Game_ID", player_col="PLAYER_ID"):
    """m_bar: average number of player rows per game (both teams combined)."""
    return df.groupby(game_col)[player_col].count().mean()


def compute_wrms(delta, m_bar):
    """Wealth Redistribution Merit Share.

    Standardizes Delta so the mean share is exactly 1/m_bar (a replacement-level
    share). Returns (wrms Series, delta mean, delta sample std).
    """
    delta_bar = delta.mean()
    delta_std = delta.std(ddof=1)
    wrms = ((delta - delta_bar) / delta_std) * (1 / m_bar) + (1 / m_bar)
    return wrms, delta_bar, delta_std


def single_game_value(total_salary_pool, n_teams=config.N_TEAMS,
                      games_per_season=config.GAMES_PER_SEASON):
    """Uniform SGV: total salary pool divided by total team-games."""
    return total_salary_pool / (n_teams * games_per_season)


def compute_roi(total_cf, salary):
    """ROI ratio. 1.0 means value generated equals salary paid."""
    return total_cf / salary


def summarize_players(logs_merged, games_per_season=config.GAMES_PER_SEASON):
    """Collapse player-game rows to one row per player."""
    player_season = logs_merged.groupby(["PLAYER_ID", "PLAYER_NAME", "PLAYER_NAME_CLEAN"]).agg(
        GAMES_PLAYED=("CF", "count"),
        TOTAL_CF=("CF", "sum"),
        MEAN_CF=("CF", "mean"),
        MEAN_DELTA=("DELTA", "mean"),
        MEAN_WRMS=("WRMS", "mean"),
        TOTAL_WRMS=("WRMS", "sum"),
    ).reset_index()
    player_season["GAMES_MISSED"] = games_per_season - player_season["GAMES_PLAYED"]
    return player_season


def build_results_table(player_season):
    """Final ranked table (highest ROI first) with value over/under contract."""
    results = player_season[[
        "PLAYER_NAME", "Team", "SALARY_CLEAN",
        "GAMES_PLAYED", "GAMES_MISSED",
        "MEAN_DELTA", "MEAN_WRMS", "TOTAL_CF", "ROI",
    ]].dropna(subset=["ROI"]).sort_values("ROI", ascending=False).reset_index(drop=True)
    results["VALUE_DIFF"] = results["TOTAL_CF"] - results["SALARY_CLEAN"]
    return results


# ── Full pipeline ─────────────────────────────────────────────────────────────

@dataclass
class RoiRun:
    """Everything produced by one run of the model."""
    results: pd.DataFrame
    logs: pd.DataFrame
    player_season: pd.DataFrame
    unmatched: pd.DataFrame
    salaries: pd.DataFrame
    params: dict = field(default_factory=dict)


def run_roi_model(logs, adv, sal, beta=config.BETA, advanced_vars=config.ADVANCED_VARS,
                  games_per_season=config.GAMES_PER_SEASON, n_teams=config.N_TEAMS,
                  min_salary=config.MIN_SALARY):
    """Run the full ROI model on raw game logs, advanced stats and salaries."""
    sal = clean_salaries(sal, min_salary=min_salary)

    logs = add_shooting_splits(logs.copy())
    logs["PLAYER_NAME_CLEAN"] = logs["PLAYER_NAME"].apply(normalize_name)

    logs_merged = attach_advanced_stats(logs, adv, advanced_vars)

    delta, col_means = compute_delta(logs_merged, beta)
    logs_merged["DELTA"] = delta

    game_col = "Game_ID" if "Game_ID" in logs_merged.columns else "GAME_DATE"
    m_bar = average_players_per_game(logs_merged, game_col=game_col)
    wrms, delta_bar, delta_std = compute_wrms(logs_merged["DELTA"], m_bar)
    logs_merged["WRMS"] = wrms

    total_salary_pool = sal["SALARY_CLEAN"].sum()
    sgv = single_game_value(total_salary_pool, n_teams, games_per_season)
    logs_merged["CF"] = sgv * logs_merged["WRMS"]

    player_season = summarize_players(logs_merged, games_per_season)
    player_season = player_season.merge(
        sal[["PLAYER_NAME_CLEAN", "SALARY_CLEAN", "Player", "Team"]],
        on="PLAYER_NAME_CLEAN", how="left",
    )
    unmatched = player_season[player_season["SALARY_CLEAN"].isna()][["PLAYER_NAME", "PLAYER_NAME_CLEAN"]]

    # Drop two-way / G-League players with no salary match
    player_season = player_season.dropna(subset=["SALARY_CLEAN"]).copy()
    player_season["ROI"] = compute_roi(player_season["TOTAL_CF"], player_season["SALARY_CLEAN"])

    results = build_results_table(player_season)

    params = {
        "season": config.SEASON,
        "games_per_season": games_per_season,
        "total_salary_pool": total_salary_pool,
        "total_team_games": n_teams * games_per_season,
        "sgv": sgv,
        "delta_bar": delta_bar,
        "delta_std": delta_std,
        "m_bar": m_bar,
        "replacement_share": 1 / m_bar,
        "replacement_cf": sgv * (1 / m_bar) * games_per_season,
        "avg_salary": sal["SALARY_CLEAN"].mean(),
        "col_means": col_means,
    }
    return RoiRun(results=results, logs=logs_merged, player_season=player_season,
                  unmatched=unmatched, salaries=sal, params=params)


def load_raw_data(gamelogs_file=config.GAMELOGS_FILE, advanced_file=config.ADVANCED_FILE,
                  salaries_file=config.SALARIES_FILE):
    """Read the three raw Excel files produced by the data-collection step."""
    missing = [str(p) for p in (gamelogs_file, advanced_file, salaries_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Raw data not found:\n  " + "\n  ".join(missing)
            + "\nRun `python scripts/fetch_data.py` first."
        )
    return (pd.read_excel(gamelogs_file), pd.read_excel(advanced_file),
            pd.read_excel(salaries_file))


def write_workbook(run, path=config.ROI_FILE, beta=config.BETA):
    """Save the 4-sheet ROI workbook (same layout as the original analysis)."""
    p = run.params
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Sheet 1: ROI Results
        run.results[[
            "PLAYER_NAME", "Team", "SALARY_CLEAN",
            "GAMES_PLAYED", "GAMES_MISSED",
            "MEAN_DELTA", "MEAN_WRMS", "TOTAL_CF", "VALUE_DIFF", "ROI",
        ]].rename(columns={
            "PLAYER_NAME": "Player",
            "SALARY_CLEAN": "Salary ($)",
            "MEAN_DELTA": "Avg Performance Score (Delta)",
            "MEAN_WRMS": "Avg WRMS Share",
            "TOTAL_CF": "Total Cash Flow ($)",
            "VALUE_DIFF": "Value Over/Under Contract ($)",
            "ROI": "ROI (Total_CF / Salary)",
        }).to_excel(writer, sheet_name="ROI Results", index=False)

        # Sheet 2: Model Parameters
        pd.DataFrame({
            "Parameter": [
                "Season", "Games per Season",
                "Total Salary Pool ($)", "Total Team-Games", "SGV per Game ($)",
                "Performance Model",
                "ROI Method",
                "Break-even ROI",
                "Global Delta Mean", "Global Delta Std",
                "Avg Players per Game (m_bar)", "Replacement Share (1/m_bar)",
            ],
            "Value": [
                p["season"], p["games_per_season"],
                round(p["total_salary_pool"], 2), p["total_team_games"], round(p["sgv"], 2),
                "Linear weighted score (Lautier et al. Table 1 weights, no sigmoid)",
                "Total_CF / Salary",
                "1.000x",
                round(p["delta_bar"], 6), round(p["delta_std"], 6),
                round(p["m_bar"], 4), round(p["replacement_share"], 6),
            ],
            "Note": [
                "", "",
                "Sum of ESPN scraped salaries", "30 teams x 82 games",
                "Salary pool / team-games (uniform SGV)",
                "Delta = sum(beta_j * (x_gm_j - x_bar_j)) across 24 variables",
                "Missed games contribute $0 — availability penalised directly",
                "Value generated equals salary paid",
                "Over all player-game observations",
                "Over all player-game observations",
                "Avg players with MIN > 0 per game",
                "Replacement-level WRMS share",
            ],
        }).to_excel(writer, sheet_name="Model Parameters", index=False)

        # Sheet 3: Performance Weights (Table 1)
        pd.DataFrame([
            {"Variable": k, "Weight (Beta)": v, "Source": "Table 1, Lautier et al."}
            for k, v in beta.items()
        ]).to_excel(writer, sheet_name="Performance Weights", index=False)

        # Sheet 4: Game Logs with Delta, WRMS, CF
        log_cols = ["PLAYER_NAME", "GAME_DATE", "MATCHUP", "WL",
                    "MIN", "PTS", "AST", "REB", "STL", "BLK", "TOV",
                    "FG2O", "FG2X", "FG3O", "FG3X",
                    "DELTA", "WRMS", "CF"]
        run.logs[[c for c in log_cols if c in run.logs.columns]].to_excel(
            writer, sheet_name="Game Logs + WRMS", index=False
        )
    return path
