"""Project-wide settings: file paths, season parameters and model weights.

All paths are built relative to the repository root, so the project runs the
same way on any machine.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# ── Season ────────────────────────────────────────────────────────────────────
SEASON = "2024-25"
SEASON_TYPE = "Regular Season"
PER_MODE = "PerGame"

# ESPN labels the 2024-25 season as "2025" in its salary URLs.
ESPN_SALARY_YEAR = 2025
ESPN_SALARY_PAGES = 13

SALARIES_FILE = RAW_DIR / f"nba_salaries_{SEASON}.xlsx"
GAMELOGS_FILE = RAW_DIR / f"nba_game_logs_{SEASON}.xlsx"
ADVANCED_FILE = RAW_DIR / f"nba_advanced_stats_{SEASON}.xlsx"
ROI_FILE = PROCESSED_DIR / f"player_roi_{SEASON}.xlsx"

# ── Model parameters ──────────────────────────────────────────────────────────
GAMES_PER_SEASON = 82
N_TEAMS = 30
MIN_SALARY = 0
# No minimum-games filter: missed games contribute $0 to Total_CF, so
# availability is penalised directly in the ROI ratio.

# Performance weights (Table 1, Lautier et al.).
# Used as a linear weighted sum of centered stats; no logistic/sigmoid applied.
BETA = {
    "FG2O": 0.251,
    "FG2X": -0.349,
    "FG3O": 0.537,
    "FG3X": -0.368,
    "FTMO": 0.122,
    "FTMX": -0.220,
    "PF": -0.197,
    "AORB": 0.356,
    "ADRB": 0.316,
    "STL": 0.443,
    "BLK": 0.132,
    "TOV": -0.347,
    "PFD": 0.214,
    "SAST": 0.076,
    "CHGD": 0.522,
    "AC2P": 0.041,
    "C3P": -0.067,
    "DBOX": 0.053,
    "DFGO": -0.230,
    "DFGX": 0.086,
    "DDIS": -1.000,
    "APM": 0.016,
    "OCRB": 0.290,
    "DCRB": 0.338,
}

# Tracking stats only available as season per-game averages. They are attached
# to every game a player appears in as a constant per-game rate.
ADVANCED_VARS = [
    "AORB", "ADRB", "OCRB", "DCRB",  # rebounding tracking
    "DFGO", "DFGX", "AC2P", "C3P",   # defensive tracking
    "SAST", "CHGD", "DBOX",          # hustle
    "APM",                           # passing
    "DDIS",                          # speed/distance
]
