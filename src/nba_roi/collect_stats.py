"""Collect 2024-25 player stats from the NBA Stats API (via the nba_api package).

Produces two raw files:
  * game-by-game box scores for every player (one row per player-game)
  * season-level standard + tracking stats (one row per player)
"""

import time

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashptdefend,
    leaguedashptstats,
    leaguehustlestatsplayer,
    playergamelog,
)

from . import config
from .cleaning import add_shooting_splits

SEASON = config.SEASON
PER_MODE = config.PER_MODE
SEASON_TYPE = config.SEASON_TYPE

# Variables from Table 1 of Lautier et al. that must be present after merging.
TABLE1_VARS = [
    "FG2O", "FG2X", "FG3O", "FG3X", "FTMO", "FTMX",
    "STL", "BLK", "PF", "TOV", "PFD",
    "AORB", "ADRB", "OCRB", "DCRB",
    "DFGO", "DFGX", "AC2P", "C3P", "DDIS",
    "APM", "SAST", "CHGD", "DBOX",
]


# ── 1. Game-by-game box scores ────────────────────────────────────────────────

def get_player_list():
    """Every player who appeared in the season."""
    return leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        per_mode_detailed="Totals",
        season_type_all_star=SEASON_TYPE,
    ).get_data_frames()[0][["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]]


def get_game_logs(player_list_df, pause=0.6):
    """Fetch the game log for every player and add the shooting splits."""
    all_logs = []
    n = len(player_list_df)
    for i, row in player_list_df.iterrows():
        print(f"[{i + 1}/{n}] {row['PLAYER_NAME']}", end="\r")
        try:
            log = playergamelog.PlayerGameLog(
                player_id=row["PLAYER_ID"],
                season=SEASON,
                season_type_all_star=SEASON_TYPE,
            ).get_data_frames()[0]
            log["PLAYER_ID"] = row["PLAYER_ID"]
            log["PLAYER_NAME"] = row["PLAYER_NAME"]
            all_logs.append(log)
        except Exception as e:  # keep going if a single player fails
            print(f"\nWARNING: {row['PLAYER_NAME']} — {e}")
        time.sleep(pause)
    print(f"\nDone. Logs collected for {len(all_logs)} players.")
    return add_shooting_splits(pd.concat(all_logs, ignore_index=True))


# ── 2-6. Season-level tracking stats ──────────────────────────────────────────

def _pt_stats(measure_type):
    return leaguedashptstats.LeagueDashPtStats(
        season=SEASON,
        per_mode_simple=PER_MODE,
        pt_measure_type=measure_type,
        player_or_team="Player",
        season_type_all_star=SEASON_TYPE,
    ).get_data_frames()[0]


def _pt_defend(category):
    return leaguedashptdefend.LeagueDashPtDefend(
        season=SEASON,
        per_mode_simple=PER_MODE,
        defense_category=category,
        season_type_all_star=SEASON_TYPE,
    ).get_data_frames()[0]


def get_rebounding():
    """Contested vs. uncontested offensive and defensive rebounds."""
    reb_raw = _pt_stats("Rebounding")
    reb_off = reb_raw[["PLAYER_ID", "PLAYER_NAME", "OREB_CONTEST", "OREB_UNCONTEST"]].rename(
        columns={"OREB_CONTEST": "OCRB", "OREB_UNCONTEST": "AORB"})
    reb_def = reb_raw[["PLAYER_ID", "PLAYER_NAME", "DREB_CONTEST", "DREB_UNCONTEST"]].rename(
        columns={"DREB_CONTEST": "DCRB", "DREB_UNCONTEST": "ADRB"})
    return reb_off, reb_def


def get_defense_2pt():
    raw = _pt_defend("2 Pointers")
    return raw[["CLOSE_DEF_PERSON_ID", "PLAYER_NAME", "FG2A"]].rename(
        columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID", "FG2A": "AC2P"})


def get_defense_3pt():
    raw = _pt_defend("3 Pointers")
    return raw[["CLOSE_DEF_PERSON_ID", "PLAYER_NAME", "FG3A"]].rename(
        columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID", "FG3A": "C3P"})


def get_defense_overall():
    raw = _pt_defend("Overall")
    d = raw[["CLOSE_DEF_PERSON_ID", "PLAYER_NAME", "D_FGM", "D_FGA"]].rename(columns={
        "CLOSE_DEF_PERSON_ID": "PLAYER_ID", "D_FGM": "DFGO", "D_FGA": "DFGA"})
    d["DFGX"] = d["DFGA"] - d["DFGO"]
    return d[["PLAYER_ID", "PLAYER_NAME", "DFGO", "DFGX"]]


def get_hustle():
    """Screen assists, charges drawn, defensive box outs."""
    raw = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
        season=SEASON,
        per_mode_time=PER_MODE,
        season_type_all_star=SEASON_TYPE,
    ).get_data_frames()[0]
    return raw[["PLAYER_ID", "PLAYER_NAME", "SCREEN_ASSISTS", "CHARGES_DRAWN", "DEF_BOXOUTS"]].rename(
        columns={"SCREEN_ASSISTS": "SAST", "CHARGES_DRAWN": "CHGD", "DEF_BOXOUTS": "DBOX"})


def get_passing():
    """Non-assist passes made: APM = PASSES_MADE - POTENTIAL_AST."""
    raw = _pt_stats("Passing")
    p = raw[["PLAYER_ID", "PLAYER_NAME", "PASSES_MADE", "POTENTIAL_AST"]].copy()
    p["APM"] = p["PASSES_MADE"] - p["POTENTIAL_AST"]
    return p[["PLAYER_ID", "PLAYER_NAME", "APM", "PASSES_MADE", "POTENTIAL_AST"]]


def get_speed_distance():
    """Offensive/defensive distance traveled (DDIS = defensive miles)."""
    raw = _pt_stats("SpeedDistance")
    return raw[["PLAYER_ID", "PLAYER_NAME", "DIST_MILES_OFF", "DIST_MILES_DEF", "DIST_MILES"]].rename(
        columns={"DIST_MILES_DEF": "DDIS", "DIST_MILES_OFF": "ODIS", "DIST_MILES": "TOTAL_DIST"})


def get_standard_season():
    """Season-level per-game standard stats with the shooting splits."""
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        per_mode_detailed=PER_MODE,
        season_type_all_star=SEASON_TYPE,
    ).get_data_frames()[0]
    return add_shooting_splits(df)


def get_advanced_season(pause=1.0):
    """Merge every tracking dataset onto the standard season stats."""
    pieces = []
    for fetch in (get_rebounding, get_defense_2pt, get_defense_3pt, get_defense_overall,
                  get_hustle, get_passing, get_speed_distance):
        time.sleep(pause)
        out = fetch()
        pieces.extend(out if isinstance(out, tuple) else [out])
    time.sleep(pause)
    advanced = get_standard_season()

    # Duplicate columns (e.g. PLAYER_NAME) are dropped after every merge; newer
    # pandas versions refuse to create a second "_drop" column with the same name.
    for piece in pieces:
        advanced = advanced.merge(piece, on="PLAYER_ID", how="left", suffixes=("", "_drop"))
        advanced = advanced.drop(columns=[c for c in advanced.columns if c.endswith("_drop")])
    return advanced


def check_table1_vars(advanced):
    """Return the Table 1 variables missing from the season-level data."""
    return [v for v in TABLE1_VARS if v not in advanced.columns]


def main(gamelogs_file=config.GAMELOGS_FILE, advanced_file=config.ADVANCED_FILE):
    players = get_player_list()
    print(f"Total players: {len(players)}")
    gamelogs = get_game_logs(players)
    advanced = get_advanced_season()

    gamelogs_file.parent.mkdir(parents=True, exist_ok=True)
    gamelogs.to_excel(gamelogs_file, index=False)
    print(f"Saved {len(gamelogs):,} game rows -> {gamelogs_file}")
    advanced.to_excel(advanced_file, index=False)
    print(f"Saved {len(advanced)} players -> {advanced_file}")

    missing = check_table1_vars(advanced)
    print("All Table 1 variables present." if not missing else f"MISSING: {missing}")
    return gamelogs, advanced


if __name__ == "__main__":
    main()
