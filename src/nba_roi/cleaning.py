"""Data cleaning helpers shared by the collection and ROI steps."""

import re
import unicodedata

import pandas as pd


def normalize_name(name):
    """Normalize a player name so ESPN and NBA API names can be matched.

    1. Strip leading/trailing whitespace
    2. Remove ESPN position suffixes, e.g. 'Nikola Jokic, C' -> 'Nikola Jokic'
    3. Decompose unicode and drop combining characters (accents/diacritics),
       e.g. Jokić -> Jokic, Dončić -> Doncic
    4. Drop any remaining non-ASCII characters
    5. Collapse multiple spaces, uppercase
    """
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = re.sub(r",\s*[A-Z]+$", "", name).strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"\s+", " ", name).strip().upper()
    return name


def clean_salaries(sal, min_salary=0):
    """Parse ESPN salary strings ('$55,761,216') into numbers and dedupe players.

    Rows whose salary can't be parsed (e.g. repeated ESPN header rows) are
    dropped. When a normalized name appears twice, the highest salary is kept.
    Adds SALARY_CLEAN and PLAYER_NAME_CLEAN columns.
    """
    sal = sal.copy()
    sal["SALARY_CLEAN"] = (
        sal["Salary"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.strip()
    )
    sal["SALARY_CLEAN"] = pd.to_numeric(sal["SALARY_CLEAN"], errors="coerce")
    sal = sal.dropna(subset=["SALARY_CLEAN"])
    sal = sal[sal["SALARY_CLEAN"] > min_salary].copy()
    sal["PLAYER_NAME_CLEAN"] = sal["Player"].apply(normalize_name)
    sal = sal.sort_values("SALARY_CLEAN", ascending=False).drop_duplicates("PLAYER_NAME_CLEAN")
    return sal


def add_shooting_splits(df):
    """Derive the made/missed shooting variables used in the paper notation.

    Works on any frame with FGM, FGA, FG3M, FG3A, FTM, FTA columns (per-game
    logs or season averages). Modifies and returns ``df``.
    """
    df["FG2O"] = df["FGM"] - df["FG3M"]   # 2-pt FG made
    df["FG2A"] = df["FGA"] - df["FG3A"]   # 2-pt FG attempted
    df["FG2X"] = df["FG2A"] - df["FG2O"]  # 2-pt FG missed
    df["FG3O"] = df["FG3M"]               # 3-pt FG made
    df["FG3X"] = df["FG3A"] - df["FG3M"]  # 3-pt FG missed
    df["FTMO"] = df["FTM"]                # FT made
    df["FTMX"] = df["FTA"] - df["FTM"]    # FT missed
    return df
