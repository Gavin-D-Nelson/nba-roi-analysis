"""Offline test of the NBA API collection logic, with the API responses mocked."""

import pandas as pd

from nba_roi import collect_stats as cs

IDS = [1, 2, 3]
NAMES = ["A One", "B Two", "C Three"]


def _base(**cols):
    return pd.DataFrame({"PLAYER_ID": IDS, "PLAYER_NAME": NAMES, **cols})


def fake_pt_stats(measure_type):
    if measure_type == "Rebounding":
        return _base(OREB_CONTEST=[1.0, 0.5, 0.0], OREB_UNCONTEST=[0.4, 0.2, 0.1],
                     DREB_CONTEST=[2.0, 1.0, 0.5], DREB_UNCONTEST=[3.0, 2.0, 1.0])
    if measure_type == "Passing":
        return _base(PASSES_MADE=[30.0, 20.0, 10.0], POTENTIAL_AST=[5.0, 4.0, 1.0])
    if measure_type == "SpeedDistance":
        return _base(DIST_MILES_OFF=[1.1, 1.0, 0.9], DIST_MILES_DEF=[1.0, 0.9, 0.8],
                     DIST_MILES=[2.1, 1.9, 1.7])
    raise ValueError(measure_type)


def fake_pt_defend(category):
    df = pd.DataFrame({"CLOSE_DEF_PERSON_ID": IDS, "PLAYER_NAME": NAMES})
    if category == "2 Pointers":
        df["FG2A"] = [5.0, 4.0, 3.0]
    elif category == "3 Pointers":
        df["FG3A"] = [3.0, 2.0, 1.0]
    else:
        df["D_FGM"] = [4.0, 3.0, 2.0]
        df["D_FGA"] = [9.0, 7.0, 5.0]
    return df


def test_get_advanced_season_builds_all_table1_vars(monkeypatch):
    monkeypatch.setattr(cs, "_pt_stats", fake_pt_stats)
    monkeypatch.setattr(cs, "_pt_defend", fake_pt_defend)
    monkeypatch.setattr(cs, "get_hustle", lambda: _base(SAST=[1.0, 0.5, 0.0],
                                                        CHGD=[0.1, 0.0, 0.0],
                                                        DBOX=[0.3, 0.2, 0.1]))
    standard = _base(FGM=[8.0, 5.0, 2.0], FGA=[16.0, 11.0, 5.0], FG3M=[2.0, 1.0, 0.0],
                     FG3A=[6.0, 3.0, 1.0], FTM=[4.0, 2.0, 1.0], FTA=[5.0, 3.0, 2.0],
                     STL=[1.0, 1.0, 0.0], BLK=[1.0, 0.0, 0.0], PF=[2.0, 2.0, 1.0],
                     TOV=[3.0, 1.0, 1.0], PFD=[4.0, 2.0, 1.0])
    monkeypatch.setattr(cs, "get_standard_season",
                        lambda: cs.add_shooting_splits(standard.copy()))

    adv = cs.get_advanced_season(pause=0)

    assert cs.check_table1_vars(adv) == []
    assert len(adv) == 3
    assert not any(c.endswith("_drop") for c in adv.columns)
    a = adv.set_index("PLAYER_ID").loc[1]
    assert a["OCRB"] == 1.0 and a["AORB"] == 0.4
    assert a["APM"] == 25.0          # 30 passes - 5 potential assists
    assert a["DFGX"] == 5.0          # 9 defended attempts - 4 makes
    assert a["AC2P"] == 5.0 and a["C3P"] == 3.0
    assert a["DDIS"] == 1.0
