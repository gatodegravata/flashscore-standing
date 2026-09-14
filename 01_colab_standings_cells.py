# ==============================================================================
# CÉLULA 1: Motor de Classificação (Cole no Google Colab)
# ==============================================================================
import re
from functools import cmp_to_key
from typing import Optional, Union, List
import pandas as pd

def _extract_round_number(round_val):
    if pd.isna(round_val):
        return None
    match = re.search(r"\d+", str(round_val))
    return int(match.group()) if match else None

def _calculate_head_to_head_points(matches_df: pd.DataFrame, team_a: str, team_b: str, pts_win: int, pts_draw: int) -> int:
    direct = matches_df[
        ((matches_df["Home"] == team_a) & (matches_df["Away"] == team_b)) |
        ((matches_df["Home"] == team_b) & (matches_df["Away"] == team_a))
    ]
    if direct.empty:
        return 0
    points_a = 0
    for _, match in direct.iterrows():
        if match["Home"] == team_a:
            if match["Home_Score"] > match["Away_Score"]:
                points_a += pts_win
            elif match["Home_Score"] == match["Away_Score"]:
                points_a += pts_draw
        else:
            if match["Away_Score"] > match["Home_Score"]:
                points_a += pts_win
            elif match["Away_Score"] == match["Home_Score"]:
                points_a += pts_draw
    return points_a

def get_standings_snapshot(
    df: pd.DataFrame,
    sub_league: Optional[str] = None,
    season: Optional[Union[str, int]] = None,
    target_round: Optional[Union[int, str]] = None,
    target_date: Optional[Union[str, pd.Timestamp]] = None,
    tiebreakers: Optional[List[str]] = None,
    pts_win: int = 3,
    pts_draw: int = 1,
    pts_loss: int = 0,
    home_col: str = "Home",
    away_col: str = "Away",
) -> pd.DataFrame:
    df_filtered = df.copy()

    if home_col not in df_filtered.columns:
        for c in ["Home_Team", "home", "home_team"]:
            if c in df_filtered.columns:
                home_col = c
                break
    if away_col not in df_filtered.columns:
        for c in ["Away_Team", "away", "away_team"]:
            if c in df_filtered.columns:
                away_col = c
                break

    if sub_league is not None and "Sub_League" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Sub_League"].astype(str).str.strip().str.lower() == str(sub_league).strip().lower()]
    
    if season is not None and "Season" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Season"].astype(str) == str(season)]

    df_filtered = df_filtered.dropna(subset=["Home_Score", "Away_Score"])

    if target_round is not None:
        target_round_num = _extract_round_number(target_round)
        if "Round_Num" not in df_filtered.columns:
            df_filtered["Round_Num"] = df_filtered["Round"].apply(_extract_round_number)
        df_filtered = df_filtered[df_filtered["Round_Num"] <= target_round_num]

    if target_date is not None:
        df_filtered["Date_Parsed"] = pd.to_datetime(df_filtered["Date"])
        df_filtered = df_filtered[df_filtered["Date_Parsed"] <= pd.to_datetime(target_date)]

    if df_filtered.empty:
        return pd.DataFrame(columns=["Pos", "Team", "PTS", "J", "V", "E", "D", "GP", "GC", "SG"])

    if tiebreakers is None:
        tiebreakers = ["points", "wins", "goal_difference", "goals_for", "head_to_head"]

    home_recs = pd.DataFrame({
        "Team": df_filtered[home_col],
        "Score_For": df_filtered["Home_Score"].astype(int),
        "Score_Against": df_filtered["Away_Score"].astype(int)
    })
    away_recs = pd.DataFrame({
        "Team": df_filtered[away_col],
        "Score_For": df_filtered["Away_Score"].astype(int),
        "Score_Against": df_filtered["Home_Score"].astype(int)
    })

    records = pd.concat([home_recs, away_recs], ignore_index=True)
    records["Win"] = (records["Score_For"] > records["Score_Against"]).astype(int)
    records["Draw"] = (records["Score_For"] == records["Score_Against"]).astype(int)
    records["Loss"] = (records["Score_For"] < records["Score_Against"]).astype(int)
    records["PTS"] = (records["Win"] * pts_win) + (records["Draw"] * pts_draw) + (records["Loss"] * pts_loss)

    standings = records.groupby("Team").agg(
        PTS=("PTS", "sum"),
        J=("Win", "count"),
        V=("Win", "sum"),
        E=("Draw", "sum"),
        D=("Loss", "sum"),
        GP=("Score_For", "sum"),
        GC=("Score_Against", "sum"),
    ).reset_index()

    standings["SG"] = standings["GP"] - standings["GC"]

    direct_match_df = pd.DataFrame({
        "Home": df_filtered[home_col].values,
        "Away": df_filtered[away_col].values,
        "Home_Score": df_filtered["Home_Score"].values,
        "Away_Score": df_filtered["Away_Score"].values,
    })

    def compare_teams(row_a: dict, row_b: dict) -> int:
        for tb in tiebreakers:
            if tb == "points" and row_a["PTS"] != row_b["PTS"]:
                return 1 if row_a["PTS"] > row_b["PTS"] else -1
            elif tb == "wins" and row_a["V"] != row_b["V"]:
                return 1 if row_a["V"] > row_b["V"] else -1
            elif tb == "goal_difference" and row_a["SG"] != row_b["SG"]:
                return 1 if row_a["SG"] > row_b["SG"] else -1
            elif tb == "goals_for" and row_a["GP"] != row_b["GP"]:
                return 1 if row_a["GP"] > row_b["GP"] else -1
            elif tb == "head_to_head":
                pts_h2h_a = _calculate_head_to_head_points(direct_match_df, row_a["Team"], row_b["Team"], pts_win, pts_draw)
                pts_h2h_b = _calculate_head_to_head_points(direct_match_df, row_b["Team"], row_a["Team"], pts_win, pts_draw)
                if pts_h2h_a != pts_h2h_b:
                    return 1 if pts_h2h_a > pts_h2h_b else -1
        return 1 if row_a["Team"] < row_b["Team"] else -1

    teams_list = standings.to_dict(orient="records")
    teams_list_sorted = sorted(teams_list, key=cmp_to_key(compare_teams), reverse=True)

    result_df = pd.DataFrame(teams_list_sorted)
    result_df.insert(0, "Pos", range(1, len(result_df) + 1))
    return result_df


def build_season_standings_history(
    df: pd.DataFrame,
    sub_league: Optional[str] = None,
    season: Optional[Union[str, int]] = None,
    tiebreakers: Optional[List[str]] = None,
    pts_win: int = 3,
    pts_draw: int = 1,
    pts_loss: int = 0,
    home_col: str = "Home",
    away_col: str = "Away",
) -> pd.DataFrame:
    """
    Calcula toda a evolução da temporada de uma só vez (rodada por rodada).
    """
    df_filtered = df.copy()

    if sub_league is not None and "Sub_League" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Sub_League"].astype(str).str.strip().str.lower() == str(sub_league).strip().lower()]
    if season is not None and "Season" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Season"].astype(str) == str(season)]

    df_filtered = df_filtered.dropna(subset=["Home_Score", "Away_Score"])

    if "Round_Num" not in df_filtered.columns:
        df_filtered["Round_Num"] = df_filtered["Round"].apply(_extract_round_number)

    all_rounds = sorted([int(r) for r in df_filtered["Round_Num"].dropna().unique()])

    history_snapshots = []
    print(f"Calculando histórico para {sub_league} - Temporada {season} ({len(all_rounds)} rodadas)...")
    for rnd in all_rounds:
        snap = get_standings_snapshot(
            df=df_filtered,
            target_round=rnd,
            tiebreakers=tiebreakers,
            pts_win=pts_win,
            pts_draw=pts_draw,
            pts_loss=pts_loss,
            home_col=home_col,
            away_col=away_col,
        )
        snap.insert(0, "Round_Num", rnd)
        if season is not None:
            snap.insert(0, "Season", season)
        if sub_league is not None:
            snap.insert(0, "Sub_League", sub_league)
        history_snapshots.append(snap)

    if not history_snapshots:
        return pd.DataFrame()

    print("Cálculo concluído!")
    return pd.concat(history_snapshots, ignore_index=True)


# ==============================================================================
# CÉLULA 2: Executa e Pré-Calcula Toda a Temporada (Uma única vez)
# ==============================================================================
"""
df_historico_brasileirao = build_season_standings_history(
    df=df_backup,
    sub_league="Serie A",     # ajuste para o nome da liga na sua base
    season=2024,
    tiebreakers=["points", "wins", "goal_difference", "goals_for", "head_to_head"]
)

# Salva na memória do notebook
print(f"Total de registros históricos gerados: {len(df_historico_brasileirao)}")
"""

# ==============================================================================
# CÉLULA 3: Consultar Qualquer Rodada Instantaneamente (Sem reprocessar!)
# ==============================================================================
"""
# Foto da Rodada 26:
rodada_desejada = 26
tabela_r26 = df_historico_brasileirao[df_historico_brasileirao['Round_Num'] == rodada_desejada]
display(tabela_r26[['Pos', 'Team', 'PTS', 'J', 'V', 'E', 'D', 'GP', 'GC', 'SG']])
"""

# ==============================================================================
# CÉLULA 4: Consultar Foto por Data Específica (usando a função direta)
# ==============================================================================
"""
# Para uma data específica qualquer (ex: até 2024-09-22):
tabela_data = get_standings_snapshot(
    df=df_backup,
    sub_league="Serie A",
    season=2024,
    target_date="2024-09-22"
)
display(tabela_data[['Pos', 'Team', 'PTS', 'J', 'V', 'E', 'D', 'GP', 'GC', 'SG']])
"""
