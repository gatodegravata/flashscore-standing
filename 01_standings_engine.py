# @title Função - calcula tabela
import json
import re
from functools import cmp_to_key
from typing import Dict, Optional, Union
import pandas as pd


def load_rules(rules_path_or_dict: Union[str, dict] = "league_rules.json") -> dict:
    """Carrega as regras de classificação a partir de um arquivo JSON ou dicionário."""
    if isinstance(rules_path_or_dict, dict):
        return rules_path_or_dict
    with open(rules_path_or_dict, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_round_number(round_val) -> Optional[int]:
    """Extrai número inteiro de formatos como 'Round 26', '26', 26, etc."""
    if pd.isna(round_val):
        return None
    match = re.search(r"\d+", str(round_val))
    return int(match.group()) if match else None


def _calculate_head_to_head_points(matches_df: pd.DataFrame, team_a: str, team_b: str, pts_win: int, pts_draw: int) -> int:
    """
    Calcula os pontos obtidos pelo team_a contra o team_b nos confrontos diretos
    presentes no recorte de partidas analisado.
    """
    direct_matches = matches_df[
        ((matches_df["Home"] == team_a) & (matches_df["Away"] == team_b))
        | ((matches_df["Home"] == team_b) & (matches_df["Away"] == team_a))
    ]
    if direct_matches.empty:
        return 0

    points_a = 0
    for _, match in direct_matches.iterrows():
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
    tiebreakers: Optional[list] = None,
    pts_win: int = 3,
    pts_draw: int = 1,
    pts_loss: int = 0,
    rule_key: Optional[str] = None,
    rules: Optional[Union[str, dict]] = None,
    home_col: str = "Home",
    away_col: str = "Away",
) -> pd.DataFrame:
    """
    Gera a foto da tabela de classificação.

    Parâmetros:
    -----------
    df : pd.DataFrame
        Base com os jogos (df_backup).
    sub_league : str, opcional
        Nome EXATO ou parcial da Sub_League a filtrar (ex: 'Serie A').
    season : str ou int, opcional
        Temporada (ex: 2024).
    target_round : int ou str, opcional
        Foto da rodada (ex: 26 ou 'Round 26').
    target_date : str ou Timestamp, opcional
        Foto por data (ex: '2024-09-20').
    tiebreakers : list, opcional
        Ordem manual dos critérios de desempate. Exemplo para o Brasileirão:
        ['points', 'wins', 'goal_difference', 'goals_for', 'head_to_head']
        Para Premier League:
        ['points', 'goal_difference', 'goals_for', 'head_to_head']
    pts_win / pts_draw / pts_loss : int
        Pontuação por vitória (default 3), empate (default 1) e derrota (default 0).
    rule_key : str, opcional
        Se preferir usar o JSON, nome da chave (ex: 'brazil_serie_a').
    rules : str ou dict, opcional
        Caminho para o JSON de regras ou dicionário (se fornecido).
    """
    df_filtered = df.copy()

    # Detectar colunas de times
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

    # 1. Filtro estrito de Sub_League (apenas a liga desejada)
    if sub_league is not None and "Sub_League" in df_filtered.columns:
        # Permite match exato ou case-insensitive
        df_filtered = df_filtered[df_filtered["Sub_League"].astype(str).str.strip().str.lower() == str(sub_league).strip().lower()]

    if season is not None and "Season" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Season"].astype(str) == str(season)]

    # Apenas partidas com placar preenchido
    df_filtered = df_filtered.dropna(subset=["Home_Score", "Away_Score"])

    # 2. Filtro de Rodada ou Data
    if target_round is not None:
        target_round_num = _extract_round_number(target_round)
        if "Round_Num" not in df_filtered.columns:
            df_filtered["Round_Num"] = df_filtered["Round"].apply(_extract_round_number)
        df_filtered = df_filtered[df_filtered["Round_Num"] <= target_round_num]

    if target_date is not None:
        df_filtered["Date_Parsed"] = pd.to_datetime(df_filtered["Date"])
        cutoff_date = pd.to_datetime(target_date)
        df_filtered = df_filtered[df_filtered["Date_Parsed"] <= cutoff_date]

    if df_filtered.empty:
        return pd.DataFrame(columns=["Pos", "Team", "PTS", "J", "V", "E", "D", "GP", "GC", "SG"])

    # 3. Definição das Regras e Critérios de Desempate
    # Prioridade:
    # A) Parâmetros manuais diretos (tiebreakers passados pelo usuário)
    # B) Regras do JSON (se rules ou rule_key informados)
    # C) Default brasileiro: points -> wins -> goal_difference -> goals_for -> head_to_head
    if tiebreakers is None:
        if rules is not None or rule_key is not None:
            all_rules = load_rules(rules if rules is not None else "league_rules.json")
            selected_rule = all_rules.get(rule_key or "default", all_rules.get("default", {}))
            pts_win = selected_rule.get("points", {}).get("win", pts_win)
            pts_draw = selected_rule.get("points", {}).get("draw", pts_draw)
            pts_loss = selected_rule.get("points", {}).get("loss", pts_loss)
            tiebreakers = selected_rule.get("tiebreakers", ["points", "wins", "goal_difference", "goals_for", "head_to_head"])
        else:
            tiebreakers = ["points", "wins", "goal_difference", "goals_for", "head_to_head"]

    # 4. Processar jogos sob a perspectiva de cada equipe
    home_records = pd.DataFrame({
        "Team": df_filtered[home_col],
        "Score_For": df_filtered["Home_Score"].astype(int),
        "Score_Against": df_filtered["Away_Score"].astype(int),
    })

    away_records = pd.DataFrame({
        "Team": df_filtered[away_col],
        "Score_For": df_filtered["Away_Score"].astype(int),
        "Score_Against": df_filtered["Home_Score"].astype(int),
    })

    records = pd.concat([home_records, away_records], ignore_index=True)
    records["Win"] = (records["Score_For"] > records["Score_Against"]).astype(int)
    records["Draw"] = (records["Score_For"] == records["Score_Against"]).astype(int)
    records["Loss"] = (records["Score_For"] < records["Score_Against"]).astype(int)
    records["PTS"] = (records["Win"] * pts_win) + (records["Draw"] * pts_draw) + (records["Loss"] * pts_loss)

    # 5. Agregar estatísticas da tabela
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

    # 6. Preparar DataFrame padronizado para cálculo de confronto direto se necessário
    direct_match_df = pd.DataFrame({
        "Home": df_filtered[home_col].values,
        "Away": df_filtered[away_col].values,
        "Home_Score": df_filtered["Home_Score"].values,
        "Away_Score": df_filtered["Away_Score"].values,
    })

    # 7. Função de Comparação com Critérios de Desempate
    def compare_teams(row_a: dict, row_b: dict) -> int:
        for tb in tiebreakers:
            if tb == "points":
                if row_a["PTS"] != row_b["PTS"]:
                    return 1 if row_a["PTS"] > row_b["PTS"] else -1
            elif tb == "wins":
                if row_a["V"] != row_b["V"]:
                    return 1 if row_a["V"] > row_b["V"] else -1
            elif tb == "goal_difference":
                if row_a["SG"] != row_b["SG"]:
                    return 1 if row_a["SG"] > row_b["SG"] else -1
            elif tb == "goals_for":
                if row_a["GP"] != row_b["GP"]:
                    return 1 if row_a["GP"] > row_b["GP"] else -1
            elif tb == "head_to_head":
                pts_h2h_a = _calculate_head_to_head_points(direct_match_df, row_a["Team"], row_b["Team"], pts_win, pts_draw)
                pts_h2h_b = _calculate_head_to_head_points(direct_match_df, row_b["Team"], row_a["Team"], pts_win, pts_draw)
                if pts_h2h_a != pts_h2h_b:
                    return 1 if pts_h2h_a > pts_h2h_b else -1
        # Se empatar em tudo, mantém alfabético
        return 1 if row_a["Team"] < row_b["Team"] else -1

    # Converter para lista de dicionários para ordenar com chave customizada
    teams_list = standings.to_dict(orient="records")
    teams_list_sorted = sorted(teams_list, key=cmp_to_key(compare_teams), reverse=True)

    result_df = pd.DataFrame(teams_list_sorted)
    result_df.insert(0, "Pos", range(1, len(result_df) + 1))
    return result_df


def build_season_standings_history(
    df: pd.DataFrame,
    sub_league: Optional[str] = None,
    season: Optional[Union[str, int]] = None,
    tiebreakers: Optional[list] = None,
    pts_win: int = 3,
    pts_draw: int = 1,
    pts_loss: int = 0,
    home_col: str = "Home",
    away_col: str = "Away",
) -> pd.DataFrame:
    """
    Calcula toda a evolução da temporada de uma vez só.
    Retorna um DataFrame acumulado com a classificação de cada time rodada a rodada.
    """
    df_filtered = df.copy()

    if sub_league is not None and "Sub_League" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Sub_League"].astype(str).str.strip().str.lower() == str(sub_league).strip().lower()]
    if season is not None and "Season" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Season"].astype(str) == str(season)]

    df_filtered = df_filtered.dropna(subset=["Home_Score", "Away_Score"])

    if "Round_Num" not in df_filtered.columns:
        df_filtered["Round_Num"] = df_filtered["Round"].apply(_extract_round_number)

    all_rounds = sorted([r for r in df_filtered["Round_Num"].dropna().unique()])

    history_snapshots = []
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

    return pd.concat(history_snapshots, ignore_index=True)


def get_snapshot_from_history(
    history_df: pd.DataFrame,
    target_round: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """
    Filtro instantâneo a partir da base histórica pré-calculada por build_season_standings_history.
    """
    if target_round is not None:
        rnd = _extract_round_number(target_round)
        return history_df[history_df["Round_Num"] == rnd].sort_values("Pos").reset_index(drop=True)
    return history_df

