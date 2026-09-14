# @title Simulação de temporada
import re
from functools import cmp_to_key
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def get_latest_team_elos(
    df: pd.DataFrame,
    sub_league: Optional[str] = "Serie A",
    home_col: str = "Home",
    away_col: str = "Away",
    home_elo_col: str = "Home_Elo_Pre",
    away_elo_col: str = "Away_Elo_Pre",
    date_col: str = "Date",
) -> Dict[str, float]:
    """
    Recupera o último Elo rating registrado de cada equipe a partir do histórico de jogos.
    """
    df_sorted = df.copy()
    if sub_league and "Sub_League" in df_sorted.columns:
        df_sorted = df_sorted[df_sorted["Sub_League"].astype(str).str.strip().str.lower() == str(sub_league).strip().lower()]

    df_sorted = df_sorted.sort_values(by=date_col)

    latest_elos = {}
    for _, row in df_sorted.iterrows():
        h_team = row[home_col]
        a_team = row[away_col]
        if pd.notna(row.get(home_elo_col)):
            latest_elos[h_team] = float(row[home_elo_col])
        if pd.notna(row.get(away_elo_col)):
            latest_elos[a_team] = float(row[away_elo_col])

    return latest_elos


def calibrate_league_parameters(
    df: pd.DataFrame,
    sub_league: Optional[str] = "Serie A",
    home_col: str = "Home",
    away_col: str = "Away",
    home_score_col: str = "Home_Score",
    away_score_col: str = "Away_Score",
    home_elo_col: str = "Home_Elo_Pre",
    away_elo_col: str = "Away_Elo_Pre",
) -> Dict[str, float]:
    """
    Calcula os parâmetros médios da liga a partir do histórico:
    - Médias de gols mandante e visitante
    - Vantagem de mando de campo em Elo
    - Sensibilidade do Elo na geração de gols
    """
    df_matches = df.copy()
    if sub_league and "Sub_League" in df_matches.columns:
        df_matches = df_matches[df_matches["Sub_League"].astype(str).str.strip().str.lower() == str(sub_league).strip().lower()]

    df_matches = df_matches.dropna(subset=[home_score_col, away_score_col])

    avg_home_goals = float(df_matches[home_score_col].mean()) if not df_matches.empty else 1.45
    avg_away_goals = float(df_matches[away_score_col].mean()) if not df_matches.empty else 0.95

    return {
        "avg_home_goals": avg_home_goals,
        "avg_away_goals": avg_away_goals,
        "home_adv_elo": 65.0,  # Vantagem histórica média de mando no Brasileirão em pontos Elo
        "elo_sensitivity": 0.0025,  # Fator de conversão delta Elo -> lambda Poisson
    }


def simulate_match_goals(
    home_elo: float,
    away_elo: float,
    params: dict,
    home_desperation: float = 0.0,
    away_desperation: float = 0.0,
) -> Tuple[int, int]:
    """
    Simula o placar (gols mandante, gols visitante) com distribuição de Poisson,
    considerando a diferença de Elo e o fator de desespero/motivação da reta final.
    """
    # Ajuste de Elo com mando de campo e motivação
    effective_home_elo = home_elo + params["home_adv_elo"] + home_desperation
    effective_away_elo = away_elo + away_desperation
    elo_diff = effective_home_elo - effective_away_elo

    # Lambda de Poisson para gols
    lambda_home = params["avg_home_goals"] * np.exp(params["elo_sensitivity"] * elo_diff)
    lambda_away = params["avg_away_goals"] * np.exp(-params["elo_sensitivity"] * elo_diff)

    # Limites de segurança para Poisson
    lambda_home = np.clip(lambda_home, 0.2, 4.5)
    lambda_away = np.clip(lambda_away, 0.2, 4.5)

    home_goals = int(np.random.poisson(lambda_home))
    away_goals = int(np.random.poisson(lambda_away))

    return home_goals, away_goals


def run_monte_carlo_season(
    df_current_standings: pd.DataFrame,
    df_fixtures: pd.DataFrame,
    latest_elos: Dict[str, float],
    params: dict,
    n_simulations: int = 1000,
    tiebreakers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Executa n_simulations completando os jogos restantes e calcula
    probabilidades de Título, G-4, G-6 e Rebaixamento (Z-4).
    """
    if tiebreakers is None:
        tiebreakers = ["points", "wins", "goal_difference", "goals_for"]

    teams = sorted(df_current_standings["Team"].unique())
    n_teams = len(teams)

    # Elo médio de fallback
    all_elos = list(latest_elos.values())
    avg_fallback_elo = float(np.mean(all_elos)) if all_elos else 1500.0

    # Estruturas para acumular estatísticas de todas as simulações
    final_pts_acc = {t: [] for t in teams}
    final_pos_acc = {t: [] for t in teams}

    champ_count = {t: 0 for t in teams}
    g4_count = {t: 0 for t in teams}
    g6_count = {t: 0 for t in teams}
    z4_count = {t: 0 for t in teams}

    fixtures_list = df_fixtures[["Home", "Away", "Round_Num"]].to_dict(orient="records")

    print(f"Rodando {n_simulations} simulações de Monte Carlo para {len(fixtures_list)} jogos restantes...")

    for sim_idx in range(n_simulations):
        # Inicia a simulação a partir do estado atual da tabela
        # Dicionário com [PTS, V, SG, GP, J]
        sim_table = {
            row["Team"]: {
                "PTS": int(row["PTS"]),
                "V": int(row["V"]),
                "SG": int(row["SG"]),
                "GP": int(row["GP"]),
                "GC": int(row["GC"]),
                "J": int(row["J"]),
                "Current_Pos": int(row["Pos"]),
            }
            for _, row in df_current_standings.iterrows()
        }

        # Simula cada jogo futuro
        for match in fixtures_list:
            h_team = match["Home"]
            a_team = match["Away"]
            rnd = match.get("Round_Num", 30)

            h_elo = latest_elos.get(h_team, avg_fallback_elo)
            a_elo = latest_elos.get(a_team, avg_fallback_elo)

            # Nuance de Reta Final (a partir da rodada 28)
            h_desperation = 0.0
            a_desperation = 0.0

            if rnd >= 28:
                h_pos = sim_table[h_team]["Current_Pos"]
                a_pos = sim_table[a_team]["Current_Pos"]

                # Desespero do Z-4 (posições 17 a 20) ou proximidade (15 e 16)
                if h_pos >= 17:
                    h_desperation += 35.0  # Fator raça contra o rebaixamento
                elif h_pos in [15, 16]:
                    h_desperation += 20.0

                if a_pos >= 17:
                    a_desperation += 35.0
                elif a_pos in [15, 16]:
                    a_desperation += 20.0

                # Zona morta (posições 9 a 13) jogando contra desesperado
                if 9 <= h_pos <= 13 and a_pos >= 16:
                    h_desperation -= 25.0  # Relaxamento / foco em copas
                if 9 <= a_pos <= 13 and h_pos >= 16:
                    a_desperation -= 25.0

            # Simula gols da partida
            h_gols, a_gols = simulate_match_goals(h_elo, a_elo, params, h_desperation, a_desperation)

            # Atualiza tabela da simulação
            sim_table[h_team]["J"] += 1
            sim_table[a_team]["J"] += 1
            sim_table[h_team]["GP"] += h_gols
            sim_table[h_team]["GC"] += a_gols
            sim_table[h_team]["SG"] += (h_gols - a_gols)
            sim_table[a_team]["GP"] += a_gols
            sim_table[a_team]["GC"] += h_gols
            sim_table[a_team]["SG"] += (a_gols - h_gols)

            if h_gols > a_gols:
                sim_table[h_team]["PTS"] += 3
                sim_table[h_team]["V"] += 1
            elif h_gols == a_gols:
                sim_table[h_team]["PTS"] += 1
                sim_table[a_team]["PTS"] += 1
            else:
                sim_table[a_team]["PTS"] += 3
                sim_table[a_team]["V"] += 1

        # Ordena a classificação final desta simulação
        sorted_teams = sorted(
            teams,
            key=lambda t: (
                sim_table[t]["PTS"],
                sim_table[t]["V"],
                sim_table[t]["SG"],
                sim_table[t]["GP"],
            ),
            reverse=True,
        )

        for pos_idx, t in enumerate(sorted_teams, start=1):
            final_pts_acc[t].append(sim_table[t]["PTS"])
            final_pos_acc[t].append(pos_idx)

            if pos_idx == 1:
                champ_count[t] += 1
            if pos_idx <= 4:
                g4_count[t] += 1
            if pos_idx <= 6:
                g6_count[t] += 1
            if pos_idx >= (n_teams - 3):  # 4 últimos (Z-4)
                z4_count[t] += 1

    # Monta DataFrame final com probabilidades
    summary_records = []
    for _, row in df_current_standings.iterrows():
        t = row["Team"]
        summary_records.append({
            "Pos_Atual": int(row["Pos"]),
            "Team": t,
            "Pts_Atual": int(row["PTS"]),
            "Pts_Medio": round(float(np.mean(final_pts_acc[t])), 1),
            "Pos_Media": round(float(np.mean(final_pos_acc[t])), 1),
            "Campeao_%": round((champ_count[t] / n_simulations) * 100, 1),
            "G4_%": round((g4_count[t] / n_simulations) * 100, 1),
            "G6_%": round((g6_count[t] / n_simulations) * 100, 1),
            "Rebaixamento_%": round((z4_count[t] / n_simulations) * 100, 1),
        })

    df_resultado = pd.DataFrame(summary_records)
    # Ordena por pontos médios esperados / probabilidade de título
    df_resultado = df_resultado.sort_values(by=["Campeao_%", "Pts_Medio"], ascending=[False, False]).reset_index(drop=True)
    df_resultado.insert(0, "Pos_Esperada", range(1, len(df_resultado) + 1))
    return df_resultado
