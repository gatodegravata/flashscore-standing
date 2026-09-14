# ==============================================================================
# CÉLULA: Simulação Monte Carlo da Temporada (Título, G-4, G-6 e Rebaixamento)
# ==============================================================================
# Pré-requisitos:
# 1. df_backup (sua base histórica com Home_Elo_Pre, Away_Elo_Pre, Home_Score, Away_Score)
# 2. tabela_atual (gerada pela função get_standings_snapshot na rodada atual)
# 3. df_futuro (gerado pelo extrator de jogos futuros do Flashscore)

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

def get_latest_team_elos(
    df: pd.DataFrame,
    sub_league: Optional[str] = "Serie A",
    home_col: str = "Home",
    away_col: str = "Away",
    home_elo_col: str = "Home_Elo_Pre",
    away_elo_col: str = "Away_Elo_Pre",
    date_col: str = "Date",
) -> Dict[str, float]:
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
    home_score_col: str = "Home_Score",
    away_score_col: str = "Away_Score",
) -> dict:
    df_matches = df.copy()
    if sub_league and "Sub_League" in df_matches.columns:
        df_matches = df_matches[df_matches["Sub_League"].astype(str).str.strip().str.lower() == str(sub_league).strip().lower()]
    df_matches = df_matches.dropna(subset=[home_score_col, away_score_col])
    avg_h = float(df_matches[home_score_col].mean()) if not df_matches.empty else 1.45
    avg_a = float(df_matches[away_score_col].mean()) if not df_matches.empty else 0.95
    return {
        "avg_home_goals": avg_h,
        "avg_away_goals": avg_a,
        "home_adv_elo": 65.0,     # Mando de campo médio em pontos Elo
        "elo_sensitivity": 0.0025 # Sensibilidade na distribuição de Poisson
    }

def simulate_match_goals(home_elo: float, away_elo: float, params: dict, home_desp: float = 0.0, away_desp: float = 0.0) -> Tuple[int, int]:
    eff_home_elo = home_elo + params["home_adv_elo"] + home_desp
    eff_away_elo = away_elo + away_desp
    diff = eff_home_elo - eff_away_elo
    lambda_h = np.clip(params["avg_home_goals"] * np.exp(params["elo_sensitivity"] * diff), 0.2, 4.5)
    lambda_a = np.clip(params["avg_away_goals"] * np.exp(-params["elo_sensitivity"] * diff), 0.2, 4.5)
    return int(np.random.poisson(lambda_h)), int(np.random.poisson(lambda_a))

def run_season_simulation(
    df_current_standings: pd.DataFrame,
    df_fixtures: pd.DataFrame,
    df_historical: pd.DataFrame,
    sub_league: str = "Serie A",
    n_simulations: int = 1000,
) -> pd.DataFrame:
    """
    Executa a simulação de Monte Carlo completa.
    """
    latest_elos = get_latest_team_elos(df_historical, sub_league=sub_league)
    params = calibrate_league_parameters(df_historical, sub_league=sub_league)
    
    teams = sorted(df_current_standings["Team"].unique())
    n_teams = len(teams)
    avg_fallback_elo = float(np.mean(list(latest_elos.values()))) if latest_elos else 1500.0

    final_pts = {t: [] for t in teams}
    final_pos = {t: [] for t in teams}
    champ_count = {t: 0 for t in teams}
    g4_count = {t: 0 for t in teams}
    g6_count = {t: 0 for t in teams}
    z4_count = {t: 0 for t in teams}

    fixtures_list = df_fixtures[["Home", "Away", "Round_Num"]].to_dict(orient="records")

    print(f"Iniciando {n_simulations} simulações de Monte Carlo para os {len(fixtures_list)} jogos restantes...")

    for _ in range(n_simulations):
        # Tabela base da iteração
        sim_table = {
            r["Team"]: {
                "PTS": int(r["PTS"]),
                "V": int(r["V"]),
                "SG": int(r["SG"]),
                "GP": int(r["GP"]),
                "Pos_Atual": int(r["Pos"]),
            }
            for _, r in df_current_standings.iterrows()
        }

        # Simula cada partida restante
        for m in fixtures_list:
            h = m["Home"]
            a = m["Away"]
            rnd = m.get("Round_Num", 30)

            h_elo = latest_elos.get(h, avg_fallback_elo)
            a_elo = latest_elos.get(a, avg_fallback_elo)

            # Nuances de Reta Final (desespero vs zona morta a partir da rodada 28)
            h_desp, a_desp = 0.0, 0.0
            if rnd >= 28:
                h_p, a_p = sim_table[h]["Pos_Atual"], sim_table[a]["Pos_Atual"]
                # Desespero contra rebaixamento
                if h_p >= 17: h_desp += 35.0
                elif h_p in [15, 16]: h_desp += 20.0
                if a_p >= 17: a_desp += 35.0
                elif a_p in [15, 16]: a_desp += 20.0
                # Relaxamento / foco em copas (meio de tabela sem risco)
                if 9 <= h_p <= 13 and a_p >= 16: h_desp -= 25.0
                if 9 <= a_p <= 13 and h_p >= 16: a_desp -= 25.0

            hg, ag = simulate_match_goals(h_elo, a_elo, params, h_desp, a_desp)

            sim_table[h]["GP"] += hg
            sim_table[h]["SG"] += (hg - ag)
            sim_table[a]["GP"] += ag
            sim_table[a]["SG"] += (ag - hg)

            if hg > ag:
                sim_table[h]["PTS"] += 3
                sim_table[h]["V"] += 1
            elif hg == ag:
                sim_table[h]["PTS"] += 1
                sim_table[a]["PTS"] += 1
            else:
                sim_table[a]["PTS"] += 3
                sim_table[a]["V"] += 1

        # Classificação final desta iteração (Critérios do Brasileirão)
        ranking = sorted(
            teams,
            key=lambda t: (sim_table[t]["PTS"], sim_table[t]["V"], sim_table[t]["SG"], sim_table[t]["GP"]),
            reverse=True
        )

        for pos, t in enumerate(ranking, start=1):
            final_pts[t].append(sim_table[t]["PTS"])
            final_pos[t].append(pos)
            if pos == 1: champ_count[t] += 1
            if pos <= 4: g4_count[t] += 1
            if pos <= 6: g6_count[t] += 1
            if pos >= (n_teams - 3): z4_count[t] += 1

    # Compilação das probabilidades
    res = []
    for _, r in df_current_standings.iterrows():
        t = r["Team"]
        res.append({
            "Pos_Atual": int(r["Pos"]),
            "Team": t,
            "Pts_Atual": int(r["PTS"]),
            "Pts_Esperado": round(float(np.mean(final_pts[t])), 1),
            "Pos_Esperada": round(float(np.mean(final_pos[t])), 1),
            "Campeao_%": round((champ_count[t] / n_simulations) * 100, 1),
            "G4_%": round((g4_count[t] / n_simulations) * 100, 1),
            "G6_%": round((g6_count[t] / n_simulations) * 100, 1),
            "Rebaixamento_%": round((z4_count[t] / n_simulations) * 100, 1),
        })

    df_res = pd.DataFrame(res).sort_values(by=["Campeao_%", "Pts_Esperado"], ascending=[False, False]).reset_index(drop=True)
    df_res.insert(0, "Ranking_Projetado", range(1, len(df_res) + 1))
    return df_res


# ==============================================================================
# EXECUÇÃO NO COLAB
# ==============================================================================
"""
# 1. Pegue a foto da tabela mais recente (ex: rodada 27):
tabela_atual = get_standings_snapshot(df_backup, sub_league="Serie A", season=2024, target_round=27)

# 2. Pegue os jogos futuros restantes:
df_futuro = get_fixtures_dataframe("https://www.flashscore.com/football/brazil/serie-a-betano/fixtures/")

# 3. Rode a simulação de 1000 cenários:
df_projecao = run_season_simulation(
    df_current_standings=tabela_atual,
    df_fixtures=df_futuro,
    df_historical=df_backup,
    sub_league="Serie A",
    n_simulations=1000
)

display(df_projecao)
"""
