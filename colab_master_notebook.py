# @title 1. Clonar Repositório e Carregar Módulos { form-width: "auto" }
!rm -rf flashscore-standing
!git clone -q https://github.com/gatodegravata/flashscore-standing.git
!pip install -q curl_cffi

import sys
import os
sys.path.append(os.path.abspath("flashscore-standing"))

# Importação dos módulos do repositório
import importlib
standings_mod = importlib.import_module("01_standings_engine")
fixtures_mod = importlib.import_module("02_fixtures_extractor")
simulator_mod = importlib.import_module("03_season_simulator")

get_standings_snapshot = standings_mod.get_standings_snapshot
build_season_standings_history = standings_mod.build_season_standings_history
get_fixtures_dataframe = fixtures_mod.get_fixtures_dataframe
run_season_simulation = simulator_mod.run_monte_carlo_season
get_latest_team_elos = simulator_mod.get_latest_team_elos
calibrate_league_parameters = simulator_mod.calibrate_league_parameters

print("Repositório clonado e módulos carregados com sucesso!")


# @title 2. Calcular Classificação Atual { form-width: "auto" }
sub_league = "Serie A Betano" #@param {type:"string"}
season = 2026 #@param {type:"integer"}
target_round = 27 #@param {type:"integer"}

tabela_atual = get_standings_snapshot(
    df=df_backup,
    sub_league=sub_league,
    season=season,
    target_round=target_round,
    tiebreakers=["points", "wins", "goal_difference", "goals_for", "head_to_head"]
)

display(tabela_atual[["Pos", "Team", "PTS", "J", "V", "E", "D", "GP", "GC", "SG"]])


# @title 3. Extrair Agenda Restante com curl_cffi { form-width: "auto" }
url_fixtures = "https://www.flashscore.com/football/brazil/serie-a-betano/fixtures/" #@param {type:"string"}

df_futuro = get_fixtures_dataframe(url_fixtures)

print(f"Total de jogos restantes encontrados: {len(df_futuro)}")
print(f"Rodadas futuras mapeadas: {sorted(df_futuro['Round_Num'].dropna().unique())}")
display(df_futuro[["Round", "Date", "Home", "Away", "Home_Logo"]].head(10))


# @title 4. Simulação de Reta Final (Monte Carlo) { form-width: "auto" }
n_simulacoes = 1000 #@param {type:"slider", min:100, max:5000, step:100}

# Extrai os últimos Elos e calibra os parâmetros a partir de df_backup
latest_elos = get_latest_team_elos(df_backup, sub_league=sub_league)
params = calibrate_league_parameters(df_backup, sub_league=sub_league)

# Roda a simulação com os critérios do Brasileirão e fatores de reta final
df_projecao, df_corte_z4 = run_season_simulation(
    df_current_standings=tabela_atual,
    df_fixtures=df_futuro,
    latest_elos=latest_elos,
    params=params,
    n_simulations=n_simulacoes
)

# 1. Exibe as probabilidades por clube (Título, G-4, G-6 e Rebaixamento)
print("=== PROJEÇÃO FINAL DA TABELA ===")
display(df_projecao)

# 2. Exibe a distribuição exata da nota de corte do Z-4 (pontuação do 17º colocado)
print("\n=== NOTA DE CORTE DO REBAIXAMENTO (PONTUAÇÃO DO 17º COLOCADO) ===")
display(df_corte_z4)
