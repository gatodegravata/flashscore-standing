# ==============================================================================
# CÉLULA 1: Clonar Repositório e Instalar Dependências
# ==============================================================================
# Rode esta célula para baixar os módulos diretamente do GitHub no ambiente do Colab:
!rm -rf flashscore-standing
!git clone https://github.com/gatodegravata/flashscore-standing.git
!pip install -q curl_cffi

import sys
import os
sys.path.append(os.path.abspath("flashscore-standing"))

# Importa as funções dos módulos clonados
from standings_engine_01 import get_standings_snapshot, build_season_standings_history
from fixtures_extractor_02 import get_fixtures_dataframe
from season_simulator_03 import run_season_simulation

print("Repositório clonado e módulos carregados com sucesso!")


# ==============================================================================
# CÉLULA 2: Calcular Classificação Atual e Histórico da Temporada
# ==============================================================================
# Premissa: df_backup já está na memória do seu Colab!

# Opção A: Foto instantânea de uma rodada específica (ex: rodada 27)
tabela_atual = get_standings_snapshot(
    df=df_backup,
    sub_league="Serie A",     # ajuste para como está nomeado na sua base se necessário
    season=2024,
    target_round=27,
    tiebreakers=["points", "wins", "goal_difference", "goals_for", "head_to_head"]
)
display(tabela_atual[["Pos", "Team", "PTS", "J", "V", "E", "D", "GP", "GC", "SG"]])

# Opção B: Pré-calcular toda a temporada rodada a rodada
# df_historico = build_season_standings_history(df_backup, sub_league="Serie A", season=2024)


# ==============================================================================
# CÉLULA 3: Extrair Jogos Restantes + Escudos via Flashscore (curl_cffi)
# ==============================================================================
url_fixtures = "https://www.flashscore.com/football/brazil/serie-a-betano/fixtures/"
df_futuro = get_fixtures_dataframe(url_fixtures)

print(f"Total de jogos restantes encontrados: {len(df_futuro)}")
print(f"Rodadas futuras: {sorted(df_futuro['Round_Num'].dropna().unique())}")
display(df_futuro[["Round", "Date", "Home", "Away", "Home_Logo"]].head(10))


# ==============================================================================
# CÉLULA 4: Simulação de Monte Carlo (Reta Final - 1000 Cenários)
# ==============================================================================
# Considera: Elo histórico de cada time, Poisson, mando de campo e fator desespero/zona morta

df_projecao = run_season_simulation(
    df_current_standings=tabela_atual,
    df_fixtures=df_futuro,
    df_historical=df_backup,
    sub_league="Serie A",
    n_simulations=1000
)

# Exibe a tabela final com probabilidades de Título, G-4, G-6 e Rebaixamento
display(df_projecao)
