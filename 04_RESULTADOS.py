# @title 4. Simulação de Reta Final — Calibração PRO + Monte Carlo { form-width: "auto" }
import importlib
import numpy as np

# ============================================================
# RECARREGA O MÓDULO
# ============================================================

simulator_mod = importlib.import_module("03_season_simulator")
importlib.reload(simulator_mod)

get_latest_team_elos = simulator_mod.get_latest_team_elos
calibrate_poisson_elo_pro = simulator_mod.calibrate_poisson_elo_pro
fit_poisson_elo_params_given_beta = simulator_mod.fit_poisson_elo_params_given_beta
favorite_calibration_report = simulator_mod.favorite_calibration_report
elo_upset_report = simulator_mod.elo_upset_report
build_season_simulation_report = simulator_mod.build_season_simulation_report
MotivationConfig = simulator_mod.MotivationConfig


# ============================================================
# CONFIGURAÇÕES
# ============================================================

n_simulacoes = 100000 #@param {type:"slider", min:1000, max:100000, step:1000}
calibration_start = "2022-01-01" #@param {type:"string"}
validation_years = [2023, 2024, 2025]
half_life_days = 730 #@param {type:"integer"}

# Não introduzimos motivação/desespero sem validação histórica própria.
motivation = MotivationConfig(enabled=False)


# ============================================================
# 1. CALIBRAÇÃO TEMPORAL DO ELO
# ============================================================

calib = calibrate_poisson_elo_pro(
    df=df_backup,
    sub_league=sub_league,
    calibration_start=calibration_start,
    validation_years=validation_years,
    half_life_days=half_life_days,
    beta_min=0.0000,
    beta_max=0.0035,
    beta_step=0.00005,
)

params = calib["params"]
best_beta = calib["best_beta"]
hist = calib["history"]

print("=" * 110)
print("🧠 CALIBRAÇÃO PRO — MODELO ANTIGO × MODELO CALIBRADO")
print("=" * 110)
display(calib["comparison"])

print("\n=== DESEMPENHO DO BETA VENCEDOR POR TEMPORADA ===")
display(calib["folds"])

print("\n=== TOP 15 BETAS DA VALIDAÇÃO TEMPORAL ===")
display(calib["beta_cv"].head(15))

print("\n" + "=" * 110)
print("🔬 QUANTO O ELO PASSOU A PESAR?")
print("=" * 110)
print(f"Beta antigo:    0.00250000")
print(f"Beta calibrado: {best_beta:.8f}")
print("\nMultiplicador da expectativa de gols do time mais forte:")
for gap in [50, 100, 150, 200, 250, 300]:
    antigo = np.exp(0.0025 * gap)
    novo = np.exp(best_beta * gap)
    print(f"+{gap:3d} Elo | antigo ×{antigo:.3f} | calibrado ×{novo:.3f}")


# ============================================================
# 2. AUDITORIA DE FAVORITOS / ZEBRAS
# ============================================================
# Usa o último ano de validação e treina somente em datas anteriores.

usable_years = calib["usable_validation_years"]
if usable_years:
    audit_year = max(usable_years)
    audit_train = hist[hist["Date"] < f"{audit_year}-01-01"].copy()
    audit_valid = hist[hist["Year"] == audit_year].copy()

    audit_params = fit_poisson_elo_params_given_beta(
        audit_train,
        beta=best_beta,
        half_life_days=half_life_days,
    )

    print("\n" + "=" * 110)
    print(f"🎯 FAVORITOS — MODELO × REALIDADE ({audit_year})")
    print("=" * 110)
    display(favorite_calibration_report(audit_valid, audit_params))

    print("\n" + "=" * 110)
    print(f"💥 ZEBRAS REAIS POR DIFERENÇA DE ELO ({audit_year})")
    print("=" * 110)
    display(elo_upset_report(audit_valid))


# ============================================================
# 3. PARÂMETROS FINAIS
# ============================================================

print("\n" + "=" * 110)
print("⚙️ PARÂMETROS FINAIS USADOS NO MONTE CARLO")
print("=" * 110)
print(f"avg_home_goals:  {params['avg_home_goals']:.6f}")
print(f"avg_away_goals:  {params['avg_away_goals']:.6f}")
print(f"home_adv_elo:    {params['home_adv_elo']:.1f}  ← zero para não duplicar mando")
print(f"elo_sensitivity: {params['elo_sensitivity']:.8f}")


# ============================================================
# 4. ELO ATUAL DOS CLUBES
# ============================================================

latest_elos = get_latest_team_elos(
    df_backup,
    sub_league=sub_league,
)


# ============================================================
# 5. MONTE CARLO
# ============================================================

reports = build_season_simulation_report(
    df_current_standings=tabela_atual,
    df_fixtures=df_futuro,
    latest_elos=latest_elos,
    params=params,
    n_simulations=n_simulacoes,
    motivation=motivation,
    random_seed=42,
)


# ============================================================
# 6. RELATÓRIOS
# ============================================================

print("\n" + "=" * 110)
print("🏆 PROJEÇÃO FINAL DO CAMPEONATO")
print("=" * 110)
display(reports["resumo_times"])

print("\n" + "=" * 110)
print("📊 PROBABILIDADE DE TERMINAR EM CADA POSIÇÃO")
print("=" * 110)
display(reports["prob_posicao"])

print("\n" + "=" * 110)
print("🎯 DISTRIBUIÇÃO DE PONTOS E POSIÇÃO FINAL")
print("=" * 110)
display(reports["distribuicao_pontos"])

print("\n" + "=" * 110)
print("✂️ NOTAS DE CORTE DO CAMPEONATO")
print("=" * 110)
display(reports["cortes"])

print("\n" + "=" * 110)
print("📈 CHANCE DE ATINGIR OBJETIVOS POR PONTUAÇÃO FINAL")
print("=" * 110)
display(reports["prob_por_pontuacao"])

print("\n" + "=" * 110)
print("🔥 DIFICULDADE DOS JOGOS RESTANTES")
print("=" * 110)
display(reports["dificuldade_restante"])

print("\n" + "=" * 110)
print("⚽ PROBABILIDADES DOS JOGOS RESTANTES")
print("=" * 110)
display(reports["jogos_probabilidades"])

print("\n" + "=" * 110)
print("⚔️ CONFRONTOS DIRETOS DA RETA FINAL")
print("=" * 110)
display(reports["confrontos_diretos"])

print("\n" + "=" * 110)
print("🗓️ TRAJETÓRIA PROJETADA RODADA A RODADA")
print("=" * 110)
display(reports["trajetoria_rodadas"])

print("\n" + "=" * 110)
print("⚙️ INFORMAÇÕES DA SIMULAÇÃO")
print("=" * 110)
display(reports["sim_meta"])
