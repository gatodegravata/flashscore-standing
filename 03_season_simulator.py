# @title Simulação de temporada — Monte Carlo V2 completo
"""
Simulador de reta final do Brasileirão.

Principais mudanças em relação à versão inicial:
- classificação dinâmica recalculada no início de cada rodada;
- desempates realmente configuráveis;
- preferência por Elo pós-jogo quando disponível;
- Monte Carlo vetorizado em NumPy;
- matriz de probabilidades por posição (1º...20º);
- distribuição de pontos e posições por equipe;
- cortes de campeão/G4/G6/Top-8/Z4;
- faixa de probabilidade por pontuação-alvo;
- dificuldade da tabela restante e xPts dos jogos;
- identificação de confrontos diretos;
- trajetória média rodada a rodada;
- configuração explícita da hipótese de "motivação/desespero".

Premissas esperadas:
- df_current_standings contém: Team, Pos, PTS, V, SG, GP, GC, J
- df_fixtures contém: Home, Away e preferencialmente Round_Num
- df histórico contém Home, Away, Date, gols e Elo.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd


# ======================================================================================
# CONFIGURAÇÕES
# ======================================================================================

DEFAULT_TIEBREAKERS = ["points", "wins", "goal_difference", "goals_for"]


@dataclass(frozen=True)
class MotivationConfig:
    """
    Hipótese opcional de motivação/desespero expressa em pontos Elo.

    Por padrão fica DESLIGADA, pois esses bônus são hipóteses comportamentais e não
    devem entrar em uma simulação estatística sem validação histórica.
    """

    enabled: bool = False
    start_round: int = 28
    z4_bonus: float = 35.0
    near_z4_bonus: float = 20.0
    dead_zone_penalty: float = 25.0
    dead_zone_min_pos: int = 9
    dead_zone_max_pos: int = 13
    near_z4_positions: Tuple[int, int] = (15, 16)


# ======================================================================================
# VALIDAÇÕES / UTILITÁRIOS
# ======================================================================================


def _require_columns(df: pd.DataFrame, columns: Iterable[str], df_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} não possui as colunas obrigatórias: {missing}")


def _normalize_team_name(value: object) -> str:
    return str(value).strip()


def _filter_sub_league(df: pd.DataFrame, sub_league: Optional[str]) -> pd.DataFrame:
    if sub_league and "Sub_League" in df.columns:
        target = str(sub_league).strip().lower()
        return df[df["Sub_League"].astype(str).str.strip().str.lower() == target].copy()
    return df.copy()


def _normalize_tiebreakers(tiebreakers: Optional[Sequence[str]]) -> List[str]:
    aliases = {
        "points": "points",
        "pts": "points",
        "pontos": "points",
        "wins": "wins",
        "v": "wins",
        "vitorias": "wins",
        "vitórias": "wins",
        "goal_difference": "goal_difference",
        "sg": "goal_difference",
        "saldo": "goal_difference",
        "goals_for": "goals_for",
        "gp": "goals_for",
        "gols_pro": "goals_for",
        "gols_pró": "goals_for",
    }

    raw = list(tiebreakers) if tiebreakers is not None else list(DEFAULT_TIEBREAKERS)
    normalized = []
    for item in raw:
        key = str(item).strip().lower()
        if key not in aliases:
            raise ValueError(
                f"Desempate desconhecido: {item!r}. "
                f"Use points, wins, goal_difference e/ou goals_for."
            )
        normalized.append(aliases[key])

    if "points" not in normalized:
        normalized.insert(0, "points")

    # remove duplicados preservando ordem
    return list(dict.fromkeys(normalized))


# ======================================================================================
# ELO ATUAL E PARÂMETROS DA LIGA
# ======================================================================================


def get_latest_team_elos(
    df: pd.DataFrame,
    sub_league: Optional[str] = "Serie A",
    home_col: str = "Home",
    away_col: str = "Away",
    date_col: str = "Date",
    home_elo_post_col: str = "Home_Elo_Post",
    away_elo_post_col: str = "Away_Elo_Post",
    home_elo_pre_col: str = "Home_Elo_Pre",
    away_elo_pre_col: str = "Away_Elo_Pre",
    warn_if_pre_only: bool = True,
) -> Dict[str, float]:
    """
    Recupera o Elo mais recente de cada equipe.

    Prioridade:
    1) Home_Elo_Post / Away_Elo_Post, quando existirem;
    2) Home_Elo_Pre / Away_Elo_Pre como fallback.

    O fallback PRE fica, por definição, um jogo atrás do estado pós-última-partida.
    """
    _require_columns(df, [home_col, away_col, date_col], "df histórico")

    work = _filter_sub_league(df, sub_league)
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)

    has_post = home_elo_post_col in work.columns and away_elo_post_col in work.columns
    has_pre = home_elo_pre_col in work.columns and away_elo_pre_col in work.columns

    if not has_post and not has_pre:
        raise ValueError(
            "Não encontrei par de colunas Elo pós-jogo nem pré-jogo no histórico."
        )

    if has_post:
        h_elo_col, a_elo_col = home_elo_post_col, away_elo_post_col
    else:
        h_elo_col, a_elo_col = home_elo_pre_col, away_elo_pre_col
        if warn_if_pre_only:
            warnings.warn(
                "Usando Elo PRE do último jogo como fallback. Isso fica um jogo atrás "
                "do Elo pós-jogo atual. Se houver colunas *_Elo_Post, prefira utilizá-las.",
                RuntimeWarning,
            )

    latest: Dict[str, float] = {}
    for row in work[[home_col, away_col, h_elo_col, a_elo_col]].itertuples(index=False, name=None):
        h_team, a_team, h_elo, a_elo = row
        h_team = _normalize_team_name(h_team)
        a_team = _normalize_team_name(a_team)
        if pd.notna(h_elo):
            latest[h_team] = float(h_elo)
        if pd.notna(a_elo):
            latest[a_team] = float(a_elo)

    return latest


def calibrate_league_parameters(
    df: pd.DataFrame,
    sub_league: Optional[str] = "Serie A",
    home_score_col: str = "Home_Score",
    away_score_col: str = "Away_Score",
    date_col: str = "Date",
    history_start: Optional[str] = None,
    history_end: Optional[str] = None,
    home_adv_elo: float = 65.0,
    elo_sensitivity: float = 0.0025,
) -> Dict[str, float]:
    """
    Calcula as médias de gols do recorte histórico e mantém explicitamente os dois
    hiperparâmetros Elo do modelo.

    Observação: home_adv_elo e elo_sensitivity NÃO são estimados aqui. Eles devem ser
    validados/calibrados em backtest se você quiser tratá-los como parâmetros ótimos.
    """
    _require_columns(df, [home_score_col, away_score_col], "df histórico")

    work = _filter_sub_league(df, sub_league)

    if date_col in work.columns and (history_start is not None or history_end is not None):
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        if history_start is not None:
            work = work[work[date_col] >= pd.Timestamp(history_start)]
        if history_end is not None:
            work = work[work[date_col] <= pd.Timestamp(history_end)]

    work = work.dropna(subset=[home_score_col, away_score_col])
    if work.empty:
        warnings.warn("Recorte histórico vazio; usando médias fallback 1.45 / 0.95.")
        avg_home_goals, avg_away_goals = 1.45, 0.95
    else:
        avg_home_goals = float(pd.to_numeric(work[home_score_col], errors="coerce").mean())
        avg_away_goals = float(pd.to_numeric(work[away_score_col], errors="coerce").mean())

    return {
        "avg_home_goals": avg_home_goals,
        "avg_away_goals": avg_away_goals,
        "home_adv_elo": float(home_adv_elo),
        "elo_sensitivity": float(elo_sensitivity),
    }


# ======================================================================================
# MODELO DE GOLS / PROBABILIDADES DE PARTIDA
# ======================================================================================


def expected_goal_lambdas(
    home_elo: float,
    away_elo: float,
    params: Mapping[str, float],
    home_motivation_elo: float = 0.0,
    away_motivation_elo: float = 0.0,
) -> Tuple[float, float]:
    effective_home = float(home_elo) + float(params["home_adv_elo"]) + float(home_motivation_elo)
    effective_away = float(away_elo) + float(away_motivation_elo)
    elo_diff = effective_home - effective_away

    lambda_home = float(params["avg_home_goals"]) * np.exp(float(params["elo_sensitivity"]) * elo_diff)
    lambda_away = float(params["avg_away_goals"]) * np.exp(-float(params["elo_sensitivity"]) * elo_diff)

    return float(np.clip(lambda_home, 0.2, 4.5)), float(np.clip(lambda_away, 0.2, 4.5))


def match_outcome_probabilities(
    home_elo: float,
    away_elo: float,
    params: Mapping[str, float],
    max_goals: int = 12,
) -> Dict[str, float]:
    """Probabilidades 1X2 derivadas do mesmo modelo Poisson usado na simulação."""
    lh, la = expected_goal_lambdas(home_elo, away_elo, params)

    ph = np.array([exp(-lh) * (lh ** k) / factorial(k) for k in range(max_goals + 1)], dtype=float)
    pa = np.array([exp(-la) * (la ** k) / factorial(k) for k in range(max_goals + 1)], dtype=float)
    matrix = np.outer(ph, pa)
    matrix /= matrix.sum()

    p_home = float(np.tril(matrix, k=-1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, k=1).sum())

    return {
        "Home_Win_%": 100.0 * p_home,
        "Draw_%": 100.0 * p_draw,
        "Away_Win_%": 100.0 * p_away,
        "Home_xPts": 3.0 * p_home + p_draw,
        "Away_xPts": 3.0 * p_away + p_draw,
        "Lambda_Home": lh,
        "Lambda_Away": la,
    }


# ======================================================================================
# RANKING VETORIZADO
# ======================================================================================


def _rank_matrix(
    pts: np.ndarray,
    wins: np.ndarray,
    gd: np.ndarray,
    gf: np.ndarray,
    tiebreakers: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna:
    - order: índices dos times ordenados em cada simulação [n_sim, n_teams]
    - positions: posição (1..N) de cada time em cada simulação [n_sim, n_teams]
    """
    arrays = {
        "points": pts,
        "wins": wins,
        "goal_difference": gd,
        "goals_for": gf,
    }
    keys = tuple(-arrays[c] for c in reversed(tiebreakers))
    order = np.lexsort(keys, axis=1)

    positions = np.empty_like(order, dtype=np.int16)
    row_idx = np.arange(order.shape[0])[:, None]
    positions[row_idx, order] = np.arange(1, order.shape[1] + 1, dtype=np.int16)
    return order, positions


def _motivation_adjustment(
    home_pos: np.ndarray,
    away_pos: np.ndarray,
    config: MotivationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    h = np.zeros(home_pos.shape[0], dtype=float)
    a = np.zeros(away_pos.shape[0], dtype=float)

    h += np.where(home_pos >= 17, config.z4_bonus, 0.0)
    a += np.where(away_pos >= 17, config.z4_bonus, 0.0)

    h += np.where(np.isin(home_pos, config.near_z4_positions), config.near_z4_bonus, 0.0)
    a += np.where(np.isin(away_pos, config.near_z4_positions), config.near_z4_bonus, 0.0)

    h_dead = (home_pos >= config.dead_zone_min_pos) & (home_pos <= config.dead_zone_max_pos) & (away_pos >= 16)
    a_dead = (away_pos >= config.dead_zone_min_pos) & (away_pos <= config.dead_zone_max_pos) & (home_pos >= 16)
    h -= np.where(h_dead, config.dead_zone_penalty, 0.0)
    a -= np.where(a_dead, config.dead_zone_penalty, 0.0)

    return h, a


# ======================================================================================
# MONTE CARLO PRINCIPAL
# ======================================================================================


def run_monte_carlo_season(
    df_current_standings: pd.DataFrame,
    df_fixtures: pd.DataFrame,
    latest_elos: Dict[str, float],
    params: Mapping[str, float],
    n_simulations: int = 50_000,
    tiebreakers: Optional[Sequence[str]] = None,
    motivation: Optional[MotivationConfig] = None,
    random_seed: Optional[int] = 42,
    collect_round_trajectory: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Simula toda a reta final e devolve um pacote de relatórios.

    Retorno:
        {
          'resumo_times': ...,
          'prob_posicao': ...,
          'distribuicao_pontos': ...,
          'cortes': ...,
          'prob_por_pontuacao': ...,
          'trajetoria_rodadas': ...,
          'sim_meta': ...,
        }
    """
    if n_simulations <= 0:
        raise ValueError("n_simulations precisa ser > 0.")

    motivation = motivation or MotivationConfig()
    tb = _normalize_tiebreakers(tiebreakers)

    required_standings = ["Team", "Pos", "PTS", "V", "SG", "GP", "GC", "J"]
    _require_columns(df_current_standings, required_standings, "df_current_standings")
    _require_columns(df_fixtures, ["Home", "Away"], "df_fixtures")

    standings = df_current_standings.copy()
    standings["Team"] = standings["Team"].map(_normalize_team_name)
    standings = standings.sort_values("Pos").reset_index(drop=True)

    teams = standings["Team"].tolist()
    n_teams = len(teams)
    if n_teams < 4:
        raise ValueError("A tabela precisa ter pelo menos 4 equipes.")

    team_to_idx = {t: i for i, t in enumerate(teams)}

    fixtures = df_fixtures.copy()
    fixtures["Home"] = fixtures["Home"].map(_normalize_team_name)
    fixtures["Away"] = fixtures["Away"].map(_normalize_team_name)

    unknown = sorted((set(fixtures["Home"]) | set(fixtures["Away"])) - set(teams))
    if unknown:
        raise ValueError(f"Há equipes nos fixtures que não estão na tabela atual: {unknown}")

    if "Round_Num" not in fixtures.columns:
        warnings.warn(
            "df_fixtures não possui Round_Num. Todos os jogos serão tratados como uma única rodada, "
            "e a motivação dinâmica por rodada ficará limitada."
        )
        fixtures["Round_Num"] = 999

    fixtures["Round_Num"] = pd.to_numeric(fixtures["Round_Num"], errors="coerce").fillna(999).astype(int)

    sort_cols = ["Round_Num"]
    if "Date" in fixtures.columns:
        fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
        sort_cols.append("Date")
    fixtures = fixtures.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    all_elos = [float(v) for v in latest_elos.values() if pd.notna(v)]
    fallback_elo = float(np.mean(all_elos)) if all_elos else 1500.0
    elo_vec = np.array([float(latest_elos.get(t, fallback_elo)) for t in teams], dtype=float)

    missing_elo_teams = [t for t in teams if t not in latest_elos or pd.isna(latest_elos.get(t))]
    if missing_elo_teams:
        warnings.warn(
            f"{len(missing_elo_teams)} equipe(s) sem Elo receberam a média {fallback_elo:.1f}: "
            f"{missing_elo_teams}"
        )

    # Estado das simulações: shape [n_simulations, n_teams]
    def tile_col(col: str, dtype=np.int32) -> np.ndarray:
        base = pd.to_numeric(standings[col], errors="raise").to_numpy(dtype=dtype)
        return np.tile(base, (n_simulations, 1))

    pts = tile_col("PTS")
    wins = tile_col("V")
    gd = tile_col("SG")
    gf = tile_col("GP")
    ga = tile_col("GC")
    played = tile_col("J")

    rng = np.random.default_rng(random_seed)

    # Acumuladores da trajetória rodada a rodada.
    trajectory_records: List[dict] = []

    unique_rounds = fixtures["Round_Num"].drop_duplicates().tolist()
    total_games = len(fixtures)
    print(
        f"Rodando {n_simulations:,} simulações para {total_games} jogos restantes "
        f"em {len(unique_rounds)} rodada(s)..."
    )

    for rnd in unique_rounds:
        # Retrato único da classificação no começo da rodada.
        _, round_start_pos = _rank_matrix(pts, wins, gd, gf, tb)

        round_matches = fixtures[fixtures["Round_Num"] == rnd]

        for match in round_matches.itertuples(index=False):
            h_team = match.Home
            a_team = match.Away
            hi = team_to_idx[h_team]
            ai = team_to_idx[a_team]

            h_mot = np.zeros(n_simulations, dtype=float)
            a_mot = np.zeros(n_simulations, dtype=float)

            if motivation.enabled and rnd >= motivation.start_round:
                h_mot, a_mot = _motivation_adjustment(
                    round_start_pos[:, hi],
                    round_start_pos[:, ai],
                    motivation,
                )

            effective_home = elo_vec[hi] + float(params["home_adv_elo"]) + h_mot
            effective_away = elo_vec[ai] + a_mot
            elo_diff = effective_home - effective_away

            lh = float(params["avg_home_goals"]) * np.exp(float(params["elo_sensitivity"]) * elo_diff)
            la = float(params["avg_away_goals"]) * np.exp(-float(params["elo_sensitivity"]) * elo_diff)
            lh = np.clip(lh, 0.2, 4.5)
            la = np.clip(la, 0.2, 4.5)

            h_goals = rng.poisson(lh)
            a_goals = rng.poisson(la)

            played[:, hi] += 1
            played[:, ai] += 1
            gf[:, hi] += h_goals
            ga[:, hi] += a_goals
            gd[:, hi] += h_goals - a_goals
            gf[:, ai] += a_goals
            ga[:, ai] += h_goals
            gd[:, ai] += a_goals - h_goals

            home_win = h_goals > a_goals
            draw = h_goals == a_goals
            away_win = h_goals < a_goals

            pts[:, hi] += 3 * home_win + draw
            pts[:, ai] += 3 * away_win + draw
            wins[:, hi] += home_win
            wins[:, ai] += away_win

        if collect_round_trajectory:
            _, end_pos = _rank_matrix(pts, wins, gd, gf, tb)
            mean_pts = pts.mean(axis=0)
            mean_pos = end_pos.mean(axis=0)
            p_g4 = (end_pos <= min(4, n_teams)).mean(axis=0) * 100
            p_g6 = (end_pos <= min(6, n_teams)).mean(axis=0) * 100
            p_z4 = (end_pos >= max(1, n_teams - 3)).mean(axis=0) * 100

            for i, team in enumerate(teams):
                trajectory_records.append(
                    {
                        "Round_Num": int(rnd),
                        "Team": team,
                        "Pts_Medio_Apos_Rodada": round(float(mean_pts[i]), 2),
                        "Pos_Media_Apos_Rodada": round(float(mean_pos[i]), 2),
                        "Top4_Se_Terminasse_Aqui_%": round(float(p_g4[i]), 2),
                        "Top6_Se_Terminasse_Aqui_%": round(float(p_g6[i]), 2),
                        "Z4_Se_Terminasse_Aqui_%": round(float(p_z4[i]), 2),
                    }
                )

    final_order, final_pos = _rank_matrix(pts, wins, gd, gf, tb)

    reports = _build_simulation_reports(
        standings=standings,
        teams=teams,
        pts=pts,
        final_pos=final_pos,
        final_order=final_order,
        n_simulations=n_simulations,
        trajectory_records=trajectory_records,
        total_games=total_games,
        tiebreakers=tb,
        motivation=motivation,
        random_seed=random_seed,
    )
    return reports


# ======================================================================================
# RELATÓRIOS DA SIMULAÇÃO
# ======================================================================================


def _build_simulation_reports(
    standings: pd.DataFrame,
    teams: List[str],
    pts: np.ndarray,
    final_pos: np.ndarray,
    final_order: np.ndarray,
    n_simulations: int,
    trajectory_records: List[dict],
    total_games: int,
    tiebreakers: Sequence[str],
    motivation: MotivationConfig,
    random_seed: Optional[int],
) -> Dict[str, pd.DataFrame]:
    n_teams = len(teams)
    current_pos = standings.set_index("Team")["Pos"].to_dict()
    current_pts = standings.set_index("Team")["PTS"].to_dict()

    # ------------------------- Resumo por equipe -------------------------
    summary = []
    point_dist = []
    pos_matrix_rows = []

    for i, team in enumerate(teams):
        team_pts = pts[:, i]
        team_pos = final_pos[:, i]

        q_pts = np.percentile(team_pts, [10, 25, 50, 75, 90])
        q_pos = np.percentile(team_pos, [10, 25, 50, 75, 90])

        summary.append(
            {
                "Pos_Atual": int(current_pos[team]),
                "Team": team,
                "Pts_Atual": int(current_pts[team]),
                "Pts_Medio": round(float(team_pts.mean()), 2),
                "Pts_Mediana": round(float(np.median(team_pts)), 1),
                "Pos_Media": round(float(team_pos.mean()), 2),
                "Pos_Mediana": round(float(np.median(team_pos)), 1),
                "Campeao_%": round(float((team_pos == 1).mean() * 100), 2),
                "G4_%": round(float((team_pos <= min(4, n_teams)).mean() * 100), 2),
                "G6_%": round(float((team_pos <= min(6, n_teams)).mean() * 100), 2),
                "Top8_%": round(float((team_pos <= min(8, n_teams)).mean() * 100), 2),
                "Top10_%": round(float((team_pos <= min(10, n_teams)).mean() * 100), 2),
                "Z4_%": round(float((team_pos >= max(1, n_teams - 3)).mean() * 100), 2),
                "Melhora_Posicao_%": round(float((team_pos < int(current_pos[team])).mean() * 100), 2),
                "Mantem_Posicao_%": round(float((team_pos == int(current_pos[team])).mean() * 100), 2),
                "Piora_Posicao_%": round(float((team_pos > int(current_pos[team])).mean() * 100), 2),
                "DP_Pontos": round(float(team_pts.std()), 2),
                "DP_Posicao": round(float(team_pos.std()), 2),
            }
        )

        point_dist.append(
            {
                "Team": team,
                "Pts_Atual": int(current_pts[team]),
                "Pts_Min_Simulado": int(team_pts.min()),
                "P10_Pts": round(float(q_pts[0]), 1),
                "P25_Pts": round(float(q_pts[1]), 1),
                "P50_Pts": round(float(q_pts[2]), 1),
                "P75_Pts": round(float(q_pts[3]), 1),
                "P90_Pts": round(float(q_pts[4]), 1),
                "Pts_Max_Simulado": int(team_pts.max()),
                "P10_Pos": round(float(q_pos[0]), 1),
                "P25_Pos": round(float(q_pos[1]), 1),
                "P50_Pos": round(float(q_pos[2]), 1),
                "P75_Pos": round(float(q_pos[3]), 1),
                "P90_Pos": round(float(q_pos[4]), 1),
            }
        )

        row = {"Team": team}
        for pos in range(1, n_teams + 1):
            row[f"{pos}º_%"] = round(float((team_pos == pos).mean() * 100), 2)
        pos_matrix_rows.append(row)

    df_summary = pd.DataFrame(summary).sort_values(
        ["Campeao_%", "Pts_Medio", "Pos_Media"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    df_summary.insert(0, "Pos_Projetada", np.arange(1, len(df_summary) + 1))

    df_point_dist = pd.DataFrame(point_dist).sort_values("P50_Pts", ascending=False).reset_index(drop=True)
    df_pos_matrix = pd.DataFrame(pos_matrix_rows)
    team_order = df_summary["Team"].tolist()
    df_pos_matrix["__ord"] = df_pos_matrix["Team"].map({t: i for i, t in enumerate(team_order)})
    df_pos_matrix = df_pos_matrix.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    # ------------------------- Linhas de corte -------------------------
    cutoff_specs = [("Campeao", 1)]
    if n_teams >= 4:
        cutoff_specs.append(("G4", 4))
    if n_teams >= 6:
        cutoff_specs.append(("G6", 6))
    if n_teams >= 8:
        cutoff_specs.append(("Top8", 8))
    cutoff_specs.append(("Ultimo_Salvo", n_teams - 4))
    cutoff_specs.append(("Primeiro_Rebaixado", n_teams - 3))

    cutoff_records = []
    cutoff_values: Dict[str, np.ndarray] = {}
    rows = np.arange(n_simulations)

    for label, pos in cutoff_specs:
        team_idx_at_pos = final_order[:, pos - 1]
        vals = pts[rows, team_idx_at_pos]
        cutoff_values[label] = vals
        q = np.percentile(vals, [10, 25, 50, 75, 90])
        cutoff_records.append(
            {
                "Objetivo": label,
                "Posicao": pos,
                "Pontos_Medios": round(float(vals.mean()), 2),
                "P10": round(float(q[0]), 1),
                "P25": round(float(q[1]), 1),
                "P50": round(float(q[2]), 1),
                "P75": round(float(q[3]), 1),
                "P90": round(float(q[4]), 1),
                "Min": int(vals.min()),
                "Max": int(vals.max()),
            }
        )

    df_cutoffs = pd.DataFrame(cutoff_records)

    # ------------------------- Probabilidade por pontuação -------------------------
    min_target = max(0, int(pts.min()) - 2)
    max_target = int(pts.max()) + 2
    score_rows = []

    champ_cut = cutoff_values["Campeao"]
    g4_cut = cutoff_values.get("G4")
    g6_cut = cutoff_values.get("G6")
    first_rel_cut = cutoff_values["Primeiro_Rebaixado"]

    for p in range(min_target, max_target + 1):
        rec = {"Pontos_Finais": p}

        # Para Top-X, pontuar MAIS que o corte garante ficar acima por pontos.
        # Igualar o corte deixa o resultado dependente dos desempates.
        rec["Titulo_Garantido_por_Pontos_%"] = round(float((p > champ_cut).mean() * 100), 2)
        rec["Titulo_Possivel_com_Empate_Corte_%"] = round(float((p >= champ_cut).mean() * 100), 2)

        if g4_cut is not None:
            rec["G4_Garantido_por_Pontos_%"] = round(float((p > g4_cut).mean() * 100), 2)
            rec["G4_Possivel_com_Empate_Corte_%"] = round(float((p >= g4_cut).mean() * 100), 2)
        if g6_cut is not None:
            rec["G6_Garantido_por_Pontos_%"] = round(float((p > g6_cut).mean() * 100), 2)
            rec["G6_Possivel_com_Empate_Corte_%"] = round(float((p >= g6_cut).mean() * 100), 2)

        # Permanência: se a pontuação é maior que a do 17º, salva por pontos.
        # Se empata com o 17º, passa a depender dos desempates.
        rec["Salvacao_Garantida_por_Pontos_%"] = round(float((p > first_rel_cut).mean() * 100), 2)
        rec["Salvacao_Possivel_com_Empate_Corte_%"] = round(float((p >= first_rel_cut).mean() * 100), 2)
        score_rows.append(rec)

    df_prob_score = pd.DataFrame(score_rows)

    # ------------------------- Trajetória -------------------------
    df_trajectory = pd.DataFrame(trajectory_records)

    # ------------------------- Metadados -------------------------
    df_meta = pd.DataFrame(
        [
            {
                "N_Simulacoes": n_simulations,
                "N_Times": n_teams,
                "Jogos_Restantes": total_games,
                "Desempates": " > ".join(tiebreakers),
                "Motivacao_Ativa": motivation.enabled,
                "Motivacao_Inicio_Rodada": motivation.start_round if motivation.enabled else None,
                "Random_Seed": random_seed,
            }
        ]
    )

    return {
        "resumo_times": df_summary,
        "prob_posicao": df_pos_matrix,
        "distribuicao_pontos": df_point_dist,
        "cortes": df_cutoffs,
        "prob_por_pontuacao": df_prob_score,
        "trajetoria_rodadas": df_trajectory,
        "sim_meta": df_meta,
    }


# ======================================================================================
# DIFICULDADE DA TABELA E CONFRONTOS DIRETOS
# ======================================================================================


def analyze_remaining_schedule(
    df_current_standings: pd.DataFrame,
    df_fixtures: pd.DataFrame,
    latest_elos: Dict[str, float],
    params: Mapping[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna:
    1) dificuldade agregada por equipe;
    2) probabilidades/xPts jogo a jogo.
    """
    _require_columns(df_current_standings, ["Team", "Pos"], "df_current_standings")
    _require_columns(df_fixtures, ["Home", "Away"], "df_fixtures")

    standings = df_current_standings.copy()
    standings["Team"] = standings["Team"].map(_normalize_team_name)
    pos_map = standings.set_index("Team")["Pos"].to_dict()

    fixtures = df_fixtures.copy()
    fixtures["Home"] = fixtures["Home"].map(_normalize_team_name)
    fixtures["Away"] = fixtures["Away"].map(_normalize_team_name)
    if "Round_Num" not in fixtures.columns:
        fixtures["Round_Num"] = np.nan

    all_elos = [float(v) for v in latest_elos.values() if pd.notna(v)]
    fallback = float(np.mean(all_elos)) if all_elos else 1500.0

    game_rows = []
    agg: Dict[str, dict] = {
        t: {
            "Team": t,
            "Jogos_Restantes": 0,
            "Casa": 0,
            "Fora": 0,
            "Soma_Elo_Adversario": 0.0,
            "Soma_Elo_Ajustado_Mando": 0.0,
            "xPts_Restantes": 0.0,
            "Contra_Top6": 0,
            "Contra_Z4_Atual": 0,
        }
        for t in standings["Team"]
    }

    for m in fixtures.itertuples(index=False):
        h, a = m.Home, m.Away
        h_elo = float(latest_elos.get(h, fallback))
        a_elo = float(latest_elos.get(a, fallback))
        probs = match_outcome_probabilities(h_elo, a_elo, params)

        game_rows.append(
            {
                "Round_Num": getattr(m, "Round_Num", np.nan),
                "Home": h,
                "Away": a,
                "Home_Elo": round(h_elo, 1),
                "Away_Elo": round(a_elo, 1),
                "Home_Win_%": round(probs["Home_Win_%"], 2),
                "Draw_%": round(probs["Draw_%"], 2),
                "Away_Win_%": round(probs["Away_Win_%"], 2),
                "Home_xPts": round(probs["Home_xPts"], 3),
                "Away_xPts": round(probs["Away_xPts"], 3),
                "Lambda_Home": round(probs["Lambda_Home"], 3),
                "Lambda_Away": round(probs["Lambda_Away"], 3),
            }
        )

        # Para o mandante, jogar em casa reduz a dificuldade relativa do adversário.
        agg[h]["Jogos_Restantes"] += 1
        agg[h]["Casa"] += 1
        agg[h]["Soma_Elo_Adversario"] += a_elo
        agg[h]["Soma_Elo_Ajustado_Mando"] += a_elo - float(params["home_adv_elo"])
        agg[h]["xPts_Restantes"] += probs["Home_xPts"]
        agg[h]["Contra_Top6"] += int(pos_map.get(a, 99) <= 6)
        agg[h]["Contra_Z4_Atual"] += int(pos_map.get(a, 0) >= len(standings) - 3)

        # Para o visitante, o Elo do adversário é acrescido da vantagem de mando.
        agg[a]["Jogos_Restantes"] += 1
        agg[a]["Fora"] += 1
        agg[a]["Soma_Elo_Adversario"] += h_elo
        agg[a]["Soma_Elo_Ajustado_Mando"] += h_elo + float(params["home_adv_elo"])
        agg[a]["xPts_Restantes"] += probs["Away_xPts"]
        agg[a]["Contra_Top6"] += int(pos_map.get(h, 99) <= 6)
        agg[a]["Contra_Z4_Atual"] += int(pos_map.get(h, 0) >= len(standings) - 3)

    schedule_rows = []
    for team, d in agg.items():
        n = d["Jogos_Restantes"]
        if n:
            avg_opp = d["Soma_Elo_Adversario"] / n
            avg_adj = d["Soma_Elo_Ajustado_Mando"] / n
            xpts_game = d["xPts_Restantes"] / n
        else:
            avg_opp = avg_adj = xpts_game = np.nan

        schedule_rows.append(
            {
                "Team": team,
                "Pos_Atual": int(pos_map[team]),
                "Jogos_Restantes": n,
                "Casa": d["Casa"],
                "Fora": d["Fora"],
                "Elo_Medio_Adversarios": round(avg_opp, 1) if pd.notna(avg_opp) else np.nan,
                "Elo_Medio_Ajustado_Mando": round(avg_adj, 1) if pd.notna(avg_adj) else np.nan,
                "xPts_Restantes": round(d["xPts_Restantes"], 2),
                "xPts_por_Jogo": round(xpts_game, 3) if pd.notna(xpts_game) else np.nan,
                "Contra_Top6": d["Contra_Top6"],
                "Contra_Z4_Atual": d["Contra_Z4_Atual"],
            }
        )

    df_schedule = pd.DataFrame(schedule_rows)
    # Menor xPts/jogo = tabela mais difícil segundo o próprio modelo.
    df_schedule = df_schedule.sort_values(
        ["xPts_por_Jogo", "Elo_Medio_Ajustado_Mando"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    df_schedule.insert(0, "Rank_Dificuldade", np.arange(1, len(df_schedule) + 1))

    return df_schedule, pd.DataFrame(game_rows)


def identify_direct_clashes(
    df_current_standings: pd.DataFrame,
    df_fixtures: pd.DataFrame,
    title_zone: int = 6,
    continental_zone: int = 10,
    relegation_from: int = 15,
) -> pd.DataFrame:
    """Marca confrontos restantes relevantes para as mesmas zonas da tabela atual."""
    _require_columns(df_current_standings, ["Team", "Pos"], "df_current_standings")
    _require_columns(df_fixtures, ["Home", "Away"], "df_fixtures")

    standings = df_current_standings.copy()
    standings["Team"] = standings["Team"].map(_normalize_team_name)
    pos = standings.set_index("Team")["Pos"].astype(int).to_dict()

    fixtures = df_fixtures.copy()
    fixtures["Home"] = fixtures["Home"].map(_normalize_team_name)
    fixtures["Away"] = fixtures["Away"].map(_normalize_team_name)
    if "Round_Num" not in fixtures.columns:
        fixtures["Round_Num"] = np.nan

    rows = []
    for m in fixtures.itertuples(index=False):
        h, a = m.Home, m.Away
        hp, ap = pos.get(h), pos.get(a)
        if hp is None or ap is None:
            continue

        categories = []
        if hp <= title_zone and ap <= title_zone:
            categories.append("Título/G6")
        if hp <= continental_zone and ap <= continental_zone:
            categories.append("Parte de cima")
        if hp >= relegation_from and ap >= relegation_from:
            categories.append("Z4/Permanência")

        if categories:
            rows.append(
                {
                    "Round_Num": getattr(m, "Round_Num", np.nan),
                    "Home": h,
                    "Pos_Home_Atual": hp,
                    "Away": a,
                    "Pos_Away_Atual": ap,
                    "Categoria": " + ".join(categories),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["Round_Num", "Home", "Pos_Home_Atual", "Away", "Pos_Away_Atual", "Categoria"]
        )

    return pd.DataFrame(rows).sort_values(["Round_Num", "Categoria"], kind="stable").reset_index(drop=True)


# ======================================================================================
# ORQUESTRADOR — UMA CHAMADA PARA TODOS OS RELATÓRIOS
# ======================================================================================


def build_season_simulation_report(
    df_current_standings: pd.DataFrame,
    df_fixtures: pd.DataFrame,
    latest_elos: Dict[str, float],
    params: Mapping[str, float],
    n_simulations: int = 50_000,
    tiebreakers: Optional[Sequence[str]] = None,
    motivation: Optional[MotivationConfig] = None,
    random_seed: Optional[int] = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Executa a simulação e acrescenta os relatórios independentes da tabela restante.

    Exemplo:
        reports = build_season_simulation_report(
            df_current_standings=df_tabela,
            df_fixtures=df_restantes,
            latest_elos=latest_elos,
            params=params,
            n_simulations=50_000,
        )

        display(reports['resumo_times'])
        display(reports['prob_posicao'])
        display(reports['cortes'])
    """
    reports = run_monte_carlo_season(
        df_current_standings=df_current_standings,
        df_fixtures=df_fixtures,
        latest_elos=latest_elos,
        params=params,
        n_simulations=n_simulations,
        tiebreakers=tiebreakers,
        motivation=motivation,
        random_seed=random_seed,
        collect_round_trajectory=True,
    )

    df_difficulty, df_games = analyze_remaining_schedule(
        df_current_standings=df_current_standings,
        df_fixtures=df_fixtures,
        latest_elos=latest_elos,
        params=params,
    )
    df_clashes = identify_direct_clashes(df_current_standings, df_fixtures)

    reports["dificuldade_restante"] = df_difficulty
    reports["jogos_probabilidades"] = df_games
    reports["confrontos_diretos"] = df_clashes

    return reports


# ======================================================================================
# IMPRESSÃO RÁPIDA / DISPLAY NO COLAB
# ======================================================================================


def print_report_overview(reports: Mapping[str, pd.DataFrame], top_n: int = 20) -> None:
    """Mostra no console as tabelas principais sem exigir bibliotecas extras."""
    sections = [
        ("RESUMO DOS TIMES", "resumo_times"),
        ("CORTES DE PONTUAÇÃO", "cortes"),
        ("DIFICULDADE DA TABELA RESTANTE", "dificuldade_restante"),
        ("CONFRONTOS DIRETOS", "confrontos_diretos"),
    ]

    for title, key in sections:
        if key in reports:
            print("\n" + "=" * 110)
            print(title)
            print("=" * 110)
            df = reports[key]
            if df.empty:
                print("Sem registros.")
            else:
                print(df.head(top_n).to_string(index=False))


# ======================================================================================
# EXEMPLO DE USO (DESCOMENTE E AJUSTE OS NOMES DOS DATAFRAMES)
# ======================================================================================
#
# latest_elos = get_latest_team_elos(
#     df_backup,
#     sub_league="Serie A",
# )
#
# params = calibrate_league_parameters(
#     df_backup,
#     sub_league="Serie A",
#     history_start="2024-01-01",  # opcional
#     home_adv_elo=65.0,
#     elo_sensitivity=0.0025,
# )
#
# reports = build_season_simulation_report(
#     df_current_standings=df_tabela_atual,
#     df_fixtures=df_jogos_restantes,
#     latest_elos=latest_elos,
#     params=params,
#     n_simulations=50_000,
#     tiebreakers=["points", "wins", "goal_difference", "goals_for"],
#     motivation=MotivationConfig(enabled=False),  # ligue só se quiser testar essa hipótese
#     random_seed=42,
# )
#
# print_report_overview(reports)
#
# # Principais DataFrames:
# # reports["resumo_times"]
# # reports["prob_posicao"]
# # reports["distribuicao_pontos"]
# # reports["cortes"]
# # reports["prob_por_pontuacao"]
# # reports["trajetoria_rodadas"]
# # reports["dificuldade_restante"]
# # reports["jogos_probabilidades"]
# # reports["confrontos_diretos"]
# # reports["sim_meta"]

# ======================================================================================
# CALIBRAÇÃO PRO — ELO -> GOLS / 1X2 COM VALIDAÇÃO TEMPORAL
# ======================================================================================


def _calibration_time_weights(dates, half_life_days: int = 730, reference_date=None) -> np.ndarray:
    dates = pd.to_datetime(dates)
    if reference_date is None:
        reference_date = dates.max()
    reference_date = pd.Timestamp(reference_date)
    age_days = (reference_date - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
    if half_life_days is None or half_life_days <= 0:
        return np.ones(len(dates), dtype=float)
    return np.power(0.5, age_days / float(half_life_days))


def prepare_league_calibration_history(
    df: pd.DataFrame,
    sub_league: Optional[str] = "Serie A",
    calibration_start: Optional[str] = "2022-01-01",
    date_col: str = "Date",
    home_score_col: str = "Home_Score",
    away_score_col: str = "Away_Score",
    home_elo_col: str = "Home_Elo_Pre",
    away_elo_col: str = "Away_Elo_Pre",
) -> pd.DataFrame:
    """Prepara somente jogos historicamente elegíveis para calibrar Elo -> resultado."""
    required = [date_col, home_score_col, away_score_col, home_elo_col, away_elo_col]
    _require_columns(df, required, "df histórico")

    cols = required.copy()
    if "Sub_League" in df.columns:
        cols.append("Sub_League")

    work = df[cols].copy()
    if "Sub_League" in work.columns and sub_league is not None:
        target = str(sub_league).strip().lower()
        work = work[
            work["Sub_League"].astype(str).str.strip().str.lower().eq(target)
        ].copy()

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    for col in [home_score_col, away_score_col, home_elo_col, away_elo_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=required).copy()
    work = work[(work[home_score_col] >= 0) & (work[away_score_col] >= 0)].copy()

    if calibration_start is not None:
        work = work[work[date_col] >= pd.Timestamp(calibration_start)].copy()

    work = work.rename(
        columns={
            date_col: "Date",
            home_score_col: "Home_Score",
            away_score_col: "Away_Score",
            home_elo_col: "Home_Elo_Pre",
            away_elo_col: "Away_Elo_Pre",
        }
    )
    work["Year"] = work["Date"].dt.year.astype(int)
    work["Elo_Diff"] = work["Home_Elo_Pre"] - work["Away_Elo_Pre"]
    return work.sort_values("Date").reset_index(drop=True)


def fit_poisson_elo_params_given_beta(
    train: pd.DataFrame,
    beta: float,
    half_life_days: int = 730,
) -> Dict[str, float]:
    """
    Ajusta as médias-base para um beta fixo.

    Modelo:
      lambda_home = mu_home * exp(+beta * Elo_Diff)
      lambda_away = mu_away * exp(-beta * Elo_Diff)

    mu_home / mu_away já absorvem a diferença casa x fora. Portanto home_adv_elo=0,
    evitando duplicar mando através das médias de gols + bônus Elo manual.
    """
    if train.empty:
        raise ValueError("Treino vazio na calibração Elo.")

    d = train["Elo_Diff"].to_numpy(dtype=float)
    gh = train["Home_Score"].to_numpy(dtype=float)
    ga = train["Away_Score"].to_numpy(dtype=float)
    w = _calibration_time_weights(train["Date"], half_life_days, train["Date"].max())

    exp_h = np.exp(np.clip(float(beta) * d, -4.0, 4.0))
    exp_a = np.exp(np.clip(-float(beta) * d, -4.0, 4.0))

    denom_h = max(float(np.sum(w * exp_h)), 1e-12)
    denom_a = max(float(np.sum(w * exp_a)), 1e-12)

    mu_home = float(np.sum(w * gh) / denom_h)
    mu_away = float(np.sum(w * ga) / denom_a)

    return {
        "avg_home_goals": float(np.clip(mu_home, 0.30, 3.00)),
        "avg_away_goals": float(np.clip(mu_away, 0.20, 2.50)),
        "home_adv_elo": 0.0,
        "elo_sensitivity": float(beta),
    }


def _history_predict_1x2(
    df_eval: pd.DataFrame,
    params: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.stats import skellam

    d = df_eval["Elo_Diff"].to_numpy(dtype=float)
    beta = float(params["elo_sensitivity"])
    home_adv = float(params.get("home_adv_elo", 0.0))
    effective_diff = d + home_adv

    lh = float(params["avg_home_goals"]) * np.exp(np.clip(beta * effective_diff, -4.0, 4.0))
    la = float(params["avg_away_goals"]) * np.exp(np.clip(-beta * effective_diff, -4.0, 4.0))
    lh = np.clip(lh, 0.15, 5.0)
    la = np.clip(la, 0.15, 5.0)

    p_home = 1.0 - skellam.cdf(0, lh, la)
    p_draw = skellam.pmf(0, lh, la)
    p_away = skellam.cdf(-1, lh, la)

    probs = np.column_stack([p_home, p_draw, p_away])
    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, lh, la


def _history_actual_1x2(df_eval: pd.DataFrame) -> np.ndarray:
    gh = df_eval["Home_Score"].to_numpy(dtype=float)
    ga = df_eval["Away_Score"].to_numpy(dtype=float)
    return np.where(gh > ga, 0, np.where(gh == ga, 1, 2)).astype(int)


def evaluate_poisson_elo_history(
    df_eval: pd.DataFrame,
    params: Mapping[str, float],
) -> Dict[str, float]:
    probs, lh, la = _history_predict_1x2(df_eval, params)
    y = _history_actual_1x2(df_eval)
    rows = np.arange(len(y))
    logloss = -float(np.mean(np.log(probs[rows, y])))
    onehot = np.eye(3)[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    accuracy = float((probs.argmax(axis=1) == y).mean())
    return {
        "n": int(len(y)),
        "logloss": logloss,
        "brier": brier,
        "accuracy": accuracy,
        "real_home_goals": float(df_eval["Home_Score"].mean()),
        "pred_home_goals": float(np.mean(lh)),
        "real_away_goals": float(df_eval["Away_Score"].mean()),
        "pred_away_goals": float(np.mean(la)),
    }


def calibrate_poisson_elo_pro(
    df: pd.DataFrame,
    sub_league: Optional[str] = "Serie A",
    calibration_start: str = "2022-01-01",
    validation_years: Sequence[int] = (2023, 2024, 2025),
    half_life_days: int = 730,
    beta_min: float = 0.0,
    beta_max: float = 0.0035,
    beta_step: float = 0.00005,
) -> Dict[str, object]:
    """
    Escolhe elo_sensitivity via rolling temporal minimizando LogLoss 1X2 fora da amostra.

    Isso faz o próprio Brasileirão decidir o quanto diferenças de Elo importam. Se favoritos
    historicamente tropeçam muito, betas altos são penalizados na validação.
    """
    hist = prepare_league_calibration_history(
        df,
        sub_league=sub_league,
        calibration_start=calibration_start,
    )
    if len(hist) < 300:
        warnings.warn(f"A calibração possui apenas {len(hist)} jogos elegíveis.")

    beta_grid = np.round(
        np.arange(beta_min, beta_max + beta_step / 2.0, beta_step),
        8,
    )

    raw_records = []
    usable_years = []
    for year in [int(y) for y in validation_years]:
        train = hist[hist["Date"] < pd.Timestamp(f"{year}-01-01")].copy()
        valid = hist[hist["Year"] == year].copy()
        if len(train) < 200 or len(valid) < 50:
            continue
        usable_years.append(year)

        for beta in beta_grid:
            p = fit_poisson_elo_params_given_beta(train, float(beta), half_life_days)
            ev = evaluate_poisson_elo_history(valid, p)
            raw_records.append(
                {
                    "Year": year,
                    "Beta": float(beta),
                    "N": ev["n"],
                    "LogLoss": ev["logloss"],
                    "Brier": ev["brier"],
                }
            )

    if not raw_records:
        raise ValueError(
            "Nenhum fold temporal pôde ser usado. Revise calibration_start/validation_years."
        )

    df_raw = pd.DataFrame(raw_records)
    summary = []
    for beta, g in df_raw.groupby("Beta", sort=True):
        w = g["N"].to_numpy(dtype=float)
        summary.append(
            {
                "Beta": float(beta),
                "Jogos_Validacao": int(g["N"].sum()),
                "LogLoss_CV": float(np.average(g["LogLoss"], weights=w)),
                "Brier_CV": float(np.average(g["Brier"], weights=w)),
            }
        )

    df_beta = pd.DataFrame(summary).sort_values(
        ["LogLoss_CV", "Brier_CV"], ascending=[True, True]
    ).reset_index(drop=True)
    best_beta = float(df_beta.iloc[0]["Beta"])

    fold_rows = []
    old_rows = []
    for year in usable_years:
        train = hist[hist["Date"] < pd.Timestamp(f"{year}-01-01")].copy()
        valid = hist[hist["Year"] == year].copy()

        p_best = fit_poisson_elo_params_given_beta(train, best_beta, half_life_days)
        ev_best = evaluate_poisson_elo_history(valid, p_best)
        fold_rows.append(
            {
                "Ano_Validacao": year,
                "Treino_Ate": train["Date"].max().date(),
                "Jogos": ev_best["n"],
                "LogLoss": round(ev_best["logloss"], 6),
                "Brier": round(ev_best["brier"], 6),
                "Acuracia_%": round(ev_best["accuracy"] * 100, 2),
                "Gol_Home_Real": round(ev_best["real_home_goals"], 3),
                "Gol_Home_Modelo": round(ev_best["pred_home_goals"], 3),
                "Gol_Away_Real": round(ev_best["real_away_goals"], 3),
                "Gol_Away_Modelo": round(ev_best["pred_away_goals"], 3),
            }
        )

        p_old = {
            "avg_home_goals": float(train["Home_Score"].mean()),
            "avg_away_goals": float(train["Away_Score"].mean()),
            "home_adv_elo": 65.0,
            "elo_sensitivity": 0.0025,
        }
        ev_old = evaluate_poisson_elo_history(valid, p_old)
        old_rows.append(
            {
                "Ano_Validacao": year,
                "Jogos": ev_old["n"],
                "LogLoss": ev_old["logloss"],
                "Brier": ev_old["brier"],
            }
        )

    df_folds = pd.DataFrame(fold_rows)
    df_old = pd.DataFrame(old_rows)
    w_old = df_old["Jogos"].to_numpy(dtype=float)
    old_ll = float(np.average(df_old["LogLoss"], weights=w_old))
    old_br = float(np.average(df_old["Brier"], weights=w_old))

    comparison = pd.DataFrame(
        [
            {
                "Modelo": "Antigo (+65 Elo, beta=0.0025)",
                "Beta": 0.0025,
                "Home_Adv_Elo_Extra": 65.0,
                "LogLoss_CV": round(old_ll, 6),
                "Brier_CV": round(old_br, 6),
            },
            {
                "Modelo": "Calibrado (rolling temporal)",
                "Beta": best_beta,
                "Home_Adv_Elo_Extra": 0.0,
                "LogLoss_CV": round(float(df_beta.iloc[0]["LogLoss_CV"]), 6),
                "Brier_CV": round(float(df_beta.iloc[0]["Brier_CV"]), 6),
            },
        ]
    )

    final_params = fit_poisson_elo_params_given_beta(hist, best_beta, half_life_days)

    return {
        "params": final_params,
        "best_beta": best_beta,
        "history": hist,
        "comparison": comparison,
        "beta_cv": df_beta,
        "folds": df_folds,
        "cv_raw": df_raw,
        "usable_validation_years": usable_years,
    }


def favorite_calibration_report(
    df_eval: pd.DataFrame,
    params: Mapping[str, float],
) -> pd.DataFrame:
    """Compara a probabilidade média do favorito com sua taxa real de vitória."""
    probs, _, _ = _history_predict_1x2(df_eval, params)
    y = _history_actual_1x2(df_eval)
    favorite_side = np.where(probs[:, 0] >= probs[:, 2], 0, 2)
    favorite_prob = np.maximum(probs[:, 0], probs[:, 2])
    favorite_won = (y == favorite_side).astype(int)

    tmp = pd.DataFrame({"Fav_Prob_Modelo": favorite_prob, "Fav_Venceu": favorite_won})
    bins = [0.00, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    labels = ["<40%", "40-45%", "45-50%", "50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-90%", "90%+"]
    tmp["Faixa"] = pd.cut(tmp["Fav_Prob_Modelo"], bins=bins, labels=labels, right=False, include_lowest=True)

    out = (
        tmp.dropna(subset=["Faixa"])
        .groupby("Faixa", observed=True)
        .agg(
            Jogos=("Fav_Venceu", "size"),
            Prob_Modelo_Media=("Fav_Prob_Modelo", "mean"),
            Taxa_Real_Vitoria=("Fav_Venceu", "mean"),
        )
        .reset_index()
    )
    out["Prob_Modelo_Media_%"] = (out["Prob_Modelo_Media"] * 100).round(2)
    out["Taxa_Real_Vitoria_Favorito_%"] = (out["Taxa_Real_Vitoria"] * 100).round(2)
    out["Erro_pp"] = (out["Prob_Modelo_Media_%"] - out["Taxa_Real_Vitoria_Favorito_%"]).round(2)
    return out[["Faixa", "Jogos", "Prob_Modelo_Media_%", "Taxa_Real_Vitoria_Favorito_%", "Erro_pp"]]


def elo_upset_report(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Taxas empíricas de vitória do maior Elo, empate e zebra por gap absoluto de Elo."""
    d = df_eval["Elo_Diff"].to_numpy(dtype=float)
    gh = df_eval["Home_Score"].to_numpy(dtype=float)
    ga = df_eval["Away_Score"].to_numpy(dtype=float)
    higher_home = d >= 0
    higher_won = np.where(higher_home, gh > ga, ga > gh)
    draw = gh == ga
    upset = ~(higher_won | draw)

    tmp = pd.DataFrame(
        {
            "Abs_Elo_Gap": np.abs(d),
            "Mais_Forte_Venceu": higher_won.astype(int),
            "Empate": draw.astype(int),
            "Zebra": upset.astype(int),
        }
    )
    bins = [0, 50, 100, 150, 200, 250, 300, 400, np.inf]
    labels = ["0-49", "50-99", "100-149", "150-199", "200-249", "250-299", "300-399", "400+"]
    tmp["Faixa_Elo"] = pd.cut(tmp["Abs_Elo_Gap"], bins=bins, labels=labels, right=False)
    out = (
        tmp.groupby("Faixa_Elo", observed=True)
        .agg(
            Jogos=("Mais_Forte_Venceu", "size"),
            Mais_Forte_Venceu=("Mais_Forte_Venceu", "mean"),
            Empate=("Empate", "mean"),
            Zebra=("Zebra", "mean"),
        )
        .reset_index()
    )
    out["Mais_Forte_Venceu_%"] = (out["Mais_Forte_Venceu"] * 100).round(2)
    out["Empate_%"] = (out["Empate"] * 100).round(2)
    out["Zebra_%"] = (out["Zebra"] * 100).round(2)
    return out[["Faixa_Elo", "Jogos", "Mais_Forte_Venceu_%", "Empate_%", "Zebra_%"]]
