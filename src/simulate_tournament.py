"""
Monte Carlo tournament simulator.

Reads the tournament bracket from data/tournament_2026.json, runs N simulations,
and outputs each team's probability of reaching each knockout stage.

Key optimization: all team-pair match probabilities are precomputed once before
the simulation loop, making 10 000 iterations fast even with a slow model.

Usage (CLI):
    python -m src.simulate_tournament [--n 10000] [--tournament data/tournament_2026.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from config import N_SIMULATIONS, WC_2026_FORMAT
from src.predict import MatchPredictor, get_predictor

logger = logging.getLogger(__name__)

# (team_a, team_b) -> (p_home_win, p_draw, p_away_win)
ProbTable = dict[tuple[str, str], tuple[float, float, float]]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TeamStats:
    __slots__ = ("name", "points", "gf", "ga", "group")

    def __init__(self, name: str, group: str) -> None:
        self.name = name
        self.points = 0
        self.gf = 0
        self.ga = 0
        self.group = group

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def reset(self) -> None:
        self.points = 0
        self.gf = 0
        self.ga = 0


# ---------------------------------------------------------------------------
# Probability precomputation
# ---------------------------------------------------------------------------

def _build_prob_table(
    teams: list[str],
    predictor: MatchPredictor,
    stage: str = "Group Stage",
    tournament: str = "FIFA World Cup",
) -> ProbTable:
    """
    Compute win/draw/loss probabilities for every ordered team pair in ONE
    batched model call, so the 10K simulation loop only does random sampling.
    """
    pairs = [(a, b) for a in teams for b in teams if a != b]
    results = predictor.predict_batch(pairs, tournament=tournament, stage=stage, neutral=True)
    return {
        pair: (r["home_win"], r["draw"], r["away_win"])
        for pair, r in zip(pairs, results)
    }


# ---------------------------------------------------------------------------
# Scoreline sampling
# ---------------------------------------------------------------------------

def _sample_scoreline(p_home: float, p_draw: float, p_away: float,
                      mu: float = 1.25) -> tuple[int, int]:
    """Sample a concrete scoreline consistent with the sampled outcome."""
    outcome = random.choices(("H", "D", "A"), weights=(p_home, p_draw, p_away))[0]
    if outcome == "H":
        while True:
            h, a = np.random.poisson(mu * 1.3), np.random.poisson(mu * 0.9)
            if h > a:
                return h, a
    elif outcome == "D":
        g = np.random.poisson(mu)
        return g, g
    else:
        while True:
            h, a = np.random.poisson(mu * 0.9), np.random.poisson(mu * 1.3)
            if a > h:
                return h, a


def _knockout_winner(team_a: str, team_b: str, ko_probs: ProbTable) -> str:
    """Return winner of a single-leg knockout match (no draws)."""
    p_h, p_d, p_a = ko_probs.get((team_a, team_b), (1/3, 1/3, 1/3))
    # Redistribute draw probability 50/50 (extra time / penalties)
    p_a_wins = p_h + p_d / 2
    return team_a if random.random() < p_a_wins else team_b


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------

def _simulate_group(
    group_name: str,
    teams: list[str],
    group_probs: ProbTable,
) -> list[TeamStats]:
    """Simulate one round-robin group and return teams ranked by standing."""
    stats = {t: TeamStats(t, group_name) for t in teams}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home, away = teams[i], teams[j]
            p_h, p_d, p_a = group_probs.get((home, away), (1/3, 1/3, 1/3))
            h_goals, a_goals = _sample_scoreline(p_h, p_d, p_a)

            stats[home].gf += h_goals
            stats[home].ga += a_goals
            stats[away].gf += a_goals
            stats[away].ga += h_goals

            if h_goals > a_goals:
                stats[home].points += 3
            elif h_goals == a_goals:
                stats[home].points += 1
                stats[away].points += 1
            else:
                stats[away].points += 3

    return sorted(
        stats.values(),
        key=lambda s: (s.points, s.gd, s.gf, random.random()),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Full tournament simulation (one run)
# ---------------------------------------------------------------------------

def _simulate_once(
    groups: dict[str, list[str]],
    knockout_rounds: list[str],
    group_probs: ProbTable,
    ko_probs: ProbTable,
    best_third_advance: int,
) -> dict[str, str]:
    """
    Run one full tournament.
    Returns a dict: team -> deepest stage reached (or 'Champion').
    """
    stage_reached: dict[str, str] = {}
    all_group_results: list[list[TeamStats]] = []

    for g_name, g_teams in groups.items():
        ranked = _simulate_group(g_name, g_teams, group_probs)
        all_group_results.append(ranked)

        for ts in ranked[2:]:
            stage_reached[ts.name] = "Group Stage"

    # Auto-advance: top 2 from each group
    alive = (
        [r[0].name for r in all_group_results] +
        [r[1].name for r in all_group_results]
    )

    # Best third-place teams
    third_place = [r[2] for r in all_group_results]
    best_thirds = sorted(
        third_place,
        key=lambda ts: (ts.points, ts.gd, ts.gf, random.random()),
        reverse=True,
    )[:best_third_advance]

    alive += [ts.name for ts in best_thirds]

    for ts in third_place:
        if ts.name not in alive:
            stage_reached[ts.name] = "Group Stage"

    random.shuffle(alive)

    for stage in knockout_rounds:
        next_round = []
        for k in range(0, len(alive) - 1, 2):
            winner = _knockout_winner(alive[k], alive[k + 1], ko_probs)
            loser = alive[k] if winner == alive[k + 1] else alive[k + 1]
            stage_reached[loser] = stage
            next_round.append(winner)
        alive = next_round

    if alive:
        stage_reached[alive[0]] = "Champion"

    return stage_reached


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "Group Stage",
    "Round of 32",
    "Round of 16",
    "Quarterfinal",
    "Semifinal",
    "Final",
    "Champion",
]


def _aggregate(results: list[dict[str, str]], all_teams: list[str]) -> pd.DataFrame:
    """Build a probability table from simulation results."""
    stage_idx = {s: i for i, s in enumerate(STAGE_ORDER)}

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sim in results:
        for team, deepest in sim.items():
            depth = stage_idx.get(deepest, 0)
            for stage in STAGE_ORDER[:depth + 1]:
                counts[team][stage] += 1

    n = len(results)
    rows = []
    for team in all_teams:
        row = {"team": team}
        for stage in STAGE_ORDER[1:]:
            row[stage] = counts[team].get(stage, 0) / n
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("Champion", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def simulate_tournament(
    tournament_file: Path | None = None,
    n_simulations: int = N_SIMULATIONS,
    predictor: MatchPredictor | None = None,
) -> pd.DataFrame:
    """Run N Monte Carlo simulations and return a team-probability table."""
    tournament_file = tournament_file or config.TOURNAMENT_FILE
    if not Path(tournament_file).exists():
        raise FileNotFoundError(
            f"Tournament file not found: {tournament_file}\n"
            "Run  python generate_mock_data.py  to create it."
        )

    with open(tournament_file) as f:
        bracket = json.load(f)

    groups: dict[str, list[str]] = bracket["groups"]
    knockout_rounds: list[str] = bracket.get(
        "knockout_rounds", WC_2026_FORMAT["knockout_rounds"]
    )
    best_third_advance: int = bracket.get(
        "best_third_advance", WC_2026_FORMAT["best_third_advance"]
    )
    all_teams = [t for g in groups.values() for t in g]

    logger.info(
        "Simulating %d-team tournament (%d groups) x %d runs ...",
        len(all_teams), len(groups), n_simulations,
    )

    if predictor is None:
        predictor = get_predictor()

    # Precompute all match probabilities once (major speed-up for large N)
    logger.info("Precomputing match probabilities (%d pairs) ...", len(all_teams) * (len(all_teams) - 1))
    group_probs = _build_prob_table(all_teams, predictor, stage="Group Stage")
    ko_probs = _build_prob_table(all_teams, predictor, stage="Semifinal")

    np.random.seed(config.RANDOM_STATE)
    random.seed(config.RANDOM_STATE)

    results: list[dict[str, str]] = []
    log_every = max(1, n_simulations // 10)
    for i in range(n_simulations):
        if i % log_every == 0:
            logger.info("  simulation %d / %d ...", i, n_simulations)
        results.append(
            _simulate_once(groups, knockout_rounds, group_probs, ko_probs, best_third_advance)
        )

    df = _aggregate(results, all_teams)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(config.SIM_RESULTS_FILE, index=False)
    logger.info("Saved results to %s", config.SIM_RESULTS_FILE)
    return df


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format probabilities as percentages for display."""
    out = df.copy()
    for col in out.columns:
        if col != "team":
            out[col] = (out[col] * 100).round(1).astype(str) + "%"
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo World Cup simulation.")
    p.add_argument("--n", type=int, default=N_SIMULATIONS, help="Number of simulations")
    p.add_argument("--tournament", default=None, help="Path to tournament JSON")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()

    df = simulate_tournament(
        tournament_file=args.tournament,
        n_simulations=args.n,
    )

    display = format_results_table(df)
    print(f"\n{'='*95}")
    print(f"{'2026 World Cup Simulation':^95}")
    print(f"{'='*95}")
    print(display.to_string(index=False))
