"""
Optional Poisson / Dixon-Coles style goal model.

Estimates team-specific attack and defense parameters from historical goal data,
then predicts full scoreline distributions and derives win/draw/loss probabilities.

Usage:
    from src.poisson_model import PoissonModel
    pm = PoissonModel()
    pm.fit(matches_df)
    result = pm.predict("Brazil", "Argentina")
    # result.home_win, result.draw, result.away_win, result.scorelines
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

MAX_GOALS = 10          # upper bound for scoreline grid
HOME_ADV_INIT = 0.25    # initial estimate for home advantage (log scale)
DEFAULT_ATTACK = 0.0    # log scale
DEFAULT_DEFENSE = 0.0


@dataclass
class PoissonResult:
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float
    # Scoreline dict: (h, a) -> probability
    scorelines: dict = field(default_factory=dict)

    def top_scorelines(self, n: int = 5) -> list[tuple[str, float]]:
        sorted_sl = sorted(self.scorelines.items(), key=lambda x: x[1], reverse=True)
        return [(f"{h}-{a}", round(p, 4)) for (h, a), p in sorted_sl[:n]]


class PoissonModel:
    """
    Simplified Dixon-Coles-style Poisson goals model.

    Fits attack (α) and defense (β) parameters per team, plus a global
    home-advantage parameter (h).  Parameters are estimated by maximum
    likelihood on historical match scores.
    """

    def __init__(self) -> None:
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_advantage: float = HOME_ADV_INIT
        self.mu: float = 0.0       # global average goals (log scale)
        self._teams: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        """
        Fit attack/defense parameters from a DataFrame with columns:
        home_team, away_team, home_score, away_score.
        """
        matches = matches.dropna(subset=["home_score", "away_score"]).copy()
        self._teams = sorted(
            set(matches["home_team"]) | set(matches["away_team"])
        )
        n_teams = len(self._teams)
        team_idx = {t: i for i, t in enumerate(self._teams)}

        # Vectorize match data up front — the objective is called hundreds of
        # times by the optimizer, so avoiding a Python-level iterrows() loop
        # per call is the difference between seconds and many minutes.
        home_i = matches["home_team"].map(team_idx).to_numpy()
        away_i = matches["away_team"].map(team_idx).to_numpy()
        home_goals = matches["home_score"].to_numpy(dtype=float)
        away_goals = matches["away_score"].to_numpy(dtype=float)
        log_fact_home = gammaln(home_goals + 1.0)
        log_fact_away = gammaln(away_goals + 1.0)

        # Initial parameters: [home_adv, mu, attack×n, defense×n]
        x0 = np.zeros(2 + 2 * n_teams)
        x0[0] = HOME_ADV_INIT   # home advantage
        x0[1] = np.log(matches[["home_score", "away_score"]].values.mean() + 1e-6)

        def neg_log_likelihood(params: np.ndarray) -> float:
            h_adv = params[0]
            mu_  = params[1]
            atk  = params[2:2 + n_teams]
            def_ = params[2 + n_teams:]

            log_lam_h = mu_ + atk[home_i] - def_[away_i] + h_adv
            log_lam_a = mu_ + atk[away_i] - def_[home_i]
            lam_h = np.exp(log_lam_h)
            lam_a = np.exp(log_lam_a)

            # log Poisson pmf, vectorized: k*log(lam) - lam - log(k!)
            ll_h = home_goals * log_lam_h - lam_h - log_fact_home
            ll_a = away_goals * log_lam_a - lam_a - log_fact_away
            return -float(ll_h.sum() + ll_a.sum())

        def neg_log_likelihood_grad(params: np.ndarray) -> np.ndarray:
            """Analytical gradient — avoids L-BFGS-B falling back to finite
            differences, which would need ~2*n_teams extra function calls
            per iteration (674 params -> minutes instead of seconds)."""
            h_adv = params[0]
            mu_  = params[1]
            atk  = params[2:2 + n_teams]
            def_ = params[2 + n_teams:]

            log_lam_h = mu_ + atk[home_i] - def_[away_i] + h_adv
            log_lam_a = mu_ + atk[away_i] - def_[home_i]
            lam_h = np.exp(log_lam_h)
            lam_a = np.exp(log_lam_a)
            res_h = home_goals - lam_h
            res_a = away_goals - lam_a

            d_h_adv = -res_h.sum()
            d_mu = -(res_h.sum() + res_a.sum())
            d_atk = -(
                np.bincount(home_i, weights=res_h, minlength=n_teams) +
                np.bincount(away_i, weights=res_a, minlength=n_teams)
            )
            d_def = (
                np.bincount(away_i, weights=res_h, minlength=n_teams) +
                np.bincount(home_i, weights=res_a, minlength=n_teams)
            )
            return np.concatenate(([d_h_adv, d_mu], d_atk, d_def))

        logger.info("Fitting Poisson model on %d matches (%d teams) …", len(matches), n_teams)
        result = minimize(
            neg_log_likelihood,
            x0,
            jac=neg_log_likelihood_grad,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-10},
        )
        if not result.success:
            logger.warning("Poisson optimisation did not fully converge: %s", result.message)

        params = result.x
        self.home_advantage = float(params[0])
        self.mu = float(params[1])
        for i, team in enumerate(self._teams):
            self.attack[team] = float(params[2 + i])
            self.defense[team] = float(params[2 + n_teams + i])

        self._fitted = True
        logger.info("Poisson model fitted.  Home advantage: %.3f", self.home_advantage)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> PoissonResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        h_adv = 0.0 if neutral else self.home_advantage
        atk_h = self.attack.get(home_team, DEFAULT_ATTACK)
        def_h = self.defense.get(home_team, DEFAULT_DEFENSE)
        atk_a = self.attack.get(away_team, DEFAULT_ATTACK)
        def_a = self.defense.get(away_team, DEFAULT_DEFENSE)

        lam_h = np.exp(self.mu + atk_h - def_a + h_adv)
        lam_a = np.exp(self.mu + atk_a - def_h)

        # Build scoreline probability grid
        scores: dict = {}
        p_home_win = 0.0
        p_draw = 0.0
        p_away_win = 0.0

        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                p = float(poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a))
                scores[(h, a)] = p
                if h > a:
                    p_home_win += p
                elif h == a:
                    p_draw += p
                else:
                    p_away_win += p

        # Normalise residual mass
        total = p_home_win + p_draw + p_away_win
        if total > 0:
            p_home_win /= total
            p_draw /= total
            p_away_win /= total

        return PoissonResult(
            home_team=home_team,
            away_team=away_team,
            home_win=round(p_home_win, 4),
            draw=round(p_draw, 4),
            away_win=round(p_away_win, 4),
            expected_home_goals=round(lam_h, 2),
            expected_away_goals=round(lam_a, 2),
            scorelines=scores,
        )

    def team_strengths(self) -> pd.DataFrame:
        rows = [
            {"team": t, "attack": self.attack.get(t, 0.0),
             "defense": self.defense.get(t, 0.0)}
            for t in self._teams
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("attack", ascending=False)
            .reset_index(drop=True)
        )


