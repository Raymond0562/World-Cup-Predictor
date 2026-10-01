"""
Build match features with strict chronological ordering to prevent data leakage.

All features for a match are computed exclusively from matches that occurred
BEFORE that match's date.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from config import (
    ELO_INITIAL, ELO_K_BASE, ELO_K_TOURNAMENT, HOME_ADVANTAGE_ELO,
    FORM_WINDOW, MOMENTUM_WINDOW,
    STAGE_ENCODING, TOURNAMENT_WEIGHT,
    DEFAULT_FORM_POINTS, DEFAULT_WIN_RATE, DEFAULT_AVG_GF, DEFAULT_AVG_GA,
    HOME_WIN, DRAW, AWAY_WIN,
)

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_rank", "away_rank", "rank_diff",
    "home_form_pts", "away_form_pts", "form_pts_diff",
    "home_form_gf", "away_form_gf",
    "home_form_ga", "away_form_ga",
    "home_form_gd", "away_form_gd",
    "home_momentum", "away_momentum", "momentum_diff",
    "is_neutral",
    "stage_enc",
    "tournament_weight",
    "home_n_matches", "away_n_matches",
]


class FeatureEngineer:
    """
    Processes matches in chronological order, maintaining running state for each
    team so that features computed for match T only use data from matches < T.
    """

    def __init__(
        self,
        form_window: int = FORM_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
    ) -> None:
        self.form_window = form_window
        self.momentum_window = momentum_window

        # Running state — reset between calls to fit_transform / fit
        self._elo: dict[str, float] = {}
        self._history: dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        matches: pd.DataFrame,
        teams: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build features for every match in `matches`.
        Resets internal state so this is safe to call from scratch.
        """
        self._reset(teams)
        return self._transform_all(matches, teams)

    def transform_single(
        self,
        home_team: str,
        away_team: str,
        tournament: str = "Friendly",
        stage: str = "Group Stage",
        neutral: bool = True,
        teams: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Build features for a future / hypothetical match using current state.
        Does NOT update state (the match hasn't happened yet).
        """
        home_meta = self._team_meta(home_team, teams)
        away_meta = self._team_meta(away_team, teams)
        return self._build_row(
            home_team, away_team, home_meta, away_meta,
            tournament, stage, neutral,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _reset(self, teams: pd.DataFrame) -> None:
        # Pre-populate with known teams; unknown teams are added on-the-fly
        self._elo = {row["team"]: ELO_INITIAL for _, row in teams.iterrows()}
        self._history: dict[str, deque] = {row["team"]: deque(maxlen=self.momentum_window * 3)
                                            for _, row in teams.iterrows()}

    def _team_meta(self, team: str, teams: pd.DataFrame | None) -> dict:
        if teams is not None:
            row = teams[teams["team"] == team]
            if not row.empty:
                return row.iloc[0].to_dict()
        return {"team": team, "fifa_rank": 100, "confederation": "Unknown"}

    def _transform_all(
        self,
        matches: pd.DataFrame,
        teams: pd.DataFrame,
    ) -> pd.DataFrame:
        team_meta = {row["team"]: row.to_dict() for _, row in teams.iterrows()}
        _default_meta = {"fifa_rank": 100, "confederation": "Unknown"}

        records: list[dict] = []

        for idx, match in matches.sort_values("date").iterrows():
            home = match["home_team"]
            away = match["away_team"]

            # Use metadata if available, otherwise fall back to defaults.
            # Elo is computed dynamically for ALL teams regardless.
            home_meta = team_meta.get(home, _default_meta)
            away_meta = team_meta.get(away, _default_meta)

            # Ensure Elo/history entries exist for first-seen teams
            if home not in self._elo:
                self._elo[home] = ELO_INITIAL
                self._history[home] = deque(maxlen=self.momentum_window * 3)
            if away not in self._elo:
                self._elo[away] = ELO_INITIAL
                self._history[away] = deque(maxlen=self.momentum_window * 3)

            row = self._build_row(
                home, away, home_meta, away_meta,
                match.get("tournament", "Friendly"),
                match.get("stage", "Group Stage"),
                bool(match.get("neutral", False)),
            )
            row["match_id"] = idx
            row["date"] = match["date"]
            row["home_team"] = home
            row["away_team"] = away
            row["tournament"] = match.get("tournament", "Friendly")

            hs, as_ = int(match["home_score"]), int(match["away_score"])
            if hs > as_:
                row["outcome"] = HOME_WIN
            elif hs == as_:
                row["outcome"] = DRAW
            else:
                row["outcome"] = AWAY_WIN

            records.append(row)

            # Update running state AFTER building row (no leakage)
            home_elo_pre = self._elo.get(home, ELO_INITIAL)
            away_elo_pre = self._elo.get(away, ELO_INITIAL)
            self._update_elo(home, away, hs, as_, match.get("tournament", "Friendly"))
            self._history.setdefault(home, deque(maxlen=self.momentum_window * 3)).append(
                {"gf": hs, "ga": as_, "opp_elo": away_elo_pre}
            )
            self._history.setdefault(away, deque(maxlen=self.momentum_window * 3)).append(
                {"gf": as_, "ga": hs, "opp_elo": home_elo_pre}
            )

        df = pd.DataFrame(records)
        logger.info("Built %d feature rows from %d matches", len(df), len(matches))
        return df

    def _build_row(
        self,
        home: str, away: str,
        home_meta: dict, away_meta: dict,
        tournament: str, stage: str, neutral: bool,
    ) -> dict[str, Any]:
        home_elo = self._elo.get(home, ELO_INITIAL)
        away_elo = self._elo.get(away, ELO_INITIAL)
        # Apply home advantage to Elo diff when not neutral
        effective_home_elo = home_elo + (0 if neutral else HOME_ADVANTAGE_ELO)

        home_form = self._compute_form(list(self._history.get(home, [])), self.form_window)
        away_form = self._compute_form(list(self._history.get(away, [])), self.form_window)
        home_mom = self._compute_form(list(self._history.get(home, [])), self.momentum_window)
        away_mom = self._compute_form(list(self._history.get(away, [])), self.momentum_window)

        h_rank = home_meta.get("fifa_rank", 100)
        a_rank = away_meta.get("fifa_rank", 100)

        return {
            # Elo
            "home_elo": effective_home_elo,
            "away_elo": away_elo,
            "elo_diff": effective_home_elo - away_elo,
            # FIFA rank (lower is better → rank_diff positive means home is higher ranked)
            "home_rank": h_rank,
            "away_rank": a_rank,
            "rank_diff": a_rank - h_rank,
            # Form
            "home_form_pts": home_form["pts_per_game"],
            "away_form_pts": away_form["pts_per_game"],
            "form_pts_diff": home_form["pts_per_game"] - away_form["pts_per_game"],
            "home_form_gf": home_form["avg_gf"],
            "away_form_gf": away_form["avg_gf"],
            "home_form_ga": home_form["avg_ga"],
            "away_form_ga": away_form["avg_ga"],
            "home_form_gd": home_form["avg_gf"] - home_form["avg_ga"],
            "away_form_gd": away_form["avg_gf"] - away_form["avg_ga"],
            # Momentum (longer window)
            "home_momentum": home_mom["win_rate"],
            "away_momentum": away_mom["win_rate"],
            "momentum_diff": home_mom["win_rate"] - away_mom["win_rate"],
            # Match context
            "is_neutral": int(neutral),
            "stage_enc": STAGE_ENCODING.get(stage, 1),
            "tournament_weight": TOURNAMENT_WEIGHT.get(tournament, 1),
            # Experience (proxy for squad depth / tournament consistency)
            "home_n_matches": len(self._history.get(home, [])),
            "away_n_matches": len(self._history.get(away, [])),
        }

    @staticmethod
    def _compute_form(history: list[dict], window: int) -> dict:
        """Compute aggregate stats from the last `window` matches."""
        recent = history[-window:] if len(history) >= window else history
        n = len(recent)

        if n == 0:
            return {
                "pts_per_game": DEFAULT_FORM_POINTS,
                "win_rate": DEFAULT_WIN_RATE,
                "avg_gf": DEFAULT_AVG_GF,
                "avg_ga": DEFAULT_AVG_GA,
            }

        wins = sum(1 for m in recent if m["gf"] > m["ga"])
        draws = sum(1 for m in recent if m["gf"] == m["ga"])
        total_gf = sum(m["gf"] for m in recent)
        total_ga = sum(m["ga"] for m in recent)

        return {
            "pts_per_game": (3 * wins + draws) / n,
            "win_rate": wins / n,
            "avg_gf": total_gf / n,
            "avg_ga": total_ga / n,
        }

    def _update_elo(
        self,
        home: str, away: str,
        home_goals: int, away_goals: int,
        tournament: str,
    ) -> None:
        """Standard Elo update with goal-difference multiplier."""
        ra = self._elo.get(home, ELO_INITIAL)
        rb = self._elo.get(away, ELO_INITIAL)

        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        if home_goals > away_goals:
            sa, sb = 1.0, 0.0
        elif home_goals == away_goals:
            sa, sb = 0.5, 0.5
        else:
            sa, sb = 0.0, 1.0

        k = ELO_K_TOURNAMENT.get(tournament, ELO_K_BASE)
        gd = abs(home_goals - away_goals)
        k_mult = 1.0 + 0.5 * np.log1p(max(gd - 1, 0))

        self._elo[home] = ra + k * k_mult * (sa - ea)
        self._elo[away] = rb + k * k_mult * (sb - eb)

    @property
    def current_elo(self) -> dict[str, float]:
        """Read current Elo ratings (useful after fit_transform)."""
        return dict(self._elo)

    @property
    def current_history(self) -> dict[str, list]:
        return {t: list(h) for t, h in self._history.items()}


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def build_features(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    form_window: int = FORM_WINDOW,
    momentum_window: int = MOMENTUM_WINDOW,
) -> tuple[pd.DataFrame, FeatureEngineer]:
    """
    Build a features DataFrame from raw match data and return the fitted engineer
    (which carries current Elo ratings and team history for future predictions).
    """
    eng = FeatureEngineer(form_window=form_window, momentum_window=momentum_window)
    features = eng.fit_transform(matches, teams)
    return features, eng
