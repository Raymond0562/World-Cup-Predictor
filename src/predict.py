"""
Load a trained model and predict match-outcome probabilities.

Usage (CLI):
    python -m src.predict --home "Brazil" --away "Argentina" --stage "Final"
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from config import HOME_WIN, DRAW, AWAY_WIN, OUTCOME_LABELS
from src.feature_engineering import FeatureEngineer, FEATURE_COLS

logger = logging.getLogger(__name__)


class MatchPredictor:
    """Wraps the saved model + scaler + feature engineer for one-shot predictions."""

    def __init__(self) -> None:
        self.model: Any = None
        self.scaler: Any = None
        self.engineer: FeatureEngineer | None = None
        self.teams: pd.DataFrame | None = None
        self._loaded = False

    def load(
        self,
        model_path: Path | None = None,
        scaler_path: Path | None = None,
        engineer_path: Path | None = None,
        teams_path: Path | None = None,
    ) -> None:
        model_path = model_path or config.MODEL_FILE
        scaler_path = scaler_path or config.SCALER_FILE
        engineer_path = engineer_path or config.MODEL_DIR / "engineer.pkl"
        teams_path = teams_path or config.TEAMS_FILE

        for p in [model_path, scaler_path, engineer_path]:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Missing artifact: {p}\nRun  python -m src.train  first."
                )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(engineer_path, "rb") as f:
            self.engineer = pickle.load(f)

        if Path(teams_path).exists():
            self.teams = pd.read_csv(teams_path)

        self._loaded = True
        logger.info("Model artifacts loaded.")

    def predict(
        self,
        home_team: str,
        away_team: str,
        tournament: str = "Friendly",
        stage: str = "Group Stage",
        neutral: bool = True,
    ) -> dict[str, float]:
        """
        Return a dict with keys 'home_win', 'draw', 'away_win' as probabilities.
        """
        if not self._loaded:
            self.load()

        raw = self.engineer.transform_single(
            home_team, away_team,
            tournament=tournament, stage=stage, neutral=neutral,
            teams=self.teams,
        )

        X = np.array([[raw[f] for f in FEATURE_COLS]])
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]

        return {
            "home_win": float(probs[HOME_WIN]),
            "draw": float(probs[DRAW]),
            "away_win": float(probs[AWAY_WIN]),
        }

    def predict_batch(
        self,
        team_pairs: list[tuple[str, str]],
        tournament: str = "FIFA World Cup",
        stage: str = "Group Stage",
        neutral: bool = True,
    ) -> list[dict[str, float]]:
        """
        Predict probabilities for a list of (home, away) pairs in one model call.
        Much faster than calling predict() in a loop for many pairs.
        """
        if not self._loaded:
            self.load()

        rows = []
        for home, away in team_pairs:
            raw = self.engineer.transform_single(
                home, away, tournament=tournament, stage=stage,
                neutral=neutral, teams=self.teams,
            )
            rows.append([raw[f] for f in FEATURE_COLS])

        X = np.array(rows, dtype=float)
        X_scaled = self.scaler.transform(X)
        probs_matrix = self.model.predict_proba(X_scaled)

        return [
            {
                "home_win": float(p[HOME_WIN]),
                "draw": float(p[DRAW]),
                "away_win": float(p[AWAY_WIN]),
            }
            for p in probs_matrix
        ]

    def predict_winner(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        stage: str = "Group Stage",
    ) -> str:
        """Return predicted most-likely outcome label."""
        probs = self.predict(home_team, away_team, tournament, stage, neutral)
        idx = max(probs, key=probs.get)
        return {
            "home_win": f"{home_team} win",
            "draw": "Draw",
            "away_win": f"{away_team} win",
        }[idx]


# Singleton for reuse across modules
_predictor: MatchPredictor | None = None


def get_predictor() -> MatchPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MatchPredictor()
        _predictor.load()
    return _predictor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Predict match-outcome probabilities.")
    p.add_argument("--home", required=True, help="Home team name")
    p.add_argument("--away", required=True, help="Away team name")
    p.add_argument("--tournament", default="FIFA World Cup")
    p.add_argument("--stage", default="Group Stage")
    p.add_argument("--neutral", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args()

    predictor = MatchPredictor()
    predictor.load()

    probs = predictor.predict(
        args.home, args.away,
        tournament=args.tournament, stage=args.stage, neutral=args.neutral,
    )

    print(f"\n  {args.home}  vs  {args.away}")
    print(f"  Stage: {args.stage}  |  Neutral: {args.neutral}")
    print(f"  {'-' * 40}")
    print(f"  {args.home} win : {probs['home_win']:.1%}")
    print(f"  Draw           : {probs['draw']:.1%}")
    print(f"  {args.away} win : {probs['away_win']:.1%}")
