"""Load, validate, and clean match and team data."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

REQUIRED_MATCH_COLS = {
    "date", "home_team", "away_team", "home_score", "away_score", "tournament",
}
REQUIRED_TEAM_COLS = {"team", "confederation", "fifa_rank"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_matches(path: Path | None = None) -> pd.DataFrame:
    """Load and clean the historical match CSV."""
    path = path or config.MATCHES_FILE
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Matches file not found: {path}\n"
            "Run  python generate_mock_data.py  to create mock data."
        )
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    df = _validate_matches(df)
    df = _clean_matches(df)
    logger.info("Loaded %d matches from %s", len(df), path)
    return df


def load_teams(path: Path | None = None) -> pd.DataFrame:
    """Load and clean the team metadata CSV."""
    path = path or config.TEAMS_FILE
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Teams file not found: {path}\n"
            "Run  python generate_mock_data.py  to create mock data."
        )
    df = pd.read_csv(path)
    df = _validate_teams(df)
    df = _clean_teams(df)
    logger.info("Loaded %d teams from %s", len(df), path)
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: load matches and teams together."""
    return load_matches(), load_teams()


def team_list(teams_df: pd.DataFrame | None = None) -> list[str]:
    """Return sorted list of all teams seen in the match history."""
    matches_path = config.MATCHES_FILE
    if matches_path.exists():
        df = pd.read_csv(matches_path, usecols=["home_team", "away_team"], encoding="utf-8")
        all_teams = pd.concat([df["home_team"], df["away_team"]]).dropna().unique()
        return sorted(all_teams.tolist())
    # Fallback to teams metadata file
    if teams_df is not None:
        return sorted(teams_df["team"].tolist())
    return []


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_matches(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_MATCH_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Matches CSV is missing columns: {missing}")
    return df


def _validate_teams(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_TEAM_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Teams CSV is missing columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    # Drop rows where either score is missing (can't determine outcome)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Standardise text columns
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()
    df["tournament"] = df["tournament"].str.strip().fillna("Friendly")

    # Ensure neutral flag exists
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].map(
        lambda v: bool(v) if not isinstance(v, str) else v.lower() == "true"
    )

    if "stage" not in df.columns:
        df["stage"] = "Group Stage"
    df["stage"] = df["stage"].fillna("Group Stage")

    df = df.sort_values("date").reset_index(drop=True)

    n_before = len(df)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    dropped = n_before - len(df)
    if dropped:
        logger.warning("Dropped %d duplicate match rows.", dropped)

    return df


def _clean_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["team"] = df["team"].str.strip()
    df["confederation"] = df["confederation"].str.strip().fillna("Unknown")
    df["fifa_rank"] = pd.to_numeric(df["fifa_rank"], errors="coerce").fillna(100).astype(int)
    if "elo_rating" not in df.columns:
        df["elo_rating"] = 1500  # will be recomputed dynamically
    df["elo_rating"] = pd.to_numeric(df["elo_rating"], errors="coerce").fillna(1500)
    return df.drop_duplicates(subset=["team"])
