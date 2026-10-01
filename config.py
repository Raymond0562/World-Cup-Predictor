"""Central configuration — edit this file to tune paths, hyperparameters, and tournament structure."""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
# REAL DATA: replace these with a real dataset for better predictions.
# A good free source: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
# Required columns for matches.csv:
#   date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
# Required columns for teams.csv:
#   team, confederation, fifa_rank, elo_rating (optional — will be computed dynamically)
MATCHES_FILE = DATA_DIR / "results.csv"   # Kaggle dataset filename
TEAMS_FILE = DATA_DIR / "teams.csv"
TOURNAMENT_FILE = DATA_DIR / "tournament_2026.json"

MODEL_FILE = MODEL_DIR / "best_model.pkl"
FEATURE_NAMES_FILE = MODEL_DIR / "feature_names.pkl"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
POISSON_MODEL_FILE = MODEL_DIR / "poisson_model.pkl"
METRICS_FILE = RESULTS_DIR / "model_metrics.json"
SIM_RESULTS_FILE = RESULTS_DIR / "simulation_results.csv"

# ---------------------------------------------------------------------------
# Elo settings
# ---------------------------------------------------------------------------
ELO_INITIAL = 1500
ELO_K_BASE = 20
ELO_K_TOURNAMENT = {
    "FIFA World Cup": 60,
    "Confederations Cup": 40,
    "Copa América": 40,
    "UEFA Euro": 40,
    "African Cup of Nations": 35,
    "AFC Asian Cup": 35,
    "Gold Cup": 35,
    "CONCACAF Nations League": 30,
    "FIFA World Cup qualification": 30,
    "UEFA Euro qualification": 25,
    "UEFA Nations League": 25,
    "African Cup of Nations qualification": 25,
    "AFC Asian Cup qualification": 25,
    "Friendly": 20,
}
HOME_ADVANTAGE_ELO = 100  # virtual Elo bonus for home team on non-neutral venue

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FORM_WINDOW = 10       # matches used for recent form features
MOMENTUM_WINDOW = 20   # matches used for longer-term momentum

STAGE_ENCODING = {
    "Group Stage": 1,
    "Round of 32": 2,
    "Round of 16": 3,
    "Quarterfinal": 4,
    "Semifinal": 5,
    "Third-place play-off": 5,
    "Final": 6,
}

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 4,
    "UEFA Euro": 3,
    "Copa América": 3,
    "African Cup of Nations": 2,
    "AFC Asian Cup": 2,
    "Gold Cup": 2,
    "CONCACAF Nations League": 2,
    "Confederations Cup": 3,
    "UEFA Nations League": 2,
    "FIFA World Cup qualification": 2,
    "UEFA Euro qualification": 2,
    "African Cup of Nations qualification": 1,
    "AFC Asian Cup qualification": 1,
    "Friendly": 1,
}

# Default values for teams with no history yet
DEFAULT_FORM_POINTS = 1.0   # points per game
DEFAULT_WIN_RATE = 0.33
DEFAULT_AVG_GF = 1.3
DEFAULT_AVG_GA = 1.3

# ---------------------------------------------------------------------------
# Time-based splits (chronological — never shuffle)
# ---------------------------------------------------------------------------
# Used for model comparison / hyperparameter selection only:
TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE = "2022-12-31"
# After model selection, the winner is retrained on all completed data found
# in the dataset — automatically picks up the latest results when you drop in
# a fresh results.csv from Kaggle.
def _latest_completed_date() -> str:
    try:
        df = pd.read_csv(MATCHES_FILE, usecols=["date", "home_score"], encoding="utf-8")
        latest = pd.to_datetime(df.loc[df["home_score"].notna(), "date"]).max()
        return latest.strftime("%Y-%m-%d")
    except Exception:
        return "2026-06-16"

FINAL_TRAIN_END_DATE: str = _latest_completed_date()

# ---------------------------------------------------------------------------
# Outcome encoding
# ---------------------------------------------------------------------------
HOME_WIN = 0
DRAW = 1
AWAY_WIN = 2
OUTCOME_LABELS = {HOME_WIN: "Home Win", DRAW: "Draw", AWAY_WIN: "Away Win"}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

LR_PARAMS = {
    "C": 0.5,
    "max_iter": 2000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 15,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 15,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

# ---------------------------------------------------------------------------
# Sample weighting (recency + tournament importance)
# ---------------------------------------------------------------------------
# Exponential time-decay: newest match gets exp(WEIGHT_DECAY_RATE) × more
# weight than the oldest match in the dataset (~150 years of data).
# decay=3 → newest ≈ 20× oldest; decay=5 → ≈ 150×.
WEIGHT_DECAY_RATE: float = 3.0
# Additional multiplier on top of the recency weight for 2026 WC matches,
# ensuring the model strongly prioritises the current tournament.
WC_2026_WEIGHT_MULTIPLIER: float = 5.0

# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------
N_SIMULATIONS = 10_000

# 2026 World Cup format
WC_2026_FORMAT = {
    "n_groups": 12,
    "group_size": 4,
    "auto_advance": 2,       # top 2 per group advance automatically
    "best_third_advance": 8, # best 3rd-place teams (total 32 advance)
    "knockout_rounds": ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"],
}
