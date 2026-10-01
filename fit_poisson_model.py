"""
Fit the Poisson scoreline model and persist it to disk.

Run this after every  python -m src.train  so the scoreline predictions stay
in sync with the latest match results.

Usage:
    python fit_poisson_model.py
"""

import logging
import pickle

import config
from src.data_loader import load_matches
from src.poisson_model import PoissonModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def fit_and_save() -> PoissonModel:
    matches = load_matches()
    pm = PoissonModel()
    pm.fit(matches)

    config.MODEL_DIR.mkdir(exist_ok=True)
    with open(config.POISSON_MODEL_FILE, "wb") as f:
        pickle.dump(pm, f)
    logger.info("Saved Poisson model to %s", config.POISSON_MODEL_FILE)
    return pm


if __name__ == "__main__":
    fit_and_save()
