# World Cup Predictor

A production-quality Python system for predicting international football match outcomes
and simulating the 2026 FIFA World Cup via Monte Carlo methods.

---

## Folder structure

```
world_cup_predictor/
├── config.py                  ← all paths, hyperparameters, tournament config
├── generate_mock_data.py      ← creates data/ from synthetic data (run first)
├── app.py                     ← Streamlit web app (or CLI fallback)
├── requirements.txt
├── data/
│   ├── matches.csv            ← historical match results  ← REPLACE WITH REAL DATA
│   ├── teams.csv              ← team metadata (rank, confederation)
│   └── tournament_2026.json  ← 2026 WC group draw & bracket
├── models/                    ← saved model artifacts (created by train)
├── results/                   ← simulation outputs
└── src/
    ├── data_loader.py         ← load & validate CSVs
    ├── feature_engineering.py ← chronological, leakage-free feature builder
    ├── train.py               ← trains LR / RF / XGBoost, calibrates, saves best
    ├── predict.py             ← single-match probability predictor
    ├── simulate_tournament.py ← Monte Carlo tournament simulation
    └── poisson_model.py       ← optional Dixon-Coles Poisson goal model
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate mock data  (or drop in real CSV — see "Real data" below)
python generate_mock_data.py

# 3. Train models
python -m src.train

# 4a. Launch the Streamlit web app
streamlit run app.py

# 4b.  OR use the CLI modules directly
python -m src.predict --home "Brazil" --away "Argentina" --stage "Final"
python -m src.simulate_tournament --n 10000
```

---

## Real data (recommended)

1. Download *Results.csv* from the Kaggle dataset:
   <https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017>

2. Rename it to `data/matches.csv`.  The required columns map directly:
   `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`

3. Optionally supply `data/teams.csv` with current FIFA rankings — the model will
   recompute Elo dynamically from match history regardless.

4. Retrain: `python -m src.train`

---

## Modeling approach

### Feature engineering (`src/feature_engineering.py`)

All features are computed in strict chronological order — no future information
leaks into a match's feature row.

| Feature group        | Features                                                    |
|----------------------|-------------------------------------------------------------|
| **Elo**              | `home_elo`, `away_elo`, `elo_diff`                          |
| **FIFA rank**        | `home_rank`, `away_rank`, `rank_diff`                       |
| **Recent form**      | Points per game, avg GF/GA/GD (last 10 matches)             |
| **Momentum**         | Win rate over last 20 matches                               |
| **Match context**    | `is_neutral`, `stage_enc` (1–6), `tournament_weight` (1–4) |
| **Experience**       | Number of historical matches in dataset                     |

Elo ratings use a K-factor that scales with tournament importance (60 for World Cup,
20 for friendlies) and a goal-difference multiplier.  Home teams receive a +100 Elo
bonus on non-neutral venues when building the `elo_diff` feature.

### Models (`src/train.py`)

Three models are compared on a **time-based** validation set (no random shuffling):

| Model               | Notes                                                      |
|---------------------|------------------------------------------------------------|
| Logistic Regression | Interpretable baseline; multinomial softmax                |
| Random Forest       | Captures non-linear interactions                           |
| XGBoost / LightGBM  | Typically best on tabular data; gradient boosted trees     |

All models are wrapped in `CalibratedClassifierCV` (isotonic regression, 5-fold)
so output probabilities are well-calibrated.

**Selection criterion:** validation log-loss (minimised).

### Evaluation metrics

- **Log-loss** — primary; measures calibration and sharpness together
- **Accuracy** — fraction of correctly predicted outcomes
- **Brier score** — mean squared error of probability estimates (multiclass average)

### Poisson goal model (`src/poisson_model.py`)

Optional Dixon-Coles-inspired model that estimates per-team attack and defense
parameters by maximum likelihood, then returns a full scoreline probability matrix.
Useful for betting-odds comparison or scoreline-level analysis.

### Tournament simulation (`src/simulate_tournament.py`)

- Reads `data/tournament_2026.json` (edit to reflect the actual draw).
- Simulates 10 000 full tournaments (configurable via `--n`).
- Group stage: round-robin; goals sampled from conditioned Poisson.
- Best-3rd-place selection: top 8 of 12 third-place finishers by points → GD → GF.
- Knockout rounds: draws resolved by 50/50 extra-time/penalties.
- Output: per-team probability of reaching each stage plus winning the tournament.

---

## Example commands

```bash
# Predict a single match
python -m src.predict --home "France" --away "Germany" --stage "Semifinal"

# Simulate the tournament (5 000 runs)
python -m src.simulate_tournament --n 5000

# Retrain with SHAP explanations
python -m src.train   # add --no-shap to skip SHAP

# Launch Streamlit UI
streamlit run app.py
```

---

## Extending the project

- **Better data** — Add bookmaker odds as optional features (column `odds_home`, `odds_draw`, `odds_away`) — the feature matrix already has a natural slot for them.
- **Player ratings** — Add a squad-average FIFA rating or market-value proxy as `home_player_strength` / `away_player_strength`.
- **Hyperparameter tuning** — Wrap the training loop with `optuna` or `sklearn.model_selection.RandomizedSearchCV`.
- **Live odds** — Fetch via the football-data.org API and compare against model probabilities.

---

## Design tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Chronological split (not random) | Reflects real deployment; prevents leakage but reduces effective training size |
| CalibratedClassifierCV isotonic | Better calibration than Platt scaling for large datasets; requires ≥ ~200 samples per fold |
| Elo over FIFA points | Elo updates more responsively; FIFA rankings lag by months |
| Single best model saved | Simpler deployment; ensemble of all models would improve calibration slightly |
| Poisson model separate | Keeps the main pipeline fast; scoreline estimates add ~10 s fitting time |
