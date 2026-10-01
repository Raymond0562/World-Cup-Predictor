"""
Generate synthetic international football match data for development and testing.

REAL DATA NOTE: Replace data/matches.csv and data/teams.csv with a real dataset
to get meaningful predictions.  A good free source is the Kaggle dataset:
  https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Run:  python generate_mock_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ---------------------------------------------------------------------------
# Team definitions  (name -> confederation, approximate FIFA rank, strength 0-1)
# ---------------------------------------------------------------------------
TEAMS = {
    "Argentina":      ("CONMEBOL", 1,  0.940),
    "France":         ("UEFA",     2,  0.930),
    "Brazil":         ("CONMEBOL", 5,  0.925),
    "England":        ("UEFA",     4,  0.895),
    "Belgium":        ("UEFA",     3,  0.890),
    "Netherlands":    ("UEFA",     6,  0.885),
    "Portugal":       ("UEFA",     7,  0.880),
    "Spain":          ("UEFA",     8,  0.875),
    "Croatia":        ("UEFA",    10,  0.855),
    "Italy":          ("UEFA",     9,  0.855),
    "Germany":        ("UEFA",    12,  0.850),
    "Uruguay":        ("CONMEBOL",11,  0.840),
    "Morocco":        ("CAF",     13,  0.820),
    "USA":            ("CONCACAF",13,  0.800),
    "Mexico":         ("CONCACAF",15,  0.795),
    "Colombia":       ("CONMEBOL",14,  0.800),
    "Switzerland":    ("UEFA",    19,  0.805),
    "Denmark":        ("UEFA",    21,  0.800),
    "Japan":          ("AFC",     18,  0.795),
    "Senegal":        ("CAF",     20,  0.790),
    "Austria":        ("UEFA",    25,  0.780),
    "Ukraine":        ("UEFA",    22,  0.778),
    "South Korea":    ("AFC",     23,  0.775),
    "Sweden":         ("UEFA",    24,  0.775),
    "Poland":         ("UEFA",    26,  0.765),
    "Serbia":         ("UEFA",    33,  0.755),
    "Australia":      ("AFC",     25,  0.745),
    "Wales":          ("UEFA",    28,  0.745),
    "Turkey":         ("UEFA",    40,  0.730),
    "Iran":           ("AFC",     22,  0.730),
    "Ecuador":        ("CONMEBOL",44,  0.730),
    "Hungary":        ("UEFA",    29,  0.725),
    "Czech Republic": ("UEFA",    37,  0.725),
    "Scotland":       ("UEFA",    38,  0.720),
    "Algeria":        ("CAF",     36,  0.715),
    "Egypt":          ("CAF",     35,  0.715),
    "Nigeria":        ("CAF",     40,  0.710),
    "Tunisia":        ("CAF",     30,  0.705),
    "Chile":          ("CONMEBOL",55,  0.705),
    "Peru":           ("CONMEBOL",62,  0.695),
    "Canada":         ("CONCACAF",50,  0.695),
    "Cameroon":       ("CAF",     51,  0.685),
    "Ghana":          ("CAF",     60,  0.675),
    "Costa Rica":     ("CONCACAF",47,  0.675),
    "Saudi Arabia":   ("AFC",     56,  0.660),
    "Qatar":          ("AFC",     48,  0.655),
    "Ivory Coast":    ("CAF",     55,  0.650),
    "Panama":         ("CONCACAF",79,  0.625),
}

TEAM_LIST = list(TEAMS.keys())

TOURNAMENTS = [
    ("FIFA World Cup",             4),
    ("FIFA World Cup qualification",2),
    ("UEFA Euro",                  3),
    ("Copa America",               3),
    ("Africa Cup of Nations",      2),
    ("AFC Asian Cup",              2),
    ("CONCACAF Gold Cup",          2),
    ("UEFA Nations League",        2),
    ("Friendly",                   3),  # higher weight so friendlies are common
]

WC_TEAMS_PER_YEAR = {
    2010: ["Brazil","Argentina","Germany","England","France","Spain","Netherlands","Portugal",
           "Italy","Chile","Uruguay","Mexico","USA","Serbia","South Korea","Japan",
           "Ghana","Ivory Coast","Cameroon","Nigeria","Algeria","Egypt","Denmark","Sweden",
           "Switzerland","Austria","Croatia","Poland","Australia","Iran","Saudi Arabia","Qatar"],
    2014: ["Brazil","Germany","Argentina","Colombia","France","Belgium","Netherlands","Portugal",
           "Spain","Uruguay","Switzerland","USA","Mexico","Chile","Italy","Costa Rica",
           "Japan","South Korea","Australia","Iran","Algeria","Ghana","Nigeria","Cameroon",
           "Croatia","Sweden","Denmark","Poland","Ecuador","Peru","Serbia","Turkey"],
    2018: ["France","Croatia","Belgium","England","Brazil","Uruguay","Russia","Sweden",
           "Germany","Argentina","Spain","Portugal","Denmark","Switzerland","Colombia","Mexico",
           "Japan","South Korea","Australia","Iran","Senegal","Nigeria","Tunisia","Egypt",
           "Poland","Serbia","Czech Republic","Scotland","Austria","Hungary","Wales","Canada"],
    2022: ["Argentina","France","Morocco","Croatia","Brazil","England","Netherlands","Portugal",
           "Spain","Germany","Belgium","Uruguay","USA","Senegal","Switzerland","Japan",
           "South Korea","Australia","Poland","Denmark","Ecuador","Serbia","Italy","Ghana",
           "Tunisia","Cameroon","Costa Rica","Saudi Arabia","Qatar","Canada","Iran","Wales"],
}


def _simulate_goals(strength_h: float, strength_a: float, neutral: bool) -> tuple[int, int]:
    """Sample a scoreline using Poisson with team-strength-based lambdas."""
    home_bonus = 0.0 if neutral else 0.25
    ratio = strength_h / max(strength_a, 0.01)
    base = 1.25
    lam_h = np.clip(base * ratio + home_bonus, 0.4, 4.0)
    lam_a = np.clip(base / ratio, 0.4, 4.0)
    return int(np.random.poisson(lam_h)), int(np.random.poisson(lam_a))


def _make_match(home: str, away: str, date: pd.Timestamp,
                tournament: str, stage: str, neutral: bool) -> dict:
    sh = TEAMS[home][2]
    sa = TEAMS[away][2]
    hg, ag = _simulate_goals(sh, sa, neutral)
    country = "Neutral" if neutral else home
    return {
        "date": date.strftime("%Y-%m-%d"),
        "home_team": home,
        "away_team": away,
        "home_score": hg,
        "away_score": ag,
        "tournament": tournament,
        "stage": stage,
        "city": "Mock City",
        "country": country,
        "neutral": neutral,
    }


def _generate_wc_matches(year: int, teams: list[str]) -> list[dict]:
    """Generate a full World Cup (8 groups ->knockout) for a given year."""
    rows = []
    month_start = pd.Timestamp(f"{year}-06-01")

    # 8 groups of 4
    groups = [teams[i*4:(i+1)*4] for i in range(8)]
    for gi, group in enumerate(groups):
        day_offset = gi * 3
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                date = month_start + pd.Timedelta(days=day_offset)
                rows.append(_make_match(group[i], group[j], date,
                                        "FIFA World Cup", "Group Stage", True))
                day_offset += 1

    # Knockout rounds
    ko_rounds = [("Round of 16", 8), ("Quarterfinal", 4),
                 ("Semifinal", 2), ("Final", 1)]
    ko_day = 20
    pool = teams[:]
    np.random.shuffle(pool)
    for stage, n in ko_rounds:
        remaining = pool[:n * 2]
        for k in range(0, len(remaining) - 1, 2):
            date = month_start + pd.Timedelta(days=ko_day)
            rows.append(_make_match(remaining[k], remaining[k + 1], date,
                                    "FIFA World Cup", stage, True))
            ko_day += 2
        pool = pool[:n]  # rough approximation; real bracket not modelled here
    return rows


def _generate_regular_matches() -> list[dict]:
    """Generate friendlies and qualifying matches 2010–2024."""
    rows = []
    rng_teams = TEAM_LIST.copy()
    date = pd.Timestamp("2010-01-15")
    end = pd.Timestamp("2024-06-01")

    t_names = [t[0] for t in TOURNAMENTS]
    t_weights = np.array([t[1] for t in TOURNAMENTS], dtype=float)
    t_weights /= t_weights.sum()

    while date < end:
        n_matches = np.random.randint(6, 18)
        for _ in range(n_matches):
            tour = np.random.choice(t_names, p=t_weights)
            home, away = np.random.choice(rng_teams, 2, replace=False)
            neutral = np.random.random() < 0.25
            rows.append(_make_match(home, away, date, tour, "Group Stage", neutral))
        date += pd.Timedelta(days=np.random.randint(4, 15))
    return rows


def generate_matches() -> pd.DataFrame:
    rows = _generate_regular_matches()
    for year, teams in WC_TEAMS_PER_YEAR.items():
        rows.extend(_generate_wc_matches(year, teams))
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def generate_teams() -> pd.DataFrame:
    rows = []
    for name, (conf, rank, strength) in TEAMS.items():
        elo = int(ELO_BASE + strength * ELO_RANGE)
        rows.append({"team": name, "confederation": conf,
                     "fifa_rank": rank, "elo_rating": elo})
    return pd.DataFrame(rows).sort_values("fifa_rank").reset_index(drop=True)


ELO_BASE = 1200
ELO_RANGE = 700


def generate_tournament_2026() -> dict:
    """Return a projected 48-team 2026 World Cup group draw."""
    # Seeded by approximate 2024/2025 FIFA rankings
    groups = {
        "A": ["USA", "Colombia", "Ivory Coast", "Serbia"],
        "B": ["Mexico", "Poland", "Ecuador", "Saudi Arabia"],
        "C": ["Canada", "Portugal", "Tunisia", "Czech Republic"],
        "D": ["Argentina", "Chile", "Algeria", "South Korea"],
        "E": ["Spain", "Japan", "Cameroon", "Hungary"],
        "F": ["France", "Uruguay", "Iran", "Scotland"],
        "G": ["Brazil", "Switzerland", "Ghana", "Costa Rica"],
        "H": ["England", "Australia", "Nigeria", "Panama"],
        "I": ["Germany", "Morocco", "Peru", "Qatar"],
        "J": ["Netherlands", "Senegal", "Turkey", "Wales"],
        "K": ["Belgium", "Croatia", "Egypt", "Canada"],
        "L": ["Italy", "Denmark", "Sweden", "Austria"],
    }
    # Fix duplicate Canada appearance
    groups["K"] = ["Belgium", "Croatia", "Egypt", "Ivory Coast"]
    groups["A"] = ["USA", "Colombia", "Ghana", "Serbia"]

    return {
        "year": 2026,
        "format": "48-team",
        "n_groups": 12,
        "group_size": 4,
        "groups": groups,
        "knockout_rounds": ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"],
        "note": "Replace with actual 2026 FIFA World Cup draw results.",
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-matches", action="store_true",
                        help="Regenerate mock matches.csv even if real data already exists.")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    # The pipeline expects results.csv (matches file); fall back to matches.csv for mock data
    real_path = out_dir / "results.csv"
    mock_path = out_dir / "matches.csv"
    matches_exist = real_path.exists() or mock_path.exists()

    if not matches_exist or args.force_matches:
        print("Generating mock match data ...")
        matches = generate_matches()
        matches.to_csv(mock_path, index=False)
        print(f"  {len(matches):,} matches  -> {mock_path}")
        print(f"  Date range: {matches['date'].min()}  to  {matches['date'].max()}")
        print("  NOTE: Using synthetic data.  For real predictions drop in data/results.csv")
    else:
        existing = real_path if real_path.exists() else mock_path
        print(f"  Match data already exists ({existing.name}) - skipping mock generation.")
        print("  (Use --force-matches to overwrite with synthetic data.)")

    print("Generating teams metadata ...")
    teams = generate_teams()
    teams_path = out_dir / "teams.csv"
    teams.to_csv(teams_path, index=False)
    print(f"  {len(teams)} teams  -> {teams_path}")

    print("Generating 2026 tournament structure ...")
    bracket = generate_tournament_2026()
    bracket_path = out_dir / "tournament_2026.json"
    with open(bracket_path, "w") as f:
        json.dump(bracket, f, indent=2)
    print(f"  Saved  -> {bracket_path}")

    print("\nDone.  Next steps:")
    print("  1. python -m src.train")
    print("  2. streamlit run app.py")


if __name__ == "__main__":
    main()
