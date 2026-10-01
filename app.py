"""
World Cup Predictor — Streamlit app.

Run:
    streamlit run app.py

If Streamlit is not installed the script falls back to a simple CLI demo.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def _run_streamlit() -> None:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px

    from src.predict import MatchPredictor
    from src.simulate_tournament import simulate_tournament, format_results_table
    from src.data_loader import load_teams, team_list
    import config

    st.set_page_config(
        page_title="World Cup Predictor",
        page_icon="⚽",
        layout="wide",
    )

    # ── Load assets ────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Loading model...")
    def get_predictor():
        p = MatchPredictor()
        p.load()
        return p

    @st.cache_data(show_spinner="Loading teams...")
    def get_teams():
        return load_teams()

    @st.cache_data(show_spinner="Loading tournament bracket...")
    def get_bracket():
        if config.TOURNAMENT_FILE.exists():
            with open(config.TOURNAMENT_FILE) as f:
                return json.load(f)
        return None

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.title("⚽ World Cup Predictor")
    page = st.sidebar.radio(
        "Navigation",
        ["Match Predictor", "Tournament Simulator", "Model Insights"],
        index=0,
    )

    if not config.MATCHES_FILE.exists():
        st.error(
            "**No match data found.**  `data/results.csv` is missing from this "
            "deployment — make sure it was committed to the repo."
        )
        st.stop()

    if not config.MODEL_FILE.exists():
        # First run on a fresh deploy: model artifacts aren't committed to git
        # (best_model.pkl alone is 100MB+), so train them once here instead.
        with st.spinner("First-time setup: training model on historical + 2026 World Cup data (~1-2 min)..."):
            from src.train import train as train_model
            train_model(use_shap=False)

    if not config.POISSON_MODEL_FILE.exists():
        with st.spinner("First-time setup: fitting scoreline model..."):
            from fit_poisson_model import fit_and_save
            fit_and_save()

    predictor = get_predictor()
    teams_df = get_teams()
    all_teams = team_list(teams_df)

    # ── Page: Match Predictor ───────────────────────────────────────────────
    if page == "Match Predictor":
        st.title("Match Outcome Predictor")
        st.caption("Predict home-win / draw / away-win probabilities for any match-up.")

        col1, col2 = st.columns(2)
        with col1:
            home = st.selectbox("Home team", all_teams, index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
        with col2:
            away = st.selectbox("Away team", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 1)

        col3, col4, col5 = st.columns(3)
        with col3:
            stage = st.selectbox("Stage", [
                "Group Stage", "Round of 32", "Round of 16",
                "Quarterfinal", "Semifinal", "Final",
            ])
        with col4:
            tournament = st.selectbox("Tournament", [
                "FIFA World Cup", "UEFA Euro", "Copa America",
                "Africa Cup of Nations", "Friendly",
            ])
        with col5:
            neutral = st.checkbox("Neutral venue", value=True)

        if home == away:
            st.error("Home and away teams must be different.")
        else:
            probs = predictor.predict(home, away, tournament=tournament, stage=stage, neutral=neutral)
            is_knockout = stage != "Group Stage"

            st.markdown("---")
            if is_knockout:
                # In knockout rounds draws go to extra time / penalties — split 50/50
                p_home = probs["home_win"] + probs["draw"] / 2
                p_away = probs["away_win"] + probs["draw"] / 2
                c1, c2 = st.columns(2)
                c1.metric(f"🏠 {home} advances", f"{p_home:.1%}")
                c2.metric(f"✈️ {away} advances", f"{p_away:.1%}")
                st.caption(
                    f"Draw probability ({probs['draw']:.1%}) split 50/50 — "
                    "knockout matches go to extra time / penalties if level after 90 min."
                )
                bar_x = [home, away]
                bar_y = [p_home, p_away]
                bar_colors = ["#1f77b4", "#ff7f0e"]
                bar_text = [f"{p_home:.1%}", f"{p_away:.1%}"]
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric(f"🏠 {home}", f"{probs['home_win']:.1%}")
                c2.metric("🤝 Draw", f"{probs['draw']:.1%}")
                c3.metric(f"✈️ {away}", f"{probs['away_win']:.1%}")
                bar_x = [home, "Draw", away]
                bar_y = [probs["home_win"], probs["draw"], probs["away_win"]]
                bar_colors = ["#1f77b4", "#aec7e8", "#ff7f0e"]
                bar_text = [f"{v:.1%}" for v in bar_y]

            # Probability bar chart
            fig = go.Figure(go.Bar(
                x=bar_x,
                y=bar_y,
                marker_color=bar_colors,
                text=bar_text,
                textposition="outside",
            ))
            fig.update_layout(
                title=f"{home} vs {away} — {stage}",
                yaxis_title="Probability",
                yaxis_tickformat=".0%",
                yaxis_range=[0, 1],
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig, width="stretch")

            # ── Predicted scoreline (Poisson model) ─────────────────────────
            try:
                if not config.POISSON_MODEL_FILE.exists():
                    raise FileNotFoundError(
                        "Scoreline model not trained yet. Run:\n"
                        "    python fit_poisson_model.py"
                    )

                @st.cache_resource(show_spinner="Loading scoreline model...")
                def get_poisson():
                    import pickle
                    with open(config.POISSON_MODEL_FILE, "rb") as f:
                        return pickle.load(f)

                pm = get_poisson()
                result = pm.predict(home, away, neutral=neutral)
                top = result.top_scorelines(8)
                top_scoreline, top_prob = top[0]

                st.markdown("---")
                st.subheader("Predicted Score")
                c1, c2 = st.columns(2)
                c1.metric(f"{home} vs {away}", top_scoreline)
                c2.metric("Confidence", f"{top_prob:.1%}")
                st.caption(
                    f"'Confidence' is the probability the exact scoreline {top_scoreline} "
                    f"occurs, out of every possible scoreline — not the same as the win/draw/loss "
                    f"probability above. Expected goals: {home} {result.expected_home_goals:.2f} — "
                    f"{away} {result.expected_away_goals:.2f}"
                )

                with st.expander("Show more likely scorelines"):
                    sl_df = pd.DataFrame(top, columns=["Scoreline", "Probability"])
                    sl_df["Probability"] = (sl_df["Probability"] * 100).round(2).astype(str) + "%"
                    st.dataframe(sl_df, width="stretch", hide_index=True)
            except Exception as exc:
                st.warning(f"Scoreline prediction unavailable: {exc}")

    # ── Page: Tournament Simulator ──────────────────────────────────────────
    elif page == "Tournament Simulator":
        st.title("2026 World Cup Simulator")
        st.caption("Monte Carlo simulation of the full tournament.  10 000 runs by default.")

        bracket = get_bracket()
        if bracket is None:
            st.error(
                "Tournament bracket not found.  Run:\n```\npython generate_mock_data.py\n```"
            )
            st.stop()

        n_sims = st.slider("Number of simulations", 1000, 50_000, 10_000, step=1000)

        if st.button("▶  Run Simulation", type="primary"):
            with st.spinner(f"Simulating {n_sims:,} tournaments..."):
                df = simulate_tournament(n_simulations=n_sims, predictor=predictor)

            st.success(f"Done — {n_sims:,} simulations completed.")

            # Full results table
            display = format_results_table(df)
            st.subheader("Team Advancement Probabilities")
            st.dataframe(display, width="stretch", hide_index=True)

            # Champion probability chart (top 16)
            top16 = df.head(16)
            fig2 = px.bar(
                top16,
                x="team",
                y="Champion",
                title="Championship Probability — Top 16 Teams",
                labels={"Champion": "P(Champion)", "team": ""},
                color="Champion",
                color_continuous_scale="Blues",
            )
            fig2.update_layout(
                yaxis_tickformat=".1%",
                coloraxis_showscale=False,
                xaxis_tickangle=-30,
                height=450,
            )
            st.plotly_chart(fig2, width="stretch")

            # Semifinal chart
            fig3 = px.bar(
                df.head(20),
                x="team",
                y="Semifinal",
                title="Semifinal Probability — Top 20 Teams",
                labels={"Semifinal": "P(Semifinal)", "team": ""},
                color="Semifinal",
                color_continuous_scale="Oranges",
            )
            fig3.update_layout(
                yaxis_tickformat=".1%",
                coloraxis_showscale=False,
                xaxis_tickangle=-30,
                height=420,
            )
            st.plotly_chart(fig3, width="stretch")

        elif config.SIM_RESULTS_FILE.exists():
            st.info("Showing previous simulation results.  Press Run to re-simulate.")
            df = pd.read_csv(config.SIM_RESULTS_FILE)
            display = format_results_table(df)
            st.dataframe(display, width="stretch", hide_index=True)

    # ── Page: Model Insights ────────────────────────────────────────────────
    elif page == "Model Insights":
        st.title("Model Performance & Insights")

        if not config.METRICS_FILE.exists():
            st.warning("Run  `python -m src.train`  to generate metrics.")
            st.stop()

        with open(config.METRICS_FILE) as f:
            metrics = json.load(f)

        st.subheader(f"Best model: **{metrics['best_model']}**")

        # Validation comparison
        st.markdown("#### Validation Metrics (model selection)")
        val_df = pd.DataFrame(metrics["validation"]).set_index("model")
        st.dataframe(val_df.style.highlight_min(axis=0, subset=["log_loss", "brier"])
                                 .highlight_max(axis=0, subset=["accuracy"]),
                     width="stretch")

        # Test metrics
        st.markdown("#### Test Metrics")
        test_df = pd.DataFrame(metrics["test"]).set_index("model")
        st.dataframe(test_df, width="stretch")

        # Feature importance
        if metrics.get("feature_importance"):
            fi_df = pd.DataFrame(metrics["feature_importance"])
            st.markdown("#### Feature Importance")
            fig_fi = px.bar(
                fi_df.head(20),
                x="importance", y="feature",
                orientation="h",
                title="Top Feature Importances",
                labels={"importance": "Importance", "feature": ""},
            )
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            st.plotly_chart(fig_fi, width="stretch")


# ---------------------------------------------------------------------------
# CLI fallback
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    from src.predict import MatchPredictor
    from src.simulate_tournament import simulate_tournament, format_results_table

    print("\n=== World Cup Predictor (CLI mode) ===")
    print("Tip: install streamlit + plotly for the full web UI\n")

    if not config.MODEL_FILE.exists():
        print("No model found.  Run:\n  python generate_mock_data.py\n  python -m src.train")
        return

    import config

    predictor = MatchPredictor()
    predictor.load()

    # Sample prediction
    pairs = [
        ("Brazil", "Argentina", "Final"),
        ("France", "England", "Semifinal"),
        ("Germany", "Spain", "Quarterfinal"),
    ]
    print("=== Sample Match Predictions ===")
    for home, away, stage in pairs:
        try:
            p = predictor.predict(home, away, stage=stage)
            print(f"  {home:15s} vs {away:15s}  [{stage}]")
            print(f"    {home}: {p['home_win']:.1%}  Draw: {p['draw']:.1%}  {away}: {p['away_win']:.1%}")
        except Exception as e:
            print(f"  {home} vs {away} — error: {e}")

    print("\n=== Tournament Simulation (1 000 runs) ===")
    try:
        df = simulate_tournament(n_simulations=1_000, predictor=predictor)
        display = format_results_table(df)
        print(display.head(16).to_string(index=False))
    except Exception as e:
        print(f"Simulation error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

try:
    import streamlit as _st_check  # noqa: F401
    # When invoked via `streamlit run app.py` the whole module is executed at
    # import time, so we call _run_streamlit() unconditionally here.
    _run_streamlit()
except ImportError:
    if __name__ == "__main__":
        print("Streamlit not found.  Falling back to CLI demo.")
        _run_cli()
