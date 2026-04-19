"""
app.py — Player Churn Prediction: Streamlit application entry point.

Run with:
    streamlit run app.py

Architecture:
  app.py          → orchestration only (page config, routing)
  src/ui.py       → all Streamlit visual components
  src/inference.py → model loading and prediction logic
  src/preprocessing.py → data cleaning and feature encoding
"""

import pandas as pd
import streamlit as st

from src.inference import load_evaluation_metrics, load_models, predict
from src.preprocessing import preprocess
from src import ui

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Player Churn Prediction",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; }
    .stMetric { background: #ffffff; border-radius: 8px; padding: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached resource loading ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models ...")
def _load_models():
    """Load models once per server session; cache across browser tabs."""
    return load_models()


@st.cache_data(show_spinner=False)
def _load_metrics():
    """Load evaluation metrics once; cached until file changes."""
    return load_evaluation_metrics()


# ── App layout ────────────────────────────────────────────────────────────────

ui.render_header()
ui.render_pipeline_overview()

uploaded_file, model_name = ui.render_sidebar()

# Load models — abort early on failure so the rest of the app stays clean
try:
    log_model, dt_model, scaler, label_encoders, feature_names = _load_models()
except Exception as exc:
    st.error(
        f"**Model loading failed:** {exc}\n\n"
        "Run `python -m src.train` to retrain and save models."
    )
    st.stop()

# ── Main prediction flow ──────────────────────────────────────────────────────

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # ── Step 1: Preprocessing ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("1. Data Processing")
        X_scaled, X_encoded = preprocess(df, label_encoders, scaler)
        st.success(
            f"Processed **{len(df):,} players** — "
            "missing values imputed, categorical features encoded."
        )

        # ── Step 2: Prediction ───────────────────────────────────────────────
        st.subheader("2. Predictions")
        model = log_model if model_name == "Logistic Regression" else dt_model
        X_input = X_scaled if model_name == "Logistic Regression" else X_encoded

        results = predict(model, X_input)

        player_ids = (
            df["PlayerID"].values
            if "PlayerID" in df.columns
            else range(1, len(df) + 1)
        )
        results_df = pd.DataFrame(
            {
                "PlayerID": player_ids,
                "Prediction": results["prediction"],
                "ChurnProbability": results["churn_prob"],
                "ChurnRisk": results["risk"],
            }
        )

        ui.render_metrics_cards(results_df)

        # ── Step 3: Results table ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("3. Player Risk Table")
        ui.render_results_table(results_df)

        # ── Step 4: Visualisations ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("4. Visualisations")
        col_left, col_right = st.columns(2)

        with col_left:
            ui.render_risk_distribution(results_df)

        with col_right:
            if model_name == "Decision Tree":
                imp_df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                ui.render_feature_importance(imp_df)
            else:
                st.info(
                    "Feature importance is available when the **Decision Tree** "
                    "model is selected."
                )

        # ── Step 5: Evaluation metrics ───────────────────────────────────────
        st.markdown("---")
        st.subheader("5. Model Evaluation Metrics")
        eval_metrics = _load_metrics()
        if eval_metrics:
            model_key = (
                "logistic_regression"
                if model_name == "Logistic Regression"
                else "decision_tree"
            )
            ui.render_evaluation_metrics(eval_metrics, model_key)
        else:
            st.info(
                "Evaluation metrics not yet generated. "
                "Run `python -m src.train` to produce them."
            )

        # ── Step 6: Download ─────────────────────────────────────────────────
        st.markdown("---")
        ui.render_download_button(results_df)

    except ValueError as exc:
        st.error(f"**Data validation error:** {exc}")
        ui.render_schema_help()
    except Exception as exc:
        st.error(f"**Unexpected error:** {exc}")
        st.exception(exc)

else:
    st.info("Upload a player-data CSV in the sidebar to get started.")
    ui.render_schema_help()
