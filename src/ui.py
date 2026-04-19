"""
ui.py — Reusable Streamlit UI components for Player Churn Prediction.

Each function is a self-contained UI block that can be imported and called
from app.py. Side effects are limited to st.* calls — no I/O or model
logic lives here.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# ── Colour palette ────────────────────────────────────────────────────────────

_RISK_COLOURS = {
    "Low": ("#d4edda", "#155724"),      # green
    "Medium": ("#fff3cd", "#856404"),   # amber
    "High": ("#f8d7da", "#721c24"),     # red
}

_BAR_PALETTE = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}


# ── Page-level components ────────────────────────────────────────────────────

def render_header() -> None:
    """Render the app title and subtitle."""
    st.title("🎮 Player Churn Prediction System")
    st.markdown(
        "Upload a CSV of player-behaviour data to identify at-risk players "
        "using ML models trained on 40 000+ records."
    )


def render_pipeline_overview() -> None:
    """Render a collapsible 4-step pipeline explanation."""
    with st.expander("How it works", expanded=False):
        cols = st.columns(4)
        steps = [
            ("1. Upload", "Provide a player-behaviour CSV file"),
            ("2. Preprocess", "Impute missing values, encode categoricals"),
            ("3. Predict", "Score each player with the selected ML model"),
            ("4. Assess", "Categorise into Low / Medium / High churn risk"),
        ]
        for col, (title, body) in zip(cols, steps):
            col.markdown(f"**{title}**")
            col.caption(body)


def render_sidebar():
    """Render sidebar controls and return user selections.

    Returns
    -------
    uploaded_file : UploadedFile or None
    model_name    : str — 'Logistic Regression' | 'Decision Tree'
    """
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader(
            "Upload Player Data (CSV)", type=["csv"], help="Must include all 11 feature columns."
        )
        model_name = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Decision Tree"],
            help="Logistic Regression uses scaled features; Decision Tree uses raw-encoded features.",
        )
        st.markdown("---")
        st.caption("Models trained on `online_gaming_behavior_dataset.csv`")
    return uploaded_file, model_name


# ── Results components ───────────────────────────────────────────────────────

def render_metrics_cards(results_df: pd.DataFrame) -> None:
    """Show KPI metric cards (total, high risk, avg probability, low risk).

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: ChurnRisk, ChurnProbability.
    """
    total = len(results_df)
    high_risk = (results_df["ChurnRisk"] == "High").sum()
    low_risk = (results_df["ChurnRisk"] == "Low").sum()
    avg_prob = results_df["ChurnProbability"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Players", f"{total:,}")
    c2.metric(
        "High Risk",
        f"{high_risk:,}",
        delta=f"{high_risk / total:.1%} of total",
        delta_color="inverse",
    )
    c3.metric("Avg Churn Probability", f"{avg_prob:.2%}")
    c4.metric("Low Risk", f"{low_risk:,}", delta=f"{low_risk / total:.1%} of total")


def render_results_table(results_df: pd.DataFrame) -> None:
    """Display the results DataFrame with colour-coded risk levels.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: ChurnRisk, ChurnProbability.
    """

    def _highlight_risk(val: str) -> str:
        bg, text = _RISK_COLOURS.get(val, ("#fff", "#000"))
        return f"background-color: {bg}; color: {text}; font-weight: bold"

    styled = results_df.style.format({"ChurnProbability": "{:.2%}"}).applymap(
        _highlight_risk, subset=["ChurnRisk"]
    )
    st.dataframe(styled, use_container_width=True)


def render_risk_distribution(results_df: pd.DataFrame) -> None:
    """Bar chart showing the count of players in each risk tier.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain column: ChurnRisk.
    """
    st.subheader("Risk Distribution")
    counts = (
        results_df["ChurnRisk"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, palette=_BAR_PALETTE, ax=ax)
    ax.set_xlabel("Churn Risk Level", fontsize=11)
    ax.set_ylabel("Player Count", fontsize=11)
    ax.set_title("Player Distribution by Churn Risk", fontsize=13, fontweight="bold")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(int(v)), ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_feature_importance(importance_df: pd.DataFrame) -> None:
    """Horizontal bar chart for Decision Tree feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must contain columns: feature, importance. Already sorted descending.
    """
    st.subheader("Feature Importances (Decision Tree)")
    top10 = importance_df.head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=top10, palette="viridis", ax=ax)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title("Top 10 Feature Importances", fontsize=13, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Evaluation metrics component ─────────────────────────────────────────────

def render_evaluation_metrics(metrics: dict, model_key: str) -> None:
    """Display cross-validation and test-set metrics for a model.

    Parameters
    ----------
    metrics : dict
        Full metrics dict loaded from evaluation_metrics.json.
    model_key : str
        Key inside metrics dict, e.g. 'logistic_regression' or 'decision_tree'.
    """
    m = metrics.get(model_key)
    if not m:
        st.info("No saved metrics found — run `python -m src.train` to generate them.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "CV Accuracy",
        f"{m.get('cv_accuracy_mean', 0):.2%}",
        f"±{m.get('cv_accuracy_std', 0):.2%}",
    )
    c2.metric("CV F1 Score", f"{m.get('cv_f1_mean', 0):.2%}")
    c3.metric("CV ROC-AUC", f"{m.get('cv_roc_auc_mean', 0):.2%}")

    report = m.get("classification_report")
    if report:
        st.markdown("**Classification Report (hold-out test set)**")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.style.format("{:.2f}", na_rep="—"),
            use_container_width=True,
        )


# ── Download button ───────────────────────────────────────────────────────────

def render_download_button(results_df: pd.DataFrame) -> None:
    """Full-width CSV download button for prediction results.

    Parameters
    ----------
    results_df : pd.DataFrame
        The complete results table to export.
    """
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Predictions (CSV)",
        data=csv_bytes,
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── Schema help ───────────────────────────────────────────────────────────────

def render_schema_help() -> None:
    """Show expected CSV schema as a table inside an expander."""
    with st.expander("Expected CSV column schema"):
        schema = pd.DataFrame(
            {
                "Column": [
                    "Age",
                    "Gender",
                    "Location",
                    "GameGenre",
                    "PlayTimeHours",
                    "InGamePurchases",
                    "GameDifficulty",
                    "SessionsPerWeek",
                    "AvgSessionDurationMinutes",
                    "PlayerLevel",
                    "AchievementsUnlocked",
                ],
                "Type": [
                    "int",
                    "str",
                    "str",
                    "str",
                    "float",
                    "int (0/1)",
                    "str",
                    "int",
                    "float",
                    "int",
                    "int",
                ],
                "Allowed Values / Range": [
                    "e.g. 18–60",
                    "Male | Female",
                    "USA | Europe | Asia | Other",
                    "Action | RPG | Simulation | Sports | Strategy",
                    "≥ 0.0",
                    "0 (No) | 1 (Yes)",
                    "Easy | Medium | Hard",
                    "≥ 0",
                    "≥ 0.0",
                    "≥ 1",
                    "≥ 0",
                ],
            }
        )
        st.table(schema)
