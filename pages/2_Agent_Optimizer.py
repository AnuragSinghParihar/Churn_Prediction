"""
pages/2_Agent_Optimizer.py — Streamlit page: Agentic Engagement Optimizer.

Accessible via the Streamlit sidebar navigation as "Agent Optimizer".
Inputs player data manually or accepts churn results from the main app.
"""
from __future__ import annotations

import json

import streamlit as st

from agent.pipeline import EngagementAgent
from agent.export import export_pdf, REPORTLAB_AVAILABLE

st.set_page_config(
    page_title="Engagement Optimizer",
    page_icon="🤖",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 AI Engagement Optimization Assistant")
st.markdown(
    "Analyse a player's behaviour profile and receive **AI-generated, personalised "
    "engagement strategies** — powered by a 6-stage agentic pipeline."
)

with st.expander("How the agent works", expanded=False):
    st.code(
        """\
INPUT_VALIDATION       — Check required fields are present
       ↓
PLAYER_ANALYSIS        — Compute engagement score & behaviour summary
       ↓
CHURN_INTERPRETATION   — Map risk level to a human-readable explanation
       ↓
STRATEGY_RETRIEVAL     — Score 8-item knowledge base, return top-3 matches
       ↓
RECOMMENDATION_GENERATION — Claude Haiku generates personalised JSON output
       ↓
OUTPUT_FORMATTING      — Parse JSON, validate schema, merge fallbacks
       ↓
     DONE ✓
""",
        language="text",
    )

st.markdown("---")

# ── Sidebar — Player Input ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Player Profile")

    age = st.slider("Age", 16, 65, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    location = st.selectbox("Location", ["USA", "Europe", "Asia", "Other"])
    genre = st.selectbox(
        "Game Genre", ["Action", "RPG", "Simulation", "Sports", "Strategy"]
    )
    difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
    playtime = st.number_input("Total Play Time (Hours)", 0.0, 200.0, 5.5, step=0.5)
    purchases = st.selectbox(
        "In-Game Purchases", [0, 1], format_func=lambda x: "Yes" if x else "No"
    )
    sessions = st.slider("Sessions per Week", 0, 14, 3)
    duration = st.slider("Avg Session Duration (min)", 5, 180, 45)
    level = st.slider("Player Level", 1, 100, 20)
    achievements = st.slider("Achievements Unlocked", 0, 50, 5)

    st.markdown("---")
    st.subheader("Churn Assessment")
    risk_input = st.selectbox("Churn Risk Level", ["Low", "Medium", "High"], index=1)
    prob_input = st.slider("Churn Probability", 0.0, 1.0, 0.50, step=0.01)

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

player_data = {
    "Age": age,
    "Gender": gender,
    "Location": location,
    "GameGenre": genre,
    "PlayTimeHours": playtime,
    "InGamePurchases": purchases,
    "GameDifficulty": difficulty,
    "SessionsPerWeek": sessions,
    "AvgSessionDurationMinutes": duration,
    "PlayerLevel": level,
    "AchievementsUnlocked": achievements,
}

churn_result = {
    "prediction": "Churned" if prob_input > 0.5 else "Retained",
    "churn_prob": prob_input,
    "risk": risk_input,
}

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running agent pipeline …"):
        agent = EngagementAgent()
        report = agent.run(player_data, churn_result)
    st.session_state["agent_report"] = report
    st.session_state["agent_player_data"] = player_data
    st.session_state["agent_churn"] = churn_result

# ── Display results ────────────────────────────────────────────────────────────
if "agent_report" in st.session_state:
    report = st.session_state["agent_report"]
    saved_churn = st.session_state.get("agent_churn", churn_result)

    risk = saved_churn["risk"]
    prob = saved_churn["churn_prob"]

    _risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")
    st.markdown(f"## {_risk_icon} {risk} Churn Risk &nbsp;·&nbsp; {prob:.1%} probability")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Behaviour Summary")
        st.info(report.get("player_behavior_summary", "—"))

        st.subheader("Churn Risk Interpretation")
        _msg = report.get("churn_risk_interpretation", "—")
        if risk == "High":
            st.error(_msg)
        elif risk == "Medium":
            st.warning(_msg)
        else:
            st.success(_msg)

    with col2:
        st.subheader("Engagement Recommendations")
        recs = report.get("engagement_recommendations", [])
        if recs:
            for i, rec in enumerate(recs, 1):
                st.success(f"**{i}.** {rec}")
        else:
            st.info("No recommendations generated.")

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        refs = report.get("supporting_references", [])
        if refs:
            st.subheader("Knowledge Base References")
            for ref in refs:
                st.markdown(f"- {ref}")

    with col4:
        notes = report.get("user_experience_notes", "")
        if notes:
            st.subheader("UX Notes")
            st.markdown(notes)

    disclaimer = report.get("ethical_disclaimer", "")
    if disclaimer:
        st.caption(f"⚠️  {disclaimer}")

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        json_bytes = json.dumps(report, indent=2).encode("utf-8")
        st.download_button(
            "⬇ Download Report (JSON)",
            data=json_bytes,
            file_name="engagement_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_b:
        if REPORTLAB_AVAILABLE:
            try:
                pdf_bytes = export_pdf(
                    report, st.session_state.get("agent_player_data", player_data)
                )
                st.download_button(
                    "⬇ Download Report (PDF)",
                    data=pdf_bytes,
                    file_name="engagement_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"PDF export failed: {exc}")
        else:
            st.info("Install reportlab for PDF export: `pip install reportlab`")

else:
    st.info(
        "Set the player profile in the sidebar and click **Run Analysis** to generate "
        "an engagement report."
    )

    # Show example output schema
    with st.expander("Expected output schema"):
        st.json(
            {
                "player_behavior_summary": "2-3 sentence player behaviour overview",
                "churn_risk_interpretation": "Why this player is at their risk level",
                "engagement_recommendations": [
                    "Specific recommendation 1",
                    "Specific recommendation 2",
                    "Specific recommendation 3",
                ],
                "supporting_references": ["Strategy name from knowledge base"],
                "ethical_disclaimer": "Responsible use note",
                "user_experience_notes": "Optional UX context",
            }
        )
