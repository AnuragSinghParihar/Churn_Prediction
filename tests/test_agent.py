"""
test_agent.py — Unit tests for the agentic engagement optimization pipeline.

Tests cover:
  - Knowledge base retrieval (scoring, top-k, risk matching)
  - Pipeline stage transitions (validation, analysis, interpretation)
  - Full end-to-end runs with mocked LLM
  - Error handling (missing data, LLM failure, malformed JSON)
  - Output schema completeness (all 6 required fields always present)
"""
from __future__ import annotations

import json
import pytest

from agent.pipeline import EngagementAgent, AgentState, Stage, _make_fallback_json
from agent.knowledge_base import retrieve_strategies, KNOWLEDGE_BASE

# ── Shared test fixtures ──────────────────────────────────────────────────────

HIGH_RISK_PLAYER = {
    "Age": 22,
    "Gender": "Male",
    "Location": "USA",
    "GameGenre": "Action",
    "PlayTimeHours": 1.5,
    "InGamePurchases": 0,
    "GameDifficulty": "Hard",
    "SessionsPerWeek": 1,
    "AvgSessionDurationMinutes": 12,
    "PlayerLevel": 3,
    "AchievementsUnlocked": 1,
}

LOW_RISK_PLAYER = {
    "Age": 30,
    "Gender": "Female",
    "Location": "Europe",
    "GameGenre": "RPG",
    "PlayTimeHours": 60.0,
    "InGamePurchases": 1,
    "GameDifficulty": "Medium",
    "SessionsPerWeek": 7,
    "AvgSessionDurationMinutes": 90,
    "PlayerLevel": 65,
    "AchievementsUnlocked": 35,
}

MISSING_DATA_PLAYER = {
    "Age": 25,
    "Gender": "Male",
    # Missing: PlayTimeHours, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel
}

HIGH_CHURN = {"prediction": "Churned", "churn_prob": 0.87, "risk": "High"}
LOW_CHURN = {"prediction": "Retained", "churn_prob": 0.10, "risk": "Low"}
MED_CHURN = {"prediction": "Retained", "churn_prob": 0.52, "risk": "Medium"}

_OUTPUT_SCHEMA = [
    "player_behavior_summary",
    "churn_risk_interpretation",
    "engagement_recommendations",
    "supporting_references",
    "ethical_disclaimer",
    "user_experience_notes",
]


def _mock_agent(llm_response: str) -> EngagementAgent:
    """Return an EngagementAgent with LLM replaced by a mock."""
    agent = EngagementAgent()
    agent._generate = lambda state: llm_response
    return agent


def _good_llm_response(recs: list[str] | None = None) -> str:
    return json.dumps(
        {
            "player_behavior_summary": "Test summary.",
            "churn_risk_interpretation": "Test interpretation.",
            "engagement_recommendations": recs or ["Recommendation A.", "Recommendation B."],
            "supporting_references": ["Re-engagement Campaigns"],
            "ethical_disclaimer": "Respect player autonomy.",
            "user_experience_notes": "Test note.",
        }
    )


# ── Knowledge base tests ──────────────────────────────────────────────────────

def test_retrieve_returns_correct_count():
    results = retrieve_strategies(HIGH_RISK_PLAYER, "High", top_k=3)
    assert len(results) == 3


def test_retrieve_top_k_one():
    results = retrieve_strategies(HIGH_RISK_PLAYER, "High", top_k=1)
    assert len(results) == 1


def test_retrieve_high_risk_includes_applicable_strategies():
    results = retrieve_strategies(HIGH_RISK_PLAYER, "High")
    applicable = [s for s in results if "High" in s["applicable_risk"]]
    assert len(applicable) > 0, "No High-risk strategies in top-3 for a high-risk player"


def test_retrieve_low_risk_does_not_force_high_only():
    results = retrieve_strategies(LOW_RISK_PLAYER, "Low")
    assert len(results) > 0


def test_retrieve_handles_missing_player_data_gracefully():
    # Should not raise even with empty player data
    results = retrieve_strategies({}, "High", top_k=3)
    assert len(results) == 3


def test_retrieve_all_results_are_from_knowledge_base():
    kb_ids = {s["id"] for s in KNOWLEDGE_BASE}
    results = retrieve_strategies(HIGH_RISK_PLAYER, "High")
    for result in results:
        assert result["id"] in kb_ids


# ── Input validation stage ────────────────────────────────────────────────────

def test_missing_required_fields_triggers_error_output():
    agent = _mock_agent(_good_llm_response())
    result = agent.run(MISSING_DATA_PLAYER, HIGH_CHURN)
    assert "Analysis could not be completed" in result["player_behavior_summary"]


def test_missing_fields_still_returns_full_schema():
    agent = _mock_agent(_good_llm_response())
    result = agent.run(MISSING_DATA_PLAYER, HIGH_CHURN)
    for field in _OUTPUT_SCHEMA:
        assert field in result, f"Missing schema field: {field}"


# ── Player analysis stage ─────────────────────────────────────────────────────

def test_player_summary_is_computed():
    agent = _mock_agent(_good_llm_response())
    state = AgentState(
        stage=Stage.INPUT_VALIDATION,
        player_data=HIGH_RISK_PLAYER,
        churn_result=HIGH_CHURN,
    )
    state = agent._analyze_player(state)
    assert state.player_summary != ""
    assert "22" in state.player_summary  # Age should appear


def test_player_summary_mentions_sessions():
    agent = _mock_agent("{}")
    state = AgentState(
        stage=Stage.PLAYER_ANALYSIS,
        player_data=HIGH_RISK_PLAYER,
        churn_result=HIGH_CHURN,
    )
    state = agent._analyze_player(state)
    assert "session" in state.player_summary.lower()


# ── Churn interpretation stage ────────────────────────────────────────────────

def test_churn_interpretation_high_risk_mentions_critical():
    agent = _mock_agent("{}")
    state = AgentState(
        stage=Stage.CHURN_INTERPRETATION,
        player_data=HIGH_RISK_PLAYER,
        churn_result=HIGH_CHURN,
    )
    state = agent._interpret_churn(state)
    text = state.churn_interpretation.lower()
    assert "high" in text or "critical" in text or "immediate" in text


def test_churn_interpretation_includes_probability():
    agent = _mock_agent("{}")
    state = AgentState(
        stage=Stage.CHURN_INTERPRETATION,
        player_data=LOW_RISK_PLAYER,
        churn_result=LOW_CHURN,
    )
    state = agent._interpret_churn(state)
    assert "10.0%" in state.churn_interpretation or "10%" in state.churn_interpretation


# ── Full pipeline tests ───────────────────────────────────────────────────────

def test_high_risk_pipeline_output_schema():
    agent = _mock_agent(_good_llm_response(["Go send notification.", "Offer bonus."]))
    result = agent.run(HIGH_RISK_PLAYER, HIGH_CHURN)
    for field in _OUTPUT_SCHEMA:
        assert field in result, f"Missing: {field}"


def test_low_risk_pipeline_output_schema():
    agent = _mock_agent(_good_llm_response())
    result = agent.run(LOW_RISK_PLAYER, LOW_CHURN)
    for field in _OUTPUT_SCHEMA:
        assert field in result


def test_recommendations_are_list():
    agent = _mock_agent(_good_llm_response())
    result = agent.run(HIGH_RISK_PLAYER, HIGH_CHURN)
    assert isinstance(result["engagement_recommendations"], list)


def test_high_risk_recommendations_not_empty():
    agent = _mock_agent(_good_llm_response(["Do X.", "Do Y.", "Do Z."]))
    result = agent.run(HIGH_RISK_PLAYER, HIGH_CHURN)
    assert len(result["engagement_recommendations"]) > 0


# ── LLM failure and fallback ──────────────────────────────────────────────────

def test_llm_failure_produces_valid_output():
    agent = EngagementAgent()

    def _fail(state):
        raise RuntimeError("Simulated API failure")

    agent._generate = _fail
    result = agent.run(HIGH_RISK_PLAYER, HIGH_CHURN)

    assert isinstance(result["engagement_recommendations"], list)
    assert len(result["engagement_recommendations"]) > 0


def test_llm_failure_sets_fallback_note():
    agent = EngagementAgent()
    agent._generate = lambda state: (_ for _ in ()).throw(Exception("fail"))
    result = agent.run(HIGH_RISK_PLAYER, HIGH_CHURN)
    notes = result.get("user_experience_notes", "").lower()
    assert "fallback" in notes


def test_malformed_json_falls_back_gracefully():
    agent = _mock_agent("Sorry, I cannot help with this.")
    result = agent.run(HIGH_RISK_PLAYER, HIGH_CHURN)
    # player_behavior_summary should fall back to computed summary
    assert result["player_behavior_summary"] != ""
    assert isinstance(result["engagement_recommendations"], list)


def test_partial_json_uses_available_fields():
    partial = json.dumps(
        {"player_behavior_summary": "Partial summary.", "engagement_recommendations": ["Rec X."]}
    )
    agent = _mock_agent(partial)
    result = agent.run(HIGH_RISK_PLAYER, HIGH_CHURN)
    assert result["player_behavior_summary"] == "Partial summary."
    assert "Rec X." in result["engagement_recommendations"]
    # Other fields fall back cleanly
    assert result["ethical_disclaimer"] != ""


# ── Fallback JSON helper ──────────────────────────────────────────────────────

def test_make_fallback_json_valid_for_all_risks():
    for risk, churn in [("High", HIGH_CHURN), ("Medium", MED_CHURN), ("Low", LOW_CHURN)]:
        state = AgentState(
            stage=Stage.RECOMMENDATION_GENERATION,
            player_data=HIGH_RISK_PLAYER,
            churn_result=churn,
            player_summary="Test summary.",
            churn_interpretation="Test interpretation.",
            retrieved_strategies=[],
        )
        raw = _make_fallback_json(state)
        parsed = json.loads(raw)
        assert isinstance(parsed["engagement_recommendations"], list)
        assert len(parsed["engagement_recommendations"]) > 0
