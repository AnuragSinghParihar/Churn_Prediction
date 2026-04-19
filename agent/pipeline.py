"""
pipeline.py — Multi-step agentic pipeline for Player Engagement Optimization.

State machine:
  INPUT_VALIDATION
       ↓
  PLAYER_ANALYSIS        (compute engagement score & summary)
       ↓
  CHURN_INTERPRETATION   (map risk level to human reasoning)
       ↓
  STRATEGY_RETRIEVAL     (score knowledge base, return top-3)
       ↓
  RECOMMENDATION_GENERATION  (LLM generates personalized JSON)
       ↓
  OUTPUT_FORMATTING      (parse JSON, merge fallbacks, validate schema)
       ↓
     DONE

Each stage receives an AgentState, mutates it, and returns it.
If any stage encounters an unrecoverable error, stage is set to ERROR
and subsequent stages are skipped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


# ── Stage enum ────────────────────────────────────────────────────────────────

class Stage(str, Enum):
    INPUT_VALIDATION = "INPUT_VALIDATION"
    PLAYER_ANALYSIS = "PLAYER_ANALYSIS"
    CHURN_INTERPRETATION = "CHURN_INTERPRETATION"
    STRATEGY_RETRIEVAL = "STRATEGY_RETRIEVAL"
    RECOMMENDATION_GENERATION = "RECOMMENDATION_GENERATION"
    OUTPUT_FORMATTING = "OUTPUT_FORMATTING"
    DONE = "DONE"
    ERROR = "ERROR"


# ── State dataclass ───────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Carries all information through the pipeline.

    Attributes
    ----------
    stage : Stage
        Current pipeline stage.
    player_data : dict
        Raw player feature values.
    churn_result : dict
        Output from the ML model: {prediction, churn_prob, risk}.
    player_summary : str
        Human-readable summary computed during PLAYER_ANALYSIS.
    churn_interpretation : str
        Explanation of why the player is at their risk level.
    retrieved_strategies : list[dict]
        Top-k strategy entries from the knowledge base.
    raw_llm_output : str
        Raw text from the LLM (before JSON parsing).
    output : dict
        Final structured output matching the required schema.
    errors : list[str]
        Accumulated non-fatal warnings and fatal errors.
    """
    stage: Stage
    player_data: dict
    churn_result: dict
    player_summary: str = ""
    churn_interpretation: str = ""
    retrieved_strategies: list = field(default_factory=list)
    raw_llm_output: str = ""
    output: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# ── Required output fields ────────────────────────────────────────────────────

_REQUIRED_FIELDS = [
    "player_behavior_summary",
    "churn_risk_interpretation",
    "engagement_recommendations",
    "supporting_references",
    "ethical_disclaimer",
    "user_experience_notes",
]

_DEFAULT_DISCLAIMER = (
    "These recommendations are data-driven suggestions. All player engagement "
    "strategies should respect user autonomy, avoid manipulative dark patterns, "
    "and comply with applicable regulations (GDPR, COPPA, etc.)."
)

# Required fields a player must supply for analysis to proceed
_REQUIRED_PLAYER_FIELDS = [
    "Age",
    "PlayTimeHours",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "PlayerLevel",
]


# ── Agent class ───────────────────────────────────────────────────────────────

class EngagementAgent:
    """Agentic pipeline that transforms player data into an engagement report.

    The LLM and retrieval functions are injected via instance attributes so
    they can be swapped easily in tests without subclassing.

    Parameters
    ----------
    None — dependencies loaded lazily on first run.
    """

    def __init__(self) -> None:
        from agent.knowledge_base import retrieve_strategies
        from agent.llm import generate_recommendations

        # Injectable for testing (replace with mocks / stubs)
        self._retrieve = retrieve_strategies
        self._generate = generate_recommendations

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, player_data: dict, churn_result: dict) -> dict:
        """Execute the full pipeline and return the structured engagement report.

        Parameters
        ----------
        player_data : dict
            Player feature values (may contain None / missing keys).
        churn_result : dict
            ML model output: {prediction, churn_prob, risk}.

        Returns
        -------
        dict
            Structured report with all 6 required fields.
        """
        state = AgentState(
            stage=Stage.INPUT_VALIDATION,
            player_data=player_data,
            churn_result=churn_result,
        )

        stage_fns = [
            self._validate_input,
            self._analyze_player,
            self._interpret_churn,
            self._retrieve_strategies,
            self._generate_recommendations,
            self._format_output,
        ]

        for fn in stage_fns:
            state = fn(state)
            if state.stage == Stage.ERROR:
                break

        return state.output

    # ── Stage implementations ─────────────────────────────────────────────────

    def _validate_input(self, state: AgentState) -> AgentState:
        state.stage = Stage.INPUT_VALIDATION

        missing = [
            f for f in _REQUIRED_PLAYER_FIELDS
            if state.player_data.get(f) is None
        ]

        if missing:
            state.errors.append(f"Missing required fields: {missing}")
            state.stage = Stage.ERROR
            state.output = _make_error_output(state.errors)
            return state

        state.stage = Stage.PLAYER_ANALYSIS
        return state

    def _analyze_player(self, state: AgentState) -> AgentState:
        state.stage = Stage.PLAYER_ANALYSIS
        d = state.player_data

        sessions = d.get("SessionsPerWeek", 0) or 0
        duration = d.get("AvgSessionDurationMinutes", 0) or 0
        playtime = d.get("PlayTimeHours", 0) or 0
        level = d.get("PlayerLevel", 1) or 1
        achievements = d.get("AchievementsUnlocked", 0) or 0
        purchases = d.get("InGamePurchases", 0) or 0
        age = d.get("Age", "Unknown")

        # Simple composite engagement score (0–100)
        raw_score = sessions * 5 + duration * 0.5 + playtime * 2
        engagement_score = min(100, raw_score)

        # Identify key patterns
        pattern_notes = []
        if sessions < 3:
            pattern_notes.append("infrequent sessions")
        if duration < 20:
            pattern_notes.append("very short play bursts")
        if achievements < 5 and level < 10:
            pattern_notes.append("limited progression")
        if not purchases:
            pattern_notes.append("no purchases yet")

        pattern_str = (
            f" Notable patterns: {', '.join(pattern_notes)}." if pattern_notes else ""
        )

        state.player_summary = (
            f"Player (Age {age}) has accumulated {playtime:.1f} total play hours, "
            f"averaging {sessions} session(s)/week at {duration:.0f} min each. "
            f"Currently Level {level} with {achievements} achievements unlocked. "
            f"{'Has' if purchases else 'Has not'} made in-game purchases. "
            f"Engagement score: {engagement_score:.0f}/100.{pattern_str}"
        )

        state.stage = Stage.CHURN_INTERPRETATION
        return state

    def _interpret_churn(self, state: AgentState) -> AgentState:
        state.stage = Stage.CHURN_INTERPRETATION
        risk = state.churn_result.get("risk", "Medium")
        prob = float(state.churn_result.get("churn_prob", 0.5))

        interpretation_map = {
            "Low": (
                f"Low churn risk ({prob:.1%}). This player shows healthy engagement patterns "
                f"with regular sessions and meaningful progression."
            ),
            "Medium": (
                f"Moderate churn risk ({prob:.1%}). Engagement signals are mixed — "
                f"session frequency or progression may be declining. Timely intervention "
                f"is recommended before further disengagement."
            ),
            "High": (
                f"High churn risk ({prob:.1%}). Critical drop-off indicators detected. "
                f"Immediate, personalised re-engagement is essential to retain this player."
            ),
        }

        state.churn_interpretation = interpretation_map.get(
            risk,
            f"Churn probability: {prob:.1%}.",
        )
        state.stage = Stage.STRATEGY_RETRIEVAL
        return state

    def _retrieve_strategies(self, state: AgentState) -> AgentState:
        state.stage = Stage.STRATEGY_RETRIEVAL
        risk = state.churn_result.get("risk", "Medium")
        state.retrieved_strategies = self._retrieve(
            player_data=state.player_data,
            risk=risk,
            top_k=3,
        )
        state.stage = Stage.RECOMMENDATION_GENERATION
        return state

    def _generate_recommendations(self, state: AgentState) -> AgentState:
        state.stage = Stage.RECOMMENDATION_GENERATION
        try:
            state.raw_llm_output = self._generate(state)
        except Exception as exc:
            state.errors.append(f"LLM generation failed: {exc}")
            # Fall back to rule-based output so the pipeline always completes
            state.raw_llm_output = _make_fallback_json(state)
        state.stage = Stage.OUTPUT_FORMATTING
        return state

    def _format_output(self, state: AgentState) -> AgentState:
        state.stage = Stage.OUTPUT_FORMATTING
        raw = state.raw_llm_output.strip()

        # Attempt JSON parse
        parsed: dict = {}
        try:
            if raw.startswith("{"):
                parsed = json.loads(raw)
            else:
                # LLM sometimes wraps JSON in prose — extract the block
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start != -1 and end > start:
                    parsed = json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        # Merge parsed output with computed fallbacks for every required field
        state.output = {
            "player_behavior_summary": parsed.get(
                "player_behavior_summary", state.player_summary
            ),
            "churn_risk_interpretation": parsed.get(
                "churn_risk_interpretation", state.churn_interpretation
            ),
            "engagement_recommendations": parsed.get(
                "engagement_recommendations", []
            ),
            "supporting_references": parsed.get(
                "supporting_references",
                [s["title"] for s in state.retrieved_strategies],
            ),
            "ethical_disclaimer": parsed.get(
                "ethical_disclaimer", _DEFAULT_DISCLAIMER
            ),
            "user_experience_notes": parsed.get("user_experience_notes", ""),
        }

        state.stage = Stage.DONE
        return state


# ── Helper functions ──────────────────────────────────────────────────────────

def _make_error_output(errors: list[str]) -> dict:
    """Return a valid schema-compliant output for validation failures."""
    return {
        "player_behavior_summary": "Analysis could not be completed due to insufficient data.",
        "churn_risk_interpretation": "Insufficient player data to assess churn risk.",
        "engagement_recommendations": [],
        "supporting_references": [],
        "ethical_disclaimer": _DEFAULT_DISCLAIMER,
        "user_experience_notes": f"Pipeline errors: {'; '.join(errors)}",
    }


def _make_fallback_json(state: AgentState) -> str:
    """Generate rule-based recommendations when the LLM is unavailable."""
    risk = state.churn_result.get("risk", "Medium")

    _fallback_recs: dict[str, list[str]] = {
        "High": [
            "Send a personalised push notification referencing the player's last in-game action to prompt re-engagement.",
            "Offer a limited-time return bonus (e.g., experience boost or rare cosmetic) valid for 48 hours.",
            "Temporarily reduce game difficulty to remove frustration barriers and restore early-win momentum.",
        ],
        "Medium": [
            "Surface a near-miss achievement notification to reignite progression motivation.",
            "Introduce a weekly challenge that fits the player's typical session length.",
            "Highlight social features such as leaderboards or co-op events to create an ongoing reason to return.",
        ],
        "Low": [
            "Introduce advanced competitive content (ranked mode or seasonal events) to deepen long-term engagement.",
            "Add a social goal such as a guild mission to build community attachment.",
            "Preview upcoming content to create anticipation and extend the engagement arc.",
        ],
    }

    output = {
        "player_behavior_summary": state.player_summary,
        "churn_risk_interpretation": state.churn_interpretation,
        "engagement_recommendations": _fallback_recs.get(risk, _fallback_recs["Medium"]),
        "supporting_references": [s["title"] for s in state.retrieved_strategies],
        "ethical_disclaimer": _DEFAULT_DISCLAIMER,
        "user_experience_notes": (
            "Generated using rule-based fallback (LLM unavailable). "
            "Set ANTHROPIC_API_KEY for AI-powered personalised recommendations."
        ),
    }
    return json.dumps(output)
