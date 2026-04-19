"""
prompts.py — Prompt templates for the engagement optimization agent.

Design principles:
  - System prompt defines role and strict output rules
  - User prompt injects all context so the LLM is fully grounded
  - Output schema is enforced in the prompt (no hallucination escape hatch)
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert game retention analyst and player engagement strategist.

Your role is to analyse player behaviour data, interpret churn risk, and produce
specific, actionable engagement recommendations grounded strictly in the provided
player data and retrieved strategy context.

Output rules (MUST follow):
1. Base ALL recommendations on the provided player data and retrieved strategies only.
2. Do NOT invent statistics, features, or claims not present in the input.
3. Recommendations must be specific to THIS player's profile — not generic advice.
4. Return ONLY valid JSON matching the schema below — no markdown fences, no prose.
5. Keep recommendations concise (1–2 sentences), actionable, and respectful.
6. The ethical_disclaimer must acknowledge responsible use of engagement mechanics.
"""

_RECOMMENDATION_TEMPLATE = """\
## Player Data
{player_data}

## Behaviour Summary (computed)
{player_summary}

## Churn Risk Assessment
{churn_interpretation}

## Retrieved Engagement Strategies (use as grounding — do not invent others)
{strategies}

## Your Task
Analyse this player's engagement patterns and generate personalised recommendations.

Return ONLY the following JSON (no markdown, no extra text outside the braces):
{{
  "player_behavior_summary": "<2-3 sentence summary of this player's engagement patterns and behaviour>",
  "churn_risk_interpretation": "<1-2 sentence explanation of WHY this player is at their risk level, referencing specific features>",
  "engagement_recommendations": [
    "<specific, actionable recommendation tailored to this player — reference their actual data>",
    "<second recommendation>",
    "<third recommendation>"
  ],
  "supporting_references": [
    "<title of strategy from retrieved context that supports recommendation 1>",
    "<title of strategy that supports recommendation 2>"
  ],
  "ethical_disclaimer": "<one sentence on responsible use of this recommendation>",
  "user_experience_notes": "<optional: note on player experience quality or additional context>"
}}
"""


def build_prompt(state) -> str:
    """Construct the full user-turn prompt from current agent state.

    Parameters
    ----------
    state : AgentState
        Must have player_data, player_summary, churn_interpretation,
        and retrieved_strategies populated.

    Returns
    -------
    str
        Formatted prompt string ready to send to the LLM.
    """
    import json

    strategies_text = "\n".join(
        f"[{s['id']}] {s['title']}: {s['content']}"
        for s in state.retrieved_strategies
    ) or "No strategies retrieved."

    return _RECOMMENDATION_TEMPLATE.format(
        player_data=json.dumps(state.player_data, indent=2),
        player_summary=state.player_summary,
        churn_interpretation=state.churn_interpretation,
        strategies=strategies_text,
    )
