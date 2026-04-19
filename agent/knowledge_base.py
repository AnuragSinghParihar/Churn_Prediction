"""
knowledge_base.py — Static knowledge base of game engagement & retention strategies.

Retrieval uses a simple heuristic scorer:
  +2 if the player's risk level matches the strategy's applicable_risk list
  +1 for each feature signal that matches the player's actual data

No vector DB or embeddings required — stays fast and fully offline.
"""
from __future__ import annotations

# ── Knowledge entries ─────────────────────────────────────────────────────────

KNOWLEDGE_BASE: list[dict] = [
    {
        "id": "kb_001",
        "title": "Progressive Achievement Systems",
        "content": (
            "Tiered achievement systems with visible progress bars create momentum. "
            "Near-miss effects — showing players they are close to an achievement — "
            "significantly reduce drop-off. Research shows 40% higher retention when "
            "milestone-based rewards are surfaced proactively."
        ),
        "tags": ["achievements", "progression", "retention"],
        "applicable_risk": ["Medium", "High"],
        "feature_signals": {"AchievementsUnlocked": "low"},
    },
    {
        "id": "kb_002",
        "title": "Session Length Optimisation",
        "content": (
            "Short session designs (5–15 minutes) with clear start/end points serve "
            "casual players. A daily challenge completable in one session improves "
            "frequency. Avoid punishing short sessions — auto-save state immediately "
            "to prevent frustration-based abandonment."
        ),
        "tags": ["session", "casual", "frequency"],
        "applicable_risk": ["Medium", "High"],
        "feature_signals": {"AvgSessionDurationMinutes": "low", "SessionsPerWeek": "low"},
    },
    {
        "id": "kb_003",
        "title": "Personalised Difficulty Curves",
        "content": (
            "Dynamic difficulty adjustment (DDA) keeps players in a flow state. "
            "Players who fail the same level repeatedly churn at 3× the average rate. "
            "Adaptive AI opponents and optional easy-mode unlocks prevent frustration-driven churn."
        ),
        "tags": ["difficulty", "flow", "frustration"],
        "applicable_risk": ["High"],
        "feature_signals": {"GameDifficulty": "Hard"},
    },
    {
        "id": "kb_004",
        "title": "Social and Community Features",
        "content": (
            "Social ties are the strongest single retention lever. Players with at least "
            "one in-game friend churn at 3× lower rates. Guilds, co-op missions, and "
            "leaderboards create ongoing social obligations that sustain engagement."
        ),
        "tags": ["social", "community", "guilds", "leaderboard"],
        "applicable_risk": ["Medium", "High"],
        "feature_signals": {"SessionsPerWeek": "low"},
    },
    {
        "id": "kb_005",
        "title": "Re-engagement Campaigns",
        "content": (
            "Players inactive for 7+ days respond best to personalised push notifications "
            "referencing their last in-game action (e.g., 'Your base needs you!'). "
            "Limited-time return bonuses convert 20–30% of lapsed players when delivered "
            "within the first 14 days of inactivity."
        ),
        "tags": ["reengagement", "notification", "lapse", "win-back"],
        "applicable_risk": ["High"],
        "feature_signals": {"PlayTimeHours": "low", "SessionsPerWeek": "low"},
    },
    {
        "id": "kb_006",
        "title": "Daily Login Reward Loops",
        "content": (
            "Predictable daily login bonuses build habit formation without exploitative "
            "mechanics. Escalating streak rewards (Day 1: 10 coins → Day 7: rare item) "
            "motivate players to return consistently. Avoid punishing missed days — "
            "grace periods maintain goodwill."
        ),
        "tags": ["rewards", "habit", "engagement", "login"],
        "applicable_risk": ["Low", "Medium"],
        "feature_signals": {"InGamePurchases": 0},
    },
    {
        "id": "kb_007",
        "title": "Early Onboarding Wins",
        "content": (
            "70% of churn happens in the first 3 sessions. Interactive tutorials that "
            "deliver an early victory within the first 5 minutes dramatically improve "
            "Day-1 retention. Show, don't tell — avoid long text tutorials; use "
            "contextual prompts during play."
        ),
        "tags": ["onboarding", "tutorial", "new-player", "D1 retention"],
        "applicable_risk": ["High"],
        "feature_signals": {"PlayerLevel": "low", "AchievementsUnlocked": "low"},
    },
    {
        "id": "kb_008",
        "title": "Seasonal Events and Limited Content",
        "content": (
            "Time-limited seasonal events create urgency and give lapsed players a "
            "compelling reason to return. Events tied to real-world holidays see "
            "40–80% engagement spikes. Exclusive cosmetic rewards — not power "
            "advantages — maintain fairness while driving participation."
        ),
        "tags": ["events", "seasonal", "FOMO", "limited-time"],
        "applicable_risk": ["Low", "Medium", "High"],
        "feature_signals": {},
    },
]

# ── Retrieval thresholds ───────────────────────────────────────────────────────

_LOW_THRESHOLD: dict[str, float] = {
    "PlayTimeHours": 5.0,
    "SessionsPerWeek": 3.0,
    "AvgSessionDurationMinutes": 30.0,
    "PlayerLevel": 10.0,
    "AchievementsUnlocked": 5.0,
}


# ── Retrieval function ────────────────────────────────────────────────────────

def retrieve_strategies(
    player_data: dict,
    risk: str,
    top_k: int = 3,
) -> list[dict]:
    """Return the top-k most relevant engagement strategies for a player.

    Scoring heuristic (higher = more relevant):
      +2  strategy's applicable_risk includes the player's risk level
      +1  per feature signal matching player's actual data

    Parameters
    ----------
    player_data : dict
        Player feature values (may include missing/None entries).
    risk : str
        'Low' | 'Medium' | 'High' churn risk level.
    top_k : int
        Number of strategies to return (default 3).

    Returns
    -------
    list[dict]
        Top-k strategy entries sorted by relevance score (descending).
    """
    scored: list[tuple[int, dict]] = []

    for strategy in KNOWLEDGE_BASE:
        score = 0

        if risk in strategy["applicable_risk"]:
            score += 2

        for feature, signal in strategy["feature_signals"].items():
            val = player_data.get(feature)
            if val is None:
                continue
            if signal == "low":
                threshold = _LOW_THRESHOLD.get(feature, 10.0)
                if isinstance(val, (int, float)) and val < threshold:
                    score += 1
            elif signal == "Hard" and val == "Hard":
                score += 1
            elif signal == 0 and val == 0:
                score += 1

        scored.append((score, strategy))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_k]]
