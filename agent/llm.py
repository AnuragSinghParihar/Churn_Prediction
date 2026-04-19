"""
llm.py — LLM client for the engagement optimization agent.

Uses Anthropic claude-haiku-4-5 (fast, low-cost, great for structured output).

Environment variable required:
  ANTHROPIC_API_KEY — your Anthropic API key

To use a different provider, replace _call_anthropic() with your own
implementation and keep the generate_recommendations() signature unchanged.
"""
from __future__ import annotations
import os

from agent.prompts import SYSTEM_PROMPT, build_prompt

# Lazy-initialised client (avoids import cost at module load)
_client = None


def _get_client():
    """Return (and lazily initialise) the Anthropic client.

    Raises
    ------
    ImportError
        If the anthropic package is not installed.
    EnvironmentError
        If ANTHROPIC_API_KEY is not set.
    """
    global _client
    if _client is not None:
        return _client

    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package not found. Install with: pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Export it: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    _client = anthropic.Anthropic(api_key=api_key)
    return _client


def generate_recommendations(state) -> str:
    """Call the LLM and return its raw text output (expected JSON string).

    Uses prompt caching on the system prompt to reduce latency and cost
    on repeated calls.

    Parameters
    ----------
    state : AgentState
        Pipeline state with player_summary, churn_interpretation, and
        retrieved_strategies populated.

    Returns
    -------
    str
        Raw LLM response text (should be a valid JSON string).

    Raises
    ------
    ImportError
        If anthropic package is missing.
    EnvironmentError
        If ANTHROPIC_API_KEY is not set.
    anthropic.APIError
        On API call failure.
    """
    client = _get_client()
    import anthropic

    prompt = build_prompt(state)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # prompt caching
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
