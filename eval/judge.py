"""LLM-as-judge for the subjective eval axes.

Each judge is a narrow yes/no over a single reply, using constrained JSON output on
the configured JUDGE_MODEL (Claude on AWS Bedrock). Judged axes are indicative rather
than authoritative: the deterministic DB checks in run_eval carry the
contradiction-handling verdict.

Each function returns (passed: bool, reason: str).
"""

from __future__ import annotations

import config
from src import llm

_BOOL = {
    "type": "object",
    "properties": {"verdict": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["verdict", "reason"],
}


def _ask(system: str, user: str) -> tuple[bool, str]:
    try:
        raw = llm.structured(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            _BOOL,
            schema_name="judgement",
            model=config.JUDGE_MODEL,
        )
        return bool(raw.get("verdict", False)), str(raw.get("reason", ""))
    except Exception as e:  # noqa: BLE001
        return False, f"judge error: {e}"


def recall(reply: str, truth: str) -> tuple[bool, str]:
    """True if the reply correctly conveys the ground-truth fact."""
    v, r = _ask(
        "You grade a companion's reply. Given the GROUND TRUTH about the user and the "
        "REPLY, decide whether the reply correctly conveys the ground-truth fact (it may "
        "be phrased differently, but the fact must be right). verdict=true if correct.",
        f"GROUND TRUTH: {truth}\nREPLY: {reply}",
    )
    return v, r


def no_leak(reply: str, stale: str) -> tuple[bool, str]:
    """True (pass) if the reply does NOT treat the stale fact as still true."""
    leaked, r = _ask(
        "You grade a companion's reply for using OUTDATED information. Given a STALE "
        "(no-longer-true) fact and the REPLY, decide whether the reply treats the stale "
        "fact as still true. verdict=true if it DOES leak the stale fact.",
        f"STALE FACT: {stale}\nREPLY: {reply}",
    )
    return (not leaked), r  # pass = did not leak


def persona(reply: str, canon: str) -> tuple[bool, str]:
    """True (pass) if the reply does NOT contradict the canon opinion."""
    contradicts, r = _ask(
        "You grade a companion's reply for staying in character. Given the companion's "
        "ESTABLISHED OPINION and the REPLY, decide whether the reply CONTRADICTS that "
        "opinion (takes the opposite stance). Agreement or not mentioning it is fine. "
        "verdict=true only if it contradicts.",
        f"ESTABLISHED OPINION: {canon}\nREPLY: {reply}",
    )
    return (not contradicts), r  # pass = no contradiction


def tone(reply: str) -> tuple[bool, str]:
    """True (pass) if the reply stays in-character rather than flattening to a
    generic assistant voice."""
    generic, r = _ask(
        "You grade whether a reply sounds like a warm, distinctive person with their own "
        "personality and opinions, or like a generic AI assistant (neutral, corporate, "
        "'I'm happy to help', over-hedged, no personality). verdict=true if it reads as a "
        "GENERIC ASSISTANT rather than a real character.",
        f"REPLY: {reply}",
    )
    return (not generic), r  # pass = in character
