"""LLM client for Claude (Sonnet 4.5 and Haiku 4.5) on AWS Bedrock.

The single seam through which every model call flows. Call sites pass
OpenAI-style message lists (with `role: "system"` entries inline); this module
translates them to Anthropic's shape — a separate top-level `system` string and
a user/assistant `messages` list — so call sites need no changes.

Auth: `config.LLM_API_KEY` is an AWS Bedrock API key (a bearer token). The
`AnthropicBedrock` client reads it from the AWS_BEARER_TOKEN_BEDROCK environment
variable, set from config here; the region comes from `config.LLM_AWS_REGION`.
Model IDs carry the "anthropic." Bedrock prefix.

Exposes:
  - chat(): a plain completion (used by the CLI loop)
  - structured(): a schema-constrained completion (extraction/reconciliation),
    using Anthropic structured outputs with a JSON-mode fallback.

Sampling: Claude 4.x rejects `temperature`/`top_p`, so the `temperature`
arguments below are accepted for call-site compatibility but are not forwarded
to the API. Steer behaviour via the prompt instead.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

import config

_client: anthropic.AnthropicBedrock | None = None


def client() -> anthropic.AnthropicBedrock:
    """Lazily construct the shared Bedrock client (key + region from config)."""
    global _client
    if _client is None:
        # AnthropicBedrock authenticates via the AWS credential chain; a Bedrock
        # API key is supplied through this env var.
        os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", config.LLM_API_KEY)
        _client = anthropic.AnthropicBedrock(aws_region=config.LLM_AWS_REGION)
    return _client


def _split(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Split OpenAI-style messages into (system_text, anthropic_messages).

    All `system` entries are concatenated into the top-level system prompt;
    the rest keep their user/assistant roles. Anthropic combines consecutive
    same-role turns, so no interleaving fixup is needed here.
    """
    system_parts: list[str] = []
    convo: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
        else:
            convo.append({"role": role, "content": content})
    return "\n\n".join(system_parts), convo


def _text(resp: anthropic.types.Message) -> str:
    """Concatenate the text blocks of a response (skips any thinking blocks)."""
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,  # accepted for compat; not sent (see module docstring)
) -> str:
    """Plain chat completion -> assistant text."""
    system, convo = _split(messages)
    resp = client().messages.create(
        model=model or config.CHAT_MODEL,
        max_tokens=config.CHAT_MAX_TOKENS,
        system=system,
        messages=convo,
    )
    return _text(resp)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of model text, tolerating ```json code fences."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]  # drop the opening ``` / ```json line
        if s.endswith("```"):
            s = s[: -3]
    return json.loads(s.strip())


def structured(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    schema_name: str = "output",
    model: str | None = None,
    temperature: float | None = None,  # accepted for compat; not sent
) -> dict[str, Any]:
    """Schema-constrained completion -> parsed dict.

    Uses Anthropic structured outputs (`output_config.format`) so the shape is
    guaranteed. Falls back to prompt-instructed JSON if the endpoint/model or
    schema is rejected, then parses.
    """
    system, convo = _split(messages)
    mdl = model or config.EXTRACT_MODEL
    try:
        resp = client().messages.create(
            model=mdl,
            max_tokens=config.EXTRACT_MAX_TOKENS,
            system=system,
            messages=convo,
            output_config={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                }
            },
        )
        return _extract_json(_text(resp))
    except Exception:
        # Fallback: instruct JSON output in the prompt, then parse it out.
        guard = (
            "Respond with ONLY a single JSON object matching this schema. "
            "No prose, no code fences.\n" + json.dumps(schema)
        )
        resp = client().messages.create(
            model=mdl,
            max_tokens=config.EXTRACT_MAX_TOKENS,
            system=(system + "\n\n" + guard).strip(),
            messages=convo,
        )
        return _extract_json(_text(resp))


def health() -> tuple[bool, str]:
    """Cheap reachability check for the CLI startup banner."""
    try:
        out = chat([{"role": "user", "content": "Reply with the single word: ok"}])
        return True, out
    except Exception as e:  # noqa: BLE001 - surface any connection problem
        return False, str(e)
