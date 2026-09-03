"""Central configuration for the companion core loop.

Tunable retrieval and decay parameters live here so behaviour can be adjusted
without touching logic. Values are read from the environment first, so models,
credentials, and paths can be changed via env vars alone.
"""

from __future__ import annotations

import os
from pathlib import Path

# Secrets and overrides live in an untracked .env (see .env.example). Load it
# before any os.getenv() below so keys like COMPANION_LLM_API_KEY resolve. Env
# vars set before launch take precedence over .env values.
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("COMPANION_DB", ROOT / "companion.db"))
# Persona library: each *.yaml here is a selectable companion. COMPANION_PERSONA
# pins one and suppresses the chat-time picker; otherwise the CLI lists this
# directory for the user to choose. Defaults to Kai.
PERSONAS_DIR = Path(os.getenv("COMPANION_PERSONAS_DIR", ROOT / "personas"))
PERSONA_PATH = Path(os.getenv("COMPANION_PERSONA", PERSONAS_DIR / "kai.yaml"))

# --- LLM serving ---------------------------------------------------------
# Claude via the Anthropic SDK on AWS Bedrock (src/llm.py builds an
# AnthropicBedrock client). The key is a Bedrock bearer token read from the
# environment; kept out of source.
LLM_API_KEY = os.getenv("COMPANION_LLM_API_KEY", "")
# AWS Bedrock region. The region is not encoded in the key, so set it to where
# this account has Claude model access granted.
LLM_AWS_REGION = os.getenv("COMPANION_AWS_REGION", "us-east-1")
# Bedrock model IDs carry a provider prefix; cross-region inference profiles add
# a geo prefix ("us."). Models are split by role: Sonnet 4.5 handles the
# persona-critical chat and nuanced judging, while extraction is a narrow
# schema-constrained task, so Haiku 4.5 runs it more cheaply on every turn.
CHAT_MODEL = os.getenv("COMPANION_CHAT_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
EXTRACT_MODEL = os.getenv("COMPANION_EXTRACT_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
# Contradiction judgement wants the strongest model; overridable on its own.
JUDGE_MODEL = os.getenv("COMPANION_JUDGE_MODEL", CHAT_MODEL)
# Anthropic requires max_tokens on every request; it is a hard output cap.
CHAT_MAX_TOKENS = int(os.getenv("COMPANION_CHAT_MAX_TOKENS", "2048"))
EXTRACT_MAX_TOKENS = int(os.getenv("COMPANION_EXTRACT_MAX_TOKENS", "2048"))

# Embeddings are OFF by default — no local model server is required to run
# this project out of the box. Without a vector, retrieval and reconciliation
# fall back to entity/subject matching instead of semantic similarity — see
# src/embeddings.py and the fallback branches in src/retrieval.py and
# src/reconcile.py. Opt in with COMPANION_EMBEDDINGS_ENABLED=true once Ollama
# (or another OpenAI-embeddings-compatible endpoint) is available, and point
# COMPANION_EMBED_BASE_URL/COMPANION_EMBED_API_KEY/COMPANION_EMBED_MODEL at it
# if it isn't the local Ollama default.
EMBEDDINGS_ENABLED = os.getenv("COMPANION_EMBEDDINGS_ENABLED", "false").strip().lower() in (
    "true", "1", "yes", "on",
)
EMBED_BASE_URL = os.getenv("COMPANION_EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_API_KEY = os.getenv("COMPANION_EMBED_API_KEY", "ollama")
EMBED_MODEL = os.getenv("COMPANION_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("COMPANION_EMBED_DIM", "768"))

# --- Working memory ------------------------------------------------------
WORKING_MEMORY_TURNS = 8  # recent turns kept verbatim in the prompt

# --- Retrieval -----------------------------------------------------------
RETRIEVAL_TOP_K = 7
SIMILARITY_FLOOR = 0.35
RANK_WEIGHTS = {
    "cosine": 0.55,
    "entity_match": 0.20,
    "recency": 0.10,
    "salience": 0.10,
    "confidence": 0.05,
}
PER_KIND_BUDGET = {"user_state": 2}  # cap volatile kinds

# Recency half-lives in days, per kind (feeds decay).
HALF_LIFE_DAYS = {
    "user_semantic": 3650,
    "user_preference": 365,
    "user_episodic": 90,
    "user_state": 7,
    "persona_canon": 100000,
    "persona_stated": 365,
}

# --- Extraction ----------------------------------------------------------
CONFIDENCE_FLOOR = 0.6
SALIENCE_FLOOR = 0.3
RECONCILE_NEIGHBORS = 5
# Cross-subject facts must be near-identical in meaning before a supersede or
# refine is allowed, preventing "new job" from retiring "broke up". Same-subject
# neighbours bypass this, since the subject already ties them together.
RECONCILE_SIM_FLOOR = 0.60        # min cosine to offer a cross-subject neighbour
# Min cosine to act on a cross-subject supersede/refine. Same-attribute
# paraphrases and unrelated pairs overlap in cosine, so this is only a coarse
# sanity floor; the LLM classifier is the primary decider. Aligned with the
# neighbour floor, so any fact offered as a neighbour is left to the classifier.
RECONCILE_CROSS_SUBJECT_MIN = 0.60

USER_KINDS = ("user_semantic", "user_preference", "user_episodic", "user_state")
PERSONA_KINDS = ("persona_canon", "persona_stated")
ALL_KINDS = USER_KINDS + PERSONA_KINDS
