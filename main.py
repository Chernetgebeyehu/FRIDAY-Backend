"""
FRIDAY FastAPI Backend Server
=============================

This is the brain of FRIDAY's cloud infrastructure.

Think of this file as a telephone switchboard operator:
- Android phones call in with messages
- This server figures out which AI to call
- Gets the AI's response
- Sends it back to the phone

The server NEVER exposes AI API keys to the phone.
The phone only needs to know this server's URL + secret token.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import os
import json
import time
import logging
from dotenv import load_dotenv

# Optional dependency: the official Anthropic SDK, used only by the Claude
# fallback provider. Import is guarded so a deployment without the package
# (or without ANTHROPIC_API_KEY) still boots — Claude is simply skipped in
# the provider chain.
try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

# ─── LOAD ENVIRONMENT VARIABLES ───────────────────────────────────────────────
# This reads your .env file and puts the secrets into os.environ
# Like opening your safe and putting the contents on your desk (privately)
load_dotenv()

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Optional providers in the automatic fallback chain (skipped when unset).
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
SECRET_TOKEN       = os.getenv("SECRET_TOKEN", "")

# CHANGE (fresh deployment): deployment-specific values now come from env vars
# instead of being hardcoded, so this backend is portable across any host.
# ALLOWED_ORIGINS: comma-separated list of origins allowed by CORS ("*" = any).
# APP_REFERER / APP_TITLE: identify this app to OpenRouter (previously hardcoded).
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]
APP_REFERER = os.getenv("APP_REFERER", "https://github.com/cheregebe/FRIDAY-AI-Assistant")
APP_TITLE   = os.getenv("APP_TITLE", "FRIDAY AI Assistant")

# CHANGE (production upgrade): single source of truth for the app version.
# Set APP_VERSION on Railway to change the reported version without a deploy.
# The default is bumped on every code change so /version fingerprints which
# build Railway is actually serving (the 2026-08-04 incident was a stale
# deployment that was indistinguishable from current code until probed).
APP_VERSION  = os.getenv("APP_VERSION", "2.1.0")
SERVICE_NAME = "FRIDAY Backend"

# ─── AI MODEL SELECTION (env-driven — NEVER hardcode model IDs) ───────────────
# WHY models must never be hardcoded:
# AI providers retire and delist models on their own schedule. A hardcoded ID
# rots silently: the day the provider drops it, every /chat request starts
# failing with an upstream 404 — which (before this change) was proxied through
# FastAPI's app-level 404 handler and reached the Android app as
# {"error":"Endpoint not found","path":"/chat"}, indistinguishable from a
# missing route. That exact incident happened with the previously pinned
# "gemini-2.5-flash" (retired by Google) and "google/gemma-3-4b-it:free"
# (delisted by OpenRouter). Env-driven models mean a rot fix is a Railway
# Variables edit + restart — no code change, no redeploy of the Android app.
#
# GEMINI_MODEL: Gemini model code. Fallback is the "-latest" rolling alias,
# which Google keeps pointed at the current flash model precisely so servers
# never pin a snapshot that can retire.
# DEFAULT_OPENROUTER_MODEL: used when the Android app doesn't specify one.
# Fallback verified live in OpenRouter's catalog (2026-08-01).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
# Model-level fallback for Gemini: if the env-configured model is rejected
# upstream (typo in a Railway variable, or a pinned snapshot Google retired),
# the same request is retried once with this rolling alias — Google keeps it
# pointed at the current flash model, so it cannot rot.
GEMINI_FALLBACK_MODEL = "gemini-flash-latest"
DEFAULT_OPENROUTER_MODEL = os.getenv(
    "DEFAULT_OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"
)
# Models for the optional chain providers — same rule: env-driven, modern
# fallbacks, never pinned in code. claude-haiku-4-5 is the fastest/cheapest
# current Claude model; gpt-5-mini is OpenAI's small current-generation model.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")

# ─── PROVIDER FALLBACK CHAIN ──────────────────────────────────────────────────
# FRIDAY is Gemini-first. When the active provider fails with a *transient*
# class of error (timeout, 429, 5xx, unreachable, model unavailable), the
# request automatically continues down the chain — the user never has to
# press Send again, and the conversation history travels with the request,
# so context is preserved across the switch.
#
# Order: gemini → openai → claude → openrouter.
# openai/claude participate only when their API keys are configured.
#
# Recovery (requirement: switch back when Gemini returns): a failing provider
# is put on a short cooldown instead of being disabled. Every request after
# the cooldown expires starts at the top of the chain again, so Gemini
# automatically becomes primary the moment it's healthy.
PROVIDER_CHAIN = ["gemini", "openai", "claude", "openrouter"]
PROVIDER_LABELS = {
    "gemini": "Gemini",
    "openai": "OpenAI",
    "claude": "Claude",
    "openrouter": "OpenRouter",
}
PROVIDER_COOLDOWN_SECONDS = float(os.getenv("PROVIDER_COOLDOWN_SECONDS", "120"))
# Per-provider call timeout. Kept well under the Android app's 90s read
# timeout so a chain that walks several providers still answers in time.
PROVIDER_TIMEOUT_SECONDS = float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "20"))

# provider name → monotonic timestamp until which it is skipped.
_provider_cooldown_until: dict = {}


def provider_configured(name: str) -> bool:
    """A provider participates in the chain only when its key is set."""
    return {
        "gemini": bool(GEMINI_API_KEY),
        "openai": bool(OPENAI_API_KEY),
        "claude": bool(ANTHROPIC_API_KEY) and ANTHROPIC_SDK_AVAILABLE,
        "openrouter": bool(OPENROUTER_API_KEY),
    }.get(name, False)

# Uptime tracking — monotonic clock so clock adjustments never skew uptime.
_START_MONOTONIC = time.monotonic()

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
# Logging = writing a diary of everything that happens
# When something breaks, you read the diary to understand what went wrong
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("friday-backend")

# CHANGE (security fix found during audit): httpx logs every request URL at
# INFO level, and the Gemini URL contains "?key=<GEMINI_API_KEY>" — which would
# write the real API key into Railway's production logs. Raise httpx to WARNING
# so request URLs (and the key) are never logged.
logging.getLogger("httpx").setLevel(logging.WARNING)

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
# This creates the web server application
# Think of it as building the building before putting rooms in it
app = FastAPI(
    title="FRIDAY Backend",
    description="Secure AI orchestration layer for the FRIDAY Android assistant",
    version=APP_VERSION
)

# ─── CORS MIDDLEWARE ──────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Without this, browsers block requests from unknown sources
# We're restrictive: only our Android app should talk to this server
# CHANGE: origins are configurable via the ALLOWED_ORIGINS env var (default "*").
# Set ALLOWED_ORIGINS on Railway to lock this down without a code change.
# Note: the Android app is protected by SECRET_TOKEN, not CORS — CORS only
# affects browsers, so "*" is acceptable here as long as the token is set.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
# GEMINI_MODEL / DEFAULT_OPENROUTER_MODEL are env-driven (see AI MODEL
# SELECTION above) — only the provider base URLs are true constants.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
# This is what tells the AI how to behave as FRIDAY
# Moving it to the server means you can update FRIDAY's personality
# without releasing a new version of the Android app
# IDENTITY: the assistant is FRIDAY — never ARIA, and never expanded into an
# acronym. If you change the name here, the whole product changes personality.
FRIDAY_SYSTEM_PROMPT = """
You are FRIDAY, a smart Android AI assistant.

CRITICAL RULES - NEVER break these:
1. ALWAYS respond with raw JSON only. No exceptions.
2. NEVER wrap your response in markdown, code blocks, or backticks.
3. Your ENTIRE response must be a single JSON object and nothing else.

═══ RESPONSE FORMATS ═══

For ACTIONS (when user wants to DO something):
{"type":"action","spoken":"What you say to the user","actions":[{"tool":"tool_name","value":"value_here"}]}

For CONVERSATION (questions, advice, information):
{"type":"chat","text":"your response here"}

For SECURITY ANALYSIS:
{"type":"security","verdict":"safe|warning|danger","text":"your analysis"}

═══ AVAILABLE TOOLS ═══

open_app       → value: app name (e.g. "youtube", "whatsapp", "camera")
search         → value: search query
call           → value: contact name or phone number
sms            → value: contact name
alarm          → value: time like "7:30am" or "14:00"
timer          → value: duration like "5 minutes" or "30 seconds"
settings       → value: "wifi", "bluetooth", "brightness", "volume", or "general"
open_url       → value: full URL

═══ EXAMPLES ═══

User: "Open YouTube"
{"type":"action","spoken":"Opening YouTube for you!","actions":[{"tool":"open_app","value":"youtube"}]}

User: "What is the capital of Ethiopia?"
{"type":"chat","text":"The capital of Ethiopia is Addis Ababa (አዲስ አበባ), which means New Flower in Amharic."}
""".strip()

# ─── REQUEST/RESPONSE MODELS ──────────────────────────────────────────────────
# Pydantic models define exactly what shape data must be in
# If the phone sends wrong data, FastAPI automatically rejects it
# Like a bouncer checking IDs at the door

class ChatRequest(BaseModel):
    """What the Android app sends to the server."""
    message: str                    # The user's text/voice input
    screen_context: Optional[str]   # Current screen content (for safety checks)
    model: Optional[str] = "gemini" # Which AI to use: "gemini" or "openrouter"
    # LEGACY (accepted, never required): older app builds send a provider-
    # specific OpenRouter model ID here. The backend OWNS model selection now —
    # this value is only used as a hint, and if the provider rejects it the
    # backend automatically retries with DEFAULT_OPENROUTER_MODEL. New clients
    # should omit it entirely.
    openrouter_model: Optional[str] = None
    history: Optional[list] = []    # Recent conversation turns
    token: str                      # Secret token - must match SECRET_TOKEN
    # Sent by the Android app; accepted for forward-compatibility.
    # user_id/store_memory are reserved for the deferred memory system (memory.py),
    # use_agent for future agent routing. Currently accepted but not acted upon.
    user_id: Optional[str] = None
    store_memory: Optional[bool] = False
    use_agent: Optional[bool] = False

class ChatResponse(BaseModel):
    """What the server sends back to the Android app."""
    response: str    # The AI's JSON response (same format as before)
    provider: str    # Which AI answered ("gemini"/"openai"/"claude"/"openrouter")
    latency_ms: int  # How long it took in milliseconds
    # New, OPTIONAL fields — old app builds ignore unknown JSON keys, so
    # adding these preserves Android compatibility.
    # fallback_from: set when the preferred provider failed and the chain
    # answered with another one (e.g. "gemini" when Claude had to step in).
    fallback_from: Optional[str] = None
    # Human-readable name of the provider that answered ("Gemini", "Claude").
    provider_label: Optional[str] = None
    # The exact model that produced this answer (backend-selected — the app
    # never needs to know or send provider-specific model IDs).
    model: Optional[str] = None

class HealthResponse(BaseModel):
    """Status check response."""
    status: str
    gemini_configured: bool
    openrouter_configured: bool
    version: str

# ─── AUTHENTICATION ───────────────────────────────────────────────────────────
# This checks every request to make sure it comes from our Android app
# It's like checking a password before letting someone in

def verify_token(request_token: str) -> bool:
    """
    Verify the secret token from the Android app.
    
    Why do we need this?
    Your server URL will be public (anyone can find it).
    Without a token, anyone could use your server and drain your API credits.
    With a token, only your Android app (which has the token hardcoded) can use it.
    """
    # CHANGE (security): previously this returned True when SECRET_TOKEN was
    # unset, leaving the API wide open on a public URL. Now it fails CLOSED —
    # if you forget to set SECRET_TOKEN on Railway, requests are rejected
    # instead of letting anyone drain your API credits.
    if not SECRET_TOKEN:
        logger.error("SECRET_TOKEN not configured — rejecting all requests. "
                     "Set SECRET_TOKEN in Railway environment variables.")
        return False
    return request_token == SECRET_TOKEN

# ─── AI PROVIDER FUNCTIONS ────────────────────────────────────────────────────

class ProviderFailure(Exception):
    """
    A transient provider failure the fallback chain should absorb:
    timeout, 429, 5xx, unreachable, or any upstream rejection (bad model,
    bad key). The chain walker catches this, puts the provider on cooldown,
    and tries the next provider — the user never has to press Send again.
    Only when EVERY provider in the chain fails does it surface as an HTTP
    error (via raise_provider_error).
    """

    def __init__(self, provider: str, upstream_status: int, message: str):
        super().__init__(message)
        self.provider = provider
        self.upstream_status = upstream_status
        self.message = message


def raise_provider_error(
    provider: str,
    upstream_status: int,
    message: str,
    attempts: Optional[list] = None,
):
    """
    Translate an upstream AI-provider HTTP status into the status WE return.

    WHY we never propagate provider status codes directly:
    Upstream codes describe the relationship between THIS SERVER and the
    provider, not between the Android app and this server. Propagating them
    verbatim caused a production incident: OpenRouter/Google returned 404 for
    a delisted model, the 404 fell into FastAPI's app-level 404 handler, and
    the Android app was told {"error":"Endpoint not found","path":"/chat"} —
    a model-rot problem disguised as a missing route.

    Translation table:
      429        → 429  (rate limiting is genuinely about request volume —
                         the app has a friendly retry message for it)
      500+       → 503  (provider outage: service temporarily unavailable —
                         the app retries, then falls back to its offline engine)
      everything
      else (4xx) → 502  (Bad Gateway: the provider rejected OUR upstream
                         request — bad model ID (404), bad server-side key
                         (401), malformed request (400). All of these are
                         server-configuration problems, never the app's fault)

    The body is structured JSON (FastAPI wraps it as {"detail": {...}}) so
    logs and clients can see exactly which provider failed and why, without
    ambiguity about whose status code they're looking at. This error is only
    raised after EVERY provider and model fallback has been exhausted, so
    `attempts` (when given) lists each provider+model tried and why it
    failed — the actual problem, not a generic "model configuration" guess.
    """
    if upstream_status == 429:
        ours = 429
    elif upstream_status >= 500:
        ours = 503
    else:
        ours = 502

    logger.error(
        f"Provider error | provider={provider} | "
        f"upstream_status={upstream_status} | returned={ours} | {message[:200]}"
    )
    detail = {
        "error": "all_providers_failed" if attempts else "upstream_provider_error",
        "provider": provider,
        "provider_status": upstream_status,
        "message": message[:300],
    }
    if attempts:
        detail["attempts"] = attempts
        tried = ", ".join(
            f"{a['provider']} ({a['model']}): {a['reason']}" for a in attempts
        )
        detail["message"] = (
            f"Every configured AI provider failed. Tried: {tried}"
        )[:600]
    raise HTTPException(status_code=ours, detail=detail)


async def call_gemini(
    message: str,
    history: list,
    screen_context: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Call Google's Gemini AI.

    `model` defaults to the env-configured GEMINI_MODEL; call_provider passes
    GEMINI_FALLBACK_MODEL on the model-level retry.

    async means this function can pause while waiting for the network
    and let other requests be handled in the meantime.
    Like a waiter who takes 5 orders before going to the kitchen,
    instead of waiting for one order to be cooked before taking the next.
    """
    if not GEMINI_API_KEY:
        raise ProviderFailure("gemini", 0, "Gemini API key not configured")
    model = model or GEMINI_MODEL

    # Build the message with optional screen context
    final_message = message
    if screen_context:
        final_message = f"{message}\n\nScreen context:\n{screen_context[:1200]}"

    # Build conversation history in Gemini format
    contents = []
    for turn in history[-20:]:  # Max 10 turns (20 = user+model pairs)
        contents.append(turn)

    # Add the new user message
    contents.append({
        "role": "user",
        "parts": [{"text": final_message}]
    })

    # Build the full request body
    body = {
        "systemInstruction": {
            "parts": [{"text": FRIDAY_SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 600,
            "responseMimeType": "application/json"
        }
    }

    # Make the HTTP request to Gemini
    # httpx is like the fetch() in JavaScript — it makes HTTP requests
    try:
        async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{GEMINI_BASE_URL}/{model}:generateContent?key={GEMINI_API_KEY}",
                json=body,
                headers={"Content-Type": "application/json"}
            )
    except (httpx.TimeoutException, httpx.TransportError) as e:
        # Timeout / unreachable — a transient class the chain absorbs.
        raise ProviderFailure("gemini", 0, f"network: {type(e).__name__}")

    if not response.is_success:
        # NEVER re-raise the upstream status verbatim — the chain walker
        # translates it (see raise_provider_error for the incident this
        # prevents) only after every provider has failed.
        error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        error_msg = error_data.get("error", {}).get("message", f"Error {response.status_code}")
        # Log the EXACT provider response body (truncated) for diagnosis.
        logger.error(
            f"Gemini raw response | model={model} | "
            f"status={response.status_code} | body={response.text[:500]}"
        )
        raise ProviderFailure("gemini", response.status_code, error_msg)

    # Parse the response
    data = response.json()
    raw_text = (
        data["candidates"][0]["content"]["parts"][0]["text"]
        .strip()
    )
    return strip_markdown(raw_text)


def to_openai_messages(
    message: str,
    history: list,
    screen_context: Optional[str] = None,
) -> list:
    """
    Convert FRIDAY's Gemini-format history + new message into the OpenAI
    message format shared by OpenAI, Claude (content shape), and OpenRouter.
    History travels with every request, so the conversation is preserved
    no matter which provider in the chain ends up answering.
    """
    final_message = message
    if screen_context:
        final_message = f"{message}\n\nScreen context:\n{screen_context[:1200]}"

    messages = []
    for turn in history[-20:]:
        # Convert from Gemini format (model) to OpenAI format (assistant)
        role = turn.get("role", "user")
        if role == "model":
            role = "assistant"
        content = ""
        parts = turn.get("parts", [])
        if parts:
            content = parts[0].get("text", "")
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": final_message})
    return messages


async def call_openai(
    message: str,
    history: list,
    screen_context: Optional[str] = None
) -> str:
    """
    Call OpenAI's Chat Completions API. Optional chain member — only used
    when OPENAI_API_KEY is configured. Same JSON-only contract as the
    other providers, so ActionRouter on Android parses it identically.
    """
    if not OPENAI_API_KEY:
        raise ProviderFailure("openai", 0, "OpenAI API key not configured")

    messages = [{"role": "system", "content": FRIDAY_SYSTEM_PROMPT}]
    messages += to_openai_messages(message, history, screen_context)

    try:
        async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "max_completion_tokens": 600,
                    "response_format": {"type": "json_object"},
                },
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise ProviderFailure("openai", 0, f"network: {type(e).__name__}")

    if not response.is_success:
        raise ProviderFailure("openai", response.status_code, response.text[:200])

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"].strip()
    return strip_markdown(raw_text)


async def call_claude(
    message: str,
    history: list,
    screen_context: Optional[str] = None
) -> str:
    """
    Call Anthropic's Claude via the official SDK (Messages API). Optional
    chain member — only used when ANTHROPIC_API_KEY is configured and the
    `anthropic` package is installed.
    """
    if not (ANTHROPIC_API_KEY and ANTHROPIC_SDK_AVAILABLE):
        raise ProviderFailure("claude", 0, "Claude not configured")

    # max_retries=0: retry/fallback policy belongs to OUR chain walker,
    # not the SDK. timeout keeps the whole chain under the app's 90s budget.
    client = AsyncAnthropic(
        api_key=ANTHROPIC_API_KEY,
        timeout=PROVIDER_TIMEOUT_SECONDS,
        max_retries=0,
    )
    try:
        response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
            system=FRIDAY_SYSTEM_PROMPT,
            messages=to_openai_messages(message, history, screen_context),
        )
    except anthropic.RateLimitError as e:
        raise ProviderFailure("claude", 429, str(e)[:200])
    except anthropic.APIStatusError as e:
        raise ProviderFailure("claude", e.status_code, str(e)[:200])
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        raise ProviderFailure("claude", 0, f"network: {type(e).__name__}")
    finally:
        await client.close()

    raw_text = next(
        (block.text for block in response.content if block.type == "text"), ""
    ).strip()
    if not raw_text:
        raise ProviderFailure("claude", 0, "empty response")
    return strip_markdown(raw_text)


async def call_openrouter(
    message: str,
    history: list,
    model: str,
    screen_context: Optional[str] = None
) -> str:
    """
    Call OpenRouter — a service that gives access to many AI models
    through one single API. Like a broker who can connect you to
    hundreds of AI models from one place.
    """
    if not OPENROUTER_API_KEY:
        raise ProviderFailure("openrouter", 0, "OpenRouter API key not configured")

    # OpenRouter uses OpenAI's message format (role: user/assistant/system)
    messages = [{"role": "system", "content": FRIDAY_SYSTEM_PROMPT}]
    messages += to_openai_messages(message, history, screen_context)

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 600,
        "response_format": {"type": "json_object"}
    }

    try:
        async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_SECONDS) as client:
            response = await client.post(
                OPENROUTER_URL,
                json=body,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    # CHANGE: previously hardcoded to an old domain; now env-configurable
                    "HTTP-Referer": APP_REFERER,
                    "X-Title": APP_TITLE
                }
            )
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise ProviderFailure("openrouter", 0, f"network: {type(e).__name__}")

    if not response.is_success:
        # NEVER re-raise the upstream status verbatim — the chain walker
        # translates it only after every provider has failed.
        # Log the EXACT provider response body so a model rejection is never
        # ambiguous in the logs again (this is what diagnosed the 2026-08-01
        # incident and its regression).
        logger.error(
            f"OpenRouter raw response | model={model} | "
            f"status={response.status_code} | body={response.text[:500]}"
        )
        raise ProviderFailure("openrouter", response.status_code, response.text[:200])

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"].strip()
    return strip_markdown(raw_text)


# ─── PROVIDER FALLBACK CHAIN WALKER ──────────────────────────────────────────

def model_for_provider(name: str, requested_openrouter_model: Optional[str]) -> str:
    """
    THE BACKEND OWNS MODEL SELECTION. The Android app only ever names a
    provider ("gemini"/"openrouter") — never a provider-specific model ID.
    Each provider resolves to its backend-configured production model:
      gemini     → GEMINI_MODEL           (env, fallback rolling alias)
      openai     → OPENAI_MODEL           (env)
      claude     → CLAUDE_MODEL           (env)
      openrouter → requested hint if a legacy client sent one, else
                   DEFAULT_OPENROUTER_MODEL (env). A rejected hint is
                   retried with the default automatically (see
                   call_provider) before the provider is marked failed.
    """
    return {
        "gemini": GEMINI_MODEL,
        "openai": OPENAI_MODEL,
        "claude": CLAUDE_MODEL,
        "openrouter": requested_openrouter_model or DEFAULT_OPENROUTER_MODEL,
    }.get(name, "")


async def call_provider(
    name: str,
    message: str,
    history: list,
    screen_context: Optional[str],
    openrouter_model: Optional[str],
) -> tuple:
    """
    Dispatch one chain member by name. Returns (text, final_model).

    Model-level fallback happens HERE, inside a single chain slot, before a
    provider is allowed to fail:
      gemini: if the env-configured GEMINI_MODEL is rejected with a 4xx
        (typo'd Railway variable, retired snapshot), the SAME provider is
        retried once with GEMINI_FALLBACK_MODEL (rolling alias, cannot rot).
      openrouter: if a legacy client sent a model hint and OpenRouter rejects
        it with a 4xx (delisted/invalid model), retried once with the
        backend's configured DEFAULT_OPENROUTER_MODEL. This makes stale model
        IDs baked into old app builds harmless.
    """
    if name == "gemini":
        try:
            return (
                await call_gemini(message, history, screen_context),
                GEMINI_MODEL,
            )
        except ProviderFailure as e:
            retriable = (
                GEMINI_MODEL != GEMINI_FALLBACK_MODEL
                and 400 <= e.upstream_status < 500
                and e.upstream_status != 429
            )
            if not retriable:
                raise
            logger.warning(
                f"Model fallback | provider=gemini | "
                f"rejected_model={GEMINI_MODEL} (upstream {e.upstream_status}) | "
                f"retrying with fallback={GEMINI_FALLBACK_MODEL}"
            )
            return (
                await call_gemini(
                    message, history, screen_context,
                    model=GEMINI_FALLBACK_MODEL,
                ),
                GEMINI_FALLBACK_MODEL,
            )
    if name == "openai":
        return await call_openai(message, history, screen_context), OPENAI_MODEL
    if name == "claude":
        return await call_claude(message, history, screen_context), CLAUDE_MODEL
    if name == "openrouter":
        model = openrouter_model or DEFAULT_OPENROUTER_MODEL
        try:
            return await call_openrouter(message, history, model, screen_context), model
        except ProviderFailure as e:
            retriable = (
                model != DEFAULT_OPENROUTER_MODEL
                and 400 <= e.upstream_status < 500
                and e.upstream_status != 429
            )
            if not retriable:
                raise
            logger.warning(
                f"Model fallback | provider=openrouter | "
                f"rejected_model={model} (upstream {e.upstream_status}) | "
                f"retrying with default={DEFAULT_OPENROUTER_MODEL}"
            )
            return (
                await call_openrouter(
                    message, history, DEFAULT_OPENROUTER_MODEL, screen_context
                ),
                DEFAULT_OPENROUTER_MODEL,
            )
    raise ProviderFailure(name, 0, f"unknown provider {name}")


async def call_with_fallback(
    preferred: str,
    message: str,
    history: list,
    screen_context: Optional[str],
    openrouter_model: Optional[str],
) -> tuple:
    """
    Walk the provider chain starting from [preferred], falling through to
    the remaining chain members on any transient failure (timeout, 429,
    5xx, unreachable, upstream rejection). Returns
    (ai_response, provider_used, final_model, fallback_from) where
    fallback_from is the provider the user asked for, or None if it
    answered directly.

    Recovery is automatic: failures only set a short cooldown (never a
    permanent flag), and every request re-evaluates the chain from the
    top — so the moment Gemini's cooldown lapses and it answers again,
    it is primary again. No restart, no user action.
    """
    # Preferred provider first, then the rest of the chain in order.
    ordered = [preferred] + [p for p in PROVIDER_CHAIN if p != preferred]
    candidates = [p for p in ordered if provider_configured(p)]
    if not candidates:
        raise_provider_error("none", 0, "No AI provider configured")

    now = time.monotonic()
    ready = [p for p in candidates if _provider_cooldown_until.get(p, 0) <= now]
    # If literally everything is cooling down, try them all anyway rather
    # than failing a request while a provider might have recovered.
    attempt_order = ready or candidates

    # SELECTION TRACE — one line answers "who was asked, who fell back, who
    # answered, with which model" for every request.
    logger.info(
        f"Provider selection | requested_provider={preferred} | "
        f"requested_model={model_for_provider(preferred, openrouter_model)} | "
        f"attempt_order={attempt_order}"
    )

    last_failure: Optional[ProviderFailure] = None
    attempts: list = []  # every provider+model tried, for the final error body
    for name in attempt_order:
        try:
            text, final_model = await call_provider(
                name, message, history, screen_context, openrouter_model
            )
            # Success clears the cooldown so this provider is instantly
            # trusted again (requirement: auto-switch back to Gemini).
            _provider_cooldown_until.pop(name, None)
            fallback_from = preferred if name != preferred else None
            logger.info(
                f"Provider selection | final_provider={name} | "
                f"final_model={final_model} | "
                f"fallback_from={fallback_from or 'none'}"
            )
            return text, name, final_model, fallback_from
        except ProviderFailure as e:
            last_failure = e
            attempts.append({
                "provider": e.provider,
                "model": model_for_provider(e.provider, openrouter_model),
                "upstream_status": e.upstream_status,
                "reason": e.message[:160],
            })
            _provider_cooldown_until[name] = (
                time.monotonic() + PROVIDER_COOLDOWN_SECONDS
            )
            logger.warning(
                f"Provider failed, trying next in chain | provider={e.provider} | "
                f"fallback_model={model_for_provider(e.provider, openrouter_model)} | "
                f"upstream_status={e.upstream_status} | {e.message[:200]}"
            )

    # Whole chain exhausted — now (and only now) surface a translated error
    # that names every provider+model tried and each upstream reason.
    raise_provider_error(
        last_failure.provider,
        last_failure.upstream_status,
        last_failure.message,
        attempts=attempts,
    )


def strip_markdown(text: str) -> str:
    """
    Remove markdown code fences if the AI wrapped the JSON in them.
    Some models do this even when told not to.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        end = text.rfind("```")
        if end >= 0:
            text = text[:end]
    return text.strip()


def build_error_response(message: str) -> str:
    """Build a safe JSON error response in FRIDAY's chat format."""
    safe = message.replace('"', "'")
    return json.dumps({"type": "chat", "text": safe})


def get_uptime_seconds() -> float:
    """Elapsed seconds since process start, from a monotonic clock."""
    return time.monotonic() - _START_MONOTONIC


def format_uptime(seconds: float) -> str:
    """Human-readable uptime like '1d 2h 3m 4s'."""
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


# ─── API ENDPOINTS ────────────────────────────────────────────────────────────
# These are the "rooms" in your server building
# Each endpoint is a URL your Android app can call

@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Like a "Are you there?" ping. Your monitoring system can call this
    every minute to make sure the server is alive.
    
    URL: GET https://your-server.com/
    """
    return HealthResponse(
        status="online",
        gemini_configured=bool(GEMINI_API_KEY),
        openrouter_configured=bool(OPENROUTER_API_KEY),
        version=APP_VERSION
    )


# ─── PRODUCTION DIAGNOSTIC ENDPOINTS ─────────────────────────────────────────
# Lightweight monitoring endpoints. None of these require authentication, call
# an AI provider, or reveal secrets — they only report configuration *status*.

@app.get("/ping")
async def ping():
    """
    Minimal connectivity test.

    No auth, no AI calls, no dependencies — just proof the server is reachable.
    Use this for the Android app's connection test and uptime monitors.
    """
    return {"status": "ok"}


@app.get("/health")
async def health():
    """
    Detailed backend health report (no secrets).

    Reports service identity, uptime, and which providers are configured.
    Keys themselves are never returned — only "configured"/"not_configured".
    """
    return {
        "status": "online",
        "service": SERVICE_NAME,
        "version": APP_VERSION,
        "uptime_seconds": round(get_uptime_seconds(), 1),
        "uptime_human": format_uptime(get_uptime_seconds()),
        "providers": {
            "gemini": "configured" if GEMINI_API_KEY else "not_configured",
            "openai": "configured" if provider_configured("openai") else "not_configured",
            "claude": "configured" if provider_configured("claude") else "not_configured",
            "openrouter": "configured" if OPENROUTER_API_KEY else "not_configured",
        },
        # Chain order actually in effect (configured providers only).
        "provider_chain": [p for p in PROVIDER_CHAIN if provider_configured(p)],
        "auth": "configured" if SECRET_TOKEN else "not_configured",
        # Railway sets RAILWAY_ENVIRONMENT (e.g. "production"); safe to expose.
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "local"),
    }


@app.get("/version")
async def version():
    """Backend version. Single source: APP_VERSION env var (default 1.0.0)."""
    return {"name": SERVICE_NAME, "version": APP_VERSION}


@app.get("/providers")
async def providers():
    """
    Which AI providers are configured (booleans only — never the keys).
    """
    return {
        "gemini": bool(GEMINI_API_KEY),
        "openai": provider_configured("openai"),
        "claude": provider_configured("claude"),
        "openrouter": bool(OPENROUTER_API_KEY),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. This is what your Android app calls.
    
    URL: POST https://your-server.com/chat
    Body: ChatRequest JSON
    Returns: ChatResponse JSON
    
    The flow:
    1. Verify the secret token
    2. Route to the right AI provider
    3. Get the AI response
    4. Return it to the Android app
    """
    # Step 1: Verify token
    if not verify_token(request.token):
        logger.warning("Unauthorized request — wrong token")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Step 2: Log the request (never log the full message for privacy)
    logger.info(
        f"Chat request | provider={request.model} | "
        f"msg_length={len(request.message)} | "
        f"has_screen={bool(request.screen_context)}"
    )

    start_time = time.time()

    # Step 3: Walk the provider chain. `request.model` picks the STARTING
    # provider (default "gemini" — FRIDAY is Gemini-first); on transient
    # failure the chain continues automatically (gemini → openai → claude
    # → openrouter, skipping unconfigured providers), carrying the full
    # conversation history so context survives the switch. Unknown values
    # (from older/newer apps) also start at the top of the chain instead
    # of failing — keeps Android compatibility as clients evolve.
    preferred = request.model if request.model in PROVIDER_CHAIN else PROVIDER_CHAIN[0]

    ai_response, provider, final_model, fallback_from = await call_with_fallback(
        preferred=preferred,
        message=request.message,
        history=request.history or [],
        screen_context=request.screen_context,
        openrouter_model=request.openrouter_model,
    )

    latency = int((time.time() - start_time) * 1000)
    if fallback_from:
        logger.info(
            f"Response sent | provider={provider} | model={final_model} | "
            f"(fallback from {fallback_from}) | latency={latency}ms"
        )
    else:
        logger.info(
            f"Response sent | provider={provider} | model={final_model} | "
            f"latency={latency}ms"
        )

    return ChatResponse(
        response=ai_response,
        provider=provider,
        latency_ms=latency,
        fallback_from=fallback_from,
        provider_label=PROVIDER_LABELS.get(provider, provider),
        model=final_model,
    )


@app.post("/analyze-security")
async def analyze_security(request: ChatRequest):
    """
    Dedicated security analysis endpoint.
    
    Sends screen content to AI specifically for security checking.
    Separated from /chat so we can apply different AI models/prompts
    for security tasks in the future.
    """
    if not verify_token(request.token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    security_prompt = f"""
Analyze this content for security threats, scams, and phishing attempts.
Respond ONLY with this exact JSON format:
{{"type":"security","verdict":"safe|warning|danger","text":"your analysis in 2-3 sentences"}}

Content to analyze:
{request.message}
""".strip()

    modified_request = ChatRequest(
        message=security_prompt,
        screen_context=request.screen_context,
        model=request.model,
        openrouter_model=request.openrouter_model,
        history=[],
        token=request.token
    )

    return await chat(modified_request)


# ─── ERROR HANDLERS ───────────────────────────────────────────────────────────
# What to do when things go wrong
# Like having a plan for when the fire alarm goes off

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def server_error(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# CHANGE (production upgrade): catch-all for *unhandled* exceptions. The 500
# handler above only fires for explicit HTTP 500s; this one catches anything
# that would otherwise leak a stack trace to the client. Full traceback is
# logged server-side; the client only ever sees a clean JSON body.
@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ─── STARTUP & SHUTDOWN ───────────────────────────────────────────────────────

def validate_configuration():
    """
    Validate environment configuration at startup and log clear warnings.

    Nothing here is fatal — the server always starts so Railway health checks
    pass, and the existing behavior handles the rest:
    - Missing SECRET_TOKEN  → auth fails CLOSED (verify_token rejects everything)
    - Missing provider keys → that provider returns a clean 500 when requested
    """
    problems = []
    if not SECRET_TOKEN:
        problems.append(
            "SECRET_TOKEN is not set — ALL /chat requests will be rejected (fail-closed). "
            "Set it in Railway's Variables tab."
        )
    configured = [p for p in PROVIDER_CHAIN if provider_configured(p)]
    if not configured:
        problems.append(
            "No AI provider configured (no provider API key set) — "
            "/chat will return errors for every request."
        )
    elif not provider_configured("gemini"):
        problems.append(
            "GEMINI_API_KEY not set — FRIDAY is Gemini-first, so every request "
            f"will start at the first fallback ({configured[0]})."
        )

    for p in problems:
        logger.warning(f"CONFIG: {p}")
    return problems


@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info(f"{SERVICE_NAME} started")
    logger.info(f"Version: {APP_VERSION}")
    logger.info(f"Host: 0.0.0.0 | PORT: {os.getenv('PORT', '8000')}")
    logger.info(f"Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'local')}")
    # Model selection is env-driven; log what was actually resolved so a bad
    # Railway variable is visible in the deploy logs immediately.
    logger.info(
        f"Gemini model: {GEMINI_MODEL} "
        f"({'env: GEMINI_MODEL' if os.getenv('GEMINI_MODEL') else 'fallback default'})"
    )
    logger.info(
        f"OpenRouter default model: {DEFAULT_OPENROUTER_MODEL} "
        f"({'env: DEFAULT_OPENROUTER_MODEL' if os.getenv('DEFAULT_OPENROUTER_MODEL') else 'fallback default'})"
    )
    if provider_configured("openai"):
        logger.info(f"OpenAI model: {OPENAI_MODEL}")
    if provider_configured("claude"):
        logger.info(f"Claude model: {CLAUDE_MODEL}")
    # Provider availability — booleans only, never key material or tokens.
    logger.info(
        "Provider availability | "
        + " | ".join(
            f"{p}={'available' if provider_configured(p) else 'NOT CONFIGURED'}"
            for p in PROVIDER_CHAIN
        )
    )
    logger.info(
        "Fallback chain (in effect): "
        + (" → ".join(p for p in PROVIDER_CHAIN if provider_configured(p)) or "NONE")
    )
    logger.info(f"Authentication configured: {bool(SECRET_TOKEN)}")
    validate_configuration()
    logger.info(f"Startup complete in {get_uptime_seconds():.2f}s")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown():
    logger.info("FRIDAY Backend shutting down")


# ─── RUN THE SERVER ───────────────────────────────────────────────────────────
# This runs the server when you execute `python main.py` directly.
# CHANGE: the port now comes from the PORT env var (Railway injects it),
# falling back to 8000 for local development. Host stays 0.0.0.0 so the
# server is reachable inside Railway's container network.
# In production Railway uses the Procfile instead of this block:
#   web: uvicorn main:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",                          # Listen on all interfaces
        port=int(os.environ.get("PORT", 8000)),  # Railway sets PORT; 8000 locally
        reload=os.getenv("RELOAD", "false").lower() == "true"  # dev-only hot reload
    )