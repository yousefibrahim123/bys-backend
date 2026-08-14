"""
LLM client — every supported provider has a free tier.

Order in auto mode: groq -> gemini -> openrouter -> ollama -> local
`local` is a rule engine that needs no key and no internet, so the assistant
always works. Having a key improves the phrasing, not the numbers — every
figure comes from tools wired to the database and the models.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal, Any

import httpx

from ..config import settings

log = logging.getLogger("ai.llm")

FREE_OPTIONS = [
    {"provider": "groq", "how": "Free key from console.groq.com/keys — fastest, recommended",
     "env": "GROQ_API_KEY"},
    {"provider": "gemini", "how": "Free key from aistudio.google.com/apikey",
     "env": "GEMINI_API_KEY"},
    {"provider": "openrouter", "how": "Free key from openrouter.ai/keys (:free models)",
     "env": "OPENROUTER_API_KEY"},
    {"provider": "ollama", "how": "Fully local: ollama pull llama3.2 — no internet needed",
     "env": "OLLAMA_BASE_URL"},
]


@dataclass
class LLMResult:
    text: str
    provider: str
    ok: bool
    error: str = ""


def configured_providers() -> List[str]:
    out = []
    if settings.groq_api_key:
        out.append("groq")
    if settings.gemini_api_key:
        out.append("gemini")
    if settings.openrouter_api_key:
        out.append("openrouter")
    if settings.ollama_base_url:
        out.append("ollama")
    return out


def _provider_order() -> List[str]:
    p = (settings.ai_provider or "auto").lower()
    if p != "auto":
        return [p]
    order = []
    # Gemini first: it is the key most people actually set, and putting Groq
    # ahead of it meant a stale GROQ_API_KEY silently shadowed a working
    # Gemini key.
    if settings.gemini_api_key:
        order.append("gemini")
    if settings.groq_api_key:
        order.append("groq")
    if settings.openrouter_api_key:
        order.append("openrouter")
    # Ollama is only tried when no cloud key is present. Falling through to a
    # local model that has to load 4 GB from disk is what made the assistant
    # appear frozen for minutes on machines where Ollama was installed but
    # never really wanted.
    if not order:
        order.append("ollama")
    return order


# ------------------------------------------------------------------ providers

async def _call_openai_compatible(url: str, key: str, model: str,
                                  system: str, messages: List[dict],
                                  json_mode: bool) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=settings.ai_timeout_seconds) as c:
        r = await c.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


import asyncio

def _format_gemini_contents(messages: List[dict]) -> List[dict]:
    formatted: List[dict] = []
    for m in messages:
        role = "user" if m.get("role") == "user" else "model"
        content = m.get("content", "")
        if not content:
            continue
        if formatted and formatted[-1]["role"] == role:
            existing_text = formatted[-1]["parts"][0]["text"]
            formatted[-1]["parts"][0]["text"] = f"{existing_text}\n\n{content}"
        else:
            formatted.append({"role": role, "parts": [{"text": content}]})
    while formatted and formatted[0]["role"] != "user":
        formatted.pop(0)
    return formatted


async def _call_gemini(system: str, messages: List[dict], json_mode: bool) -> str:
    model = settings.gemini_model
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={settings.gemini_api_key}")
    contents = _format_gemini_contents(messages)
    if not contents:
        raise ValueError("Gemini requires at least one user message")

    payload: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
    }
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    max_retries = 3
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=settings.ai_timeout_seconds) as c:
                r = await c.post(url, json=payload)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    raise RuntimeError("Gemini returned no response candidates")
                first = candidates[0]
                content = first.get("content") or {}
                parts = content.get("parts") or []
                if not parts or not parts[0].get("text"):
                    finish_reason = first.get("finishReason", "UNKNOWN")
                    raise RuntimeError(f"Gemini response empty (finishReason={finish_reason})")
                return parts[0]["text"]
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1 and isinstance(e, (httpx.TimeoutException, httpx.NetworkError)):
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            break

    if last_err:
        raise last_err
    raise RuntimeError("Gemini request failed after retries")


# Cache of the resolved Ollama model name (so we don't hit /api/tags each time)
_ollama_resolved: Optional[str] = None

# Models that can't do chat — embeddings and rerankers
_NOT_CHAT = ("embed", "bge", "gte", "minilm", "e5-", "nomic-embed", "rerank")


async def ollama_models(timeout: float = 4.0) -> List[str]:
    """Names of models actually pulled on the user's machine."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(url)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]


async def resolve_ollama_model(force: bool = False) -> str:
    """
    Picks a model that actually exists instead of assuming llama3.2.

    This was the main reason Ollama appeared not to work: the name was
    hardcoded, so if the user had pulled a different model Ollama returned 404
    and the code silently fell back to the local engine — making it look like
    there was no AI at all.
    """
    global _ollama_resolved
    if _ollama_resolved and not force:
        return _ollama_resolved

    want = (settings.ollama_model or "").strip()
    try:
        available = await ollama_models()
    except Exception:
        _ollama_resolved = want or "llama3.2"
        return _ollama_resolved

    if not available:
        _ollama_resolved = want or "llama3.2"
        return _ollama_resolved

    # Exact match, else ignoring the tag (llama3.2 == llama3.2:latest)
    if want:
        if want in available:
            _ollama_resolved = want
            return want
        base = want.split(":")[0]
        for m in available:
            if m.split(":")[0] == base:
                _ollama_resolved = m
                log.info("Ollama: '%s' not found exactly — using '%s'", want, m)
                return m

    chat = [m for m in available
            if not any(k in m.lower() for k in _NOT_CHAT)] or available

    # Prefer a small model. Picking chat[0] blindly could land on a 35B MoE
    # that spills out of VRAM onto the CPU and takes minutes per reply — the
    # assistant then looks frozen even though nothing is broken.
    preferred = [x.strip() for x in settings.ollama_preferred.split(",") if x.strip()]
    for pref in preferred:
        for m in chat:
            if m == pref or m.split(":")[0] == pref.split(":")[0]:
                _ollama_resolved = m
                log.info("Ollama: '%s' not pulled — using '%s' (small and fast)",
                         want or "(none)", m)
                return m

    _ollama_resolved = min(chat, key=_size_hint)
    log.info("Ollama: '%s' not pulled. Available: %s — using '%s'. "
             "A large local model can take minutes per reply; "
             "`ollama pull llama3.2:1b` is far more responsive.",
             want or "(none)", ", ".join(available), _ollama_resolved)
    return _ollama_resolved


def _size_hint(name: str) -> float:
    """
    Rough parameter count parsed from the tag, so the smallest pulled model
    wins when none of the preferred names are present. Unknown -> treated as
    large, because guessing small on a huge model is the costly mistake.
    """
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", name.lower())
    return float(m.group(1)) if m else 999.0


async def _call_ollama(system: str, messages: List[dict], json_mode: bool) -> str:
    model = await resolve_ollama_model()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    payload["keep_alive"] = settings.ollama_keep_alive
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as c:
        r = await c.post(f"{settings.ollama_base_url.rstrip('/')}/api/chat", json=payload)
        if r.status_code >= 400:
            # Ollama puts the reason in the body — surface it instead of a bare status
            detail = ""
            try:
                detail = r.json().get("error", "")
            except Exception:
                detail = r.text[:200]
            raise RuntimeError(f"Ollama {r.status_code}: {detail}")
        data = r.json()
        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise RuntimeError(f"Ollama returned an empty response (done_reason={data.get('done_reason')})")
        return content


async def complete(system: str, messages: List[dict],
                   json_mode: bool = False) -> LLMResult:
    """Try providers in order. First success wins; if all fail -> local."""
    errors: List[str] = []
    for provider in _provider_order():
        try:
            if provider == "groq" and settings.groq_api_key:
                txt = await _call_openai_compatible(
                    "https://api.groq.com/openai/v1/chat/completions",
                    settings.groq_api_key, settings.groq_model, system, messages, json_mode)
            elif provider == "gemini" and settings.gemini_api_key:
                txt = await _call_gemini(system, messages, json_mode)
            elif provider == "openrouter" and settings.openrouter_api_key:
                txt = await _call_openai_compatible(
                    "https://openrouter.ai/api/v1/chat/completions",
                    settings.openrouter_api_key, settings.openrouter_model,
                    system, messages, json_mode)
            elif provider == "ollama":
                txt = await _call_ollama(system, messages, json_mode)
            else:
                continue
            if txt and txt.strip():
                return LLMResult(text=txt.strip(), provider=provider, ok=True)
            errors.append(f"{provider}: empty response")
        except Exception as e:
            # ReadTimeout and ConnectError stringify to empty — spell it out
            msg = str(e).strip() or type(e).__name__
            if isinstance(e, httpx.ReadTimeout):
                secs = (settings.ollama_timeout_seconds if provider == "ollama"
                        else settings.ai_timeout_seconds)
                msg = (f"Timed out after {secs:.0f}s. The local model may still be "
                       f"loading — raise OLLAMA_TIMEOUT_SECONDS or use a smaller model.")
            errors.append(f"{provider}: {msg}")
            # No local ollama is the normal case — not a warning
            level = logging.DEBUG if isinstance(e, httpx.ConnectError) else logging.WARNING
            log.log(level, "LLM provider %s failed: %s", provider, msg)
    return LLMResult(text="", provider="local", ok=False, error="; ".join(errors))


async def ollama_status() -> dict:
    """Ollama diagnostics: is it up, what is pulled, what will be used."""
    try:
        models = await ollama_models()
    except Exception as e:
        return {"reachable": False, "models": [], "selected": None,
                "hint": (f"No response from {settings.ollama_base_url} ({type(e).__name__}). "
                         "Start Ollama first: ollama serve")}
    if not models:
        return {"reachable": True, "models": [], "selected": None,
                "hint": "Ollama is running but no model is pulled. Try: ollama pull llama3.2"}
    selected = await resolve_ollama_model(force=True)
    return {"reachable": True, "models": models, "selected": selected,
            "hint": f"Will use '{selected}'. Set OLLAMA_MODEL in backend/.env to change it"}


_health_cache: Optional[dict] = None
_health_at: float = 0.0
HEALTH_TTL = 20.0   # seconds


def invalidate_health() -> None:
    """Invalidate the health cache — called when provider state may have changed."""
    global _health_cache, _health_at
    _health_cache, _health_at = None, 0.0


async def health(force: bool = False) -> dict:
    """
    Provider health check.

    Important: a full generation here was unworkable — a local model takes
    minutes on first load, so every /api/ai/status call hung until timeout.
    Ollama is now probed via /api/tags (millisecond response) and the result
    is cached.
    """
    global _health_cache, _health_at
    now = time.monotonic()
    if _health_cache and not force and (now - _health_at) < HEALTH_TTL:
        return _health_cache

    order = _provider_order()
    result: Optional[dict] = None

    # Ollama: lightweight probe instead of a full generation
    if "ollama" in order and not (settings.groq_api_key or settings.gemini_api_key
                                  or settings.openrouter_api_key):
        try:
            models = await ollama_models(timeout=4.0)
            if models:
                selected = await resolve_ollama_model()
                result = {"llm_available": True, "provider": "ollama",
                          "model": selected,
                          "reason": f"Ollama is up with {len(models)} model(s) pulled"}
            else:
                result = {"llm_available": False, "provider": "local",
                          "model": "rule-engine",
                          "reason": "Ollama is up but no model is pulled — try: ollama pull llama3.2"}
        except Exception:
            result = None   # not up — fall through to the normal check

    if result is None:
        res = await complete("Reply with the single word: ok",
                             [{"role": "user", "content": "ping"}])
        if res.ok:
            model = {"groq": settings.groq_model, "gemini": settings.gemini_model,
                     "openrouter": settings.openrouter_model,
                     "ollama": _ollama_resolved or settings.ollama_model}.get(res.provider, "")
            result = {"llm_available": True, "provider": res.provider, "model": model,
                      "reason": "Provider responding normally"}
        else:
            result = {
                "llm_available": False, "provider": "local", "model": "rule-engine",
                "reason": (res.error or "No provider configured")
                          + " — the assistant is running on the local engine "
                            "(all figures are still real).",
            }

    _health_cache, _health_at = result, now
    return result


def extract_json(text: str) -> Optional[dict]:
    """Some models wrap JSON in ```json fences — strip them."""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
