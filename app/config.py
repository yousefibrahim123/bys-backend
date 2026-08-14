"""App settings — all from environment variables, with working defaults."""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
ML_ASSETS = BASE_DIR / "ml_assets"


class Settings(BaseSettings):
    app_name: str = "Pharmacy Smart Supply Chain API"
    version: str = "1.0.0"

    database_url: str = f"sqlite:///{BASE_DIR / 'pharma.db'}"

    jwt_secret: str = os.getenv("JWT_SECRET", "dev-secret-change-me-in-production")
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60 * 12
    refresh_token_days: int = 30

    cors_origins: str = "*"
    # Base URL used to build links in verification and invite emails
    app_base_url: str = os.getenv("APP_BASE_URL", "http://localhost:5173")

    # ---- SMTP is optional. Without it the activation link is returned in the
    # response body and written to the log instead of being emailed.
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "")
    smtp_starttls: bool = os.getenv("SMTP_STARTTLS", "true").lower() == "true"
    smtp_ssl: bool = os.getenv("SMTP_SSL", "false").lower() == "true"

    # ---- AI: entirely optional. With no key at all it runs on the local engine.
    ai_provider: str = os.getenv("AI_PROVIDER", "auto")   # auto|groq|gemini|openrouter|ollama|local
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    ai_timeout_seconds: float = float(os.getenv("AI_TIMEOUT_SECONDS", "45"))
    # Local models can take minutes to load the first time, so the 45s timeout
    # used for cloud APIs is far too short for them.
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
    # Hard ceiling for one chat turn. If the model is still generating when
    # this expires we return the locally composed answer instead of leaving
    # the UI spinning. A 35B model on 8 GB of VRAM spills to CPU and needs
    # minutes per reply, which is what made the assistant look frozen.
    ai_reply_budget_seconds: float = float(os.getenv("AI_REPLY_BUDGET_SECONDS", "60"))
    # Ask the LLM to plan tools as well as write the answer. Off by default:
    # it doubles latency, and the keyword planner already routes correctly.
    ai_llm_planner: bool = os.getenv("AI_LLM_PLANNER", "false").lower() == "true"
    # Preferred local models, smallest first. Used when OLLAMA_MODEL is not
    # pulled, so we pick something that actually runs at a usable speed.
    ollama_preferred: str = os.getenv(
        "OLLAMA_PREFERRED",
        "llama3.2:1b,qwen2.5:1.5b,gemma2:2b,llama3.2:3b,phi3.5,qwen2.5:3b,llama3.2")
    # Keeps the model resident in RAM between requests instead of reloading it
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

    # ---- Returning the verification link in the API response is a
    # development convenience that completely defeats email verification:
    # anyone can register any address and immediately verify it, because the
    # proof-of-inbox-ownership is handed straight back to the caller.
    # OFF by default — opt in explicitly while setting SMTP up.
    dev_expose_links: bool = os.getenv("DEV_EXPOSE_LINKS", "false").lower() == "true"

    # ---- Demo/sample data. OFF by default: a production database starts
    # empty (plus the predefined test admin) and everything on screen comes
    # from records users actually create.
    seed_demo_data: bool = os.getenv("SEED_DEMO_DATA", "false").lower() == "true"

    # ---- Gemini is now the default cloud provider. Ollama stays as a manual
    # opt-in (AI_PROVIDER=ollama) — it is no longer tried automatically.
    gemini_only: bool = os.getenv("GEMINI_ONLY", "false").lower() == "true"

    @field_validator("smtp_password", "smtp_user", "smtp_from", "smtp_host",
                     mode="before")
    @classmethod
    def _strip_smtp(cls, v):
        """
        Gmail shows App Passwords as 'abcd efgh ijkl mnop'. Pasted verbatim the
        spaces travel into the SMTP AUTH string and Gmail answers 535, which
        looked exactly like 'SMTP not configured'. Strip them here, once.
        """
        if isinstance(v, str):
            return v.replace('"', "").replace("'", "").strip().replace(" ", "")
        return v

    # env_file was relative, so it only resolved when uvicorn happened to be
    # started from inside backend/. Launched from the repo root the whole file
    # was silently ignored and every SMTP/AI key came back empty.
    model_config = {"env_file": str(BASE_DIR / ".env"), "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
