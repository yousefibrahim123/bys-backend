"""Health check."""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..ml.registry import registry
from ..models import Product, User

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    try:
        db.execute(text("SELECT 1"))
        db_ok, db_err = True, None
    except Exception as e:
        db_ok, db_err = False, str(e)

    ml = registry.status()
    return {
        "status": "healthy" if db_ok else "degraded",
        "version": settings.version,
        "database": {"connected": db_ok, "error": db_err,
                     "products": db.scalar(select(func.count(Product.id))) if db_ok else 0,
                     "users": db.scalar(select(func.count(User.id))) if db_ok else 0},
        "models": {k: v.get("available", False) for k, v in ml.items()},
    }


@router.get("/api/health")
def health_api(db: Session = Depends(get_db)) -> dict:
    return health(db)


@router.get("/api/diagnostics/config")
def config_diagnostics() -> dict:
    """
    One place to answer "is my .env actually being read?" without guessing.

    This exists because the previous failure mode was invisible: env_file was
    a relative path, so starting the server from the repo root silently
    ignored the whole file and every symptom looked like a different bug —
    "email not configured", "the AI is stuck on Ollama", "my key doesn't
    work". Secrets are never echoed back, only whether they are present and
    how long they are.
    """
    from pathlib import Path

    from ..config import settings
    from ..emailer import smtp_configured

    env_path = Path(str(settings.model_config.get("env_file", "")))
    pw = settings.smtp_password or ""

    return {
        "envFile": {
            "path": str(env_path),
            "exists": env_path.exists(),
        },
        "smtp": {
            "configured": smtp_configured(),
            "host": settings.smtp_host or None,
            "port": settings.smtp_port,
            "user": settings.smtp_user or None,
            "from": settings.smtp_from or settings.smtp_user or None,
            "passwordLength": len(pw),
            # A Gmail App Password is exactly 16 characters once the display
            # spaces are stripped. Any other length is the usual cause of a
            # 535 rejection.
            "passwordLooksLikeAppPassword": len(pw) == 16,
            "startTls": settings.smtp_starttls,
            "ssl": settings.smtp_ssl,
        },
        "ai": {
            "provider": settings.ai_provider,
            "geminiKeySet": bool(settings.gemini_api_key),
            "geminiModel": settings.gemini_model,
            "groqKeySet": bool(settings.groq_api_key),
            "openrouterKeySet": bool(settings.openrouter_api_key),
        },
        "appBaseUrl": settings.app_base_url,
    }
