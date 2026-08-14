"""AI assistant endpoints."""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ai import llm
from ..ai.assistant import answer
from ..ai.tools import TOOL_SPECS
from ..config import settings
from ..database import get_db
from ..models import ChatMessage, User
from ..schemas import AIStatus, ChatChart, ChatRequest, ChatResponse
from ..security import get_current_user, resolve_branch

router = APIRouter(prefix="/api/ai", tags=["ai"])

MAX_HISTORY = 6


@router.get("/status", response_model=AIStatus)
async def ai_status(refresh: bool = False) -> AIStatus:
    """
    Reports which provider is live and what free options exist.
    With no key it returns llm_available=false and the assistant keeps working
    on the local engine.
    """
    h = await llm.health(force=refresh)
    return AIStatus(
        provider=h["provider"], model=h["model"], llm_available=h["llm_available"],
        reason=h["reason"], configured_providers=llm.configured_providers(),
        free_options=llm.FREE_OPTIONS)


@router.get("/ollama")
async def ollama_diagnostic() -> dict:
    """
    Ollama diagnostics: is it up, which models are pulled, which will be used.
    Open without a token so it can be checked straight from the browser.
    """
    st = await llm.ollama_status()
    llm.invalidate_health()   # state may have changed — don't serve a stale cached result
    return {
        "baseUrl": settings.ollama_base_url,
        "configuredModel": settings.ollama_model,
        "reachable": st["reachable"],
        "installedModels": st["models"],
        "selectedModel": st["selected"],
        "hint": st["hint"],
    }


@router.get("/tools")
def list_tools(user: User = Depends(get_current_user)) -> dict:
    """Tools the assistant can use — useful for docs and diagnostics."""
    return {"tools": TOOL_SPECS, "max_per_turn": 3}


def _load_history(db: Session, user_id: int,
                  conversation_id: Optional[str]) -> List[dict]:
    """
    Recent turns, in order, always starting on a user message.

    Two bugs lived here.

    1. Ordering was `created_at.desc()` with no tie-breaker. The user row and
       the assistant row are inserted in the same commit, so their timestamps
       can be identical — and then their relative order was arbitrary. A
       flipped pair makes the model answer the previous question again.

    2. The limit counted *rows*, not turns, and took the newest N before
       reversing. That could slice mid-turn and hand the model a history
       starting with an assistant message whose question had been cut off,
       so it would restate that answer and keep appending to it.

    Now: order by (created_at, id) for a stable sequence, take whole turns,
    and drop any leading assistant row.
    """
    if not conversation_id:
        return []

    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id,
               ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(MAX_HISTORY * 2)).all()
    rows = list(reversed(rows))

    while rows and rows[0].role != "user":
        rows.pop(0)
    while rows and rows[-1].role == "user":
        rows.pop()

    return [{"role": r.role, "content": r.content} for r in rows[-(MAX_HISTORY * 2):]]


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: Session = Depends(get_db),
               user: User = Depends(get_current_user)) -> ChatResponse:
    msg = payload.message.strip()
    if not msg:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Message is empty")
    if len(msg) > 2000:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Message too long (2000 character limit)")

    branch = resolve_branch(user, payload.branch)

    history = _load_history(db, user.id, payload.conversation_id)

    result = await answer(db, msg, branch, history, payload.conversation_id)

    db.add(ChatMessage(user_id=user.id, conversation_id=result["conversation_id"],
                       role="user", content=msg))
    db.add(ChatMessage(user_id=user.id, conversation_id=result["conversation_id"],
                       role="assistant", content=result["reply"],
                       meta=json.dumps({"tools": result["tools_used"],
                                        "chart": result["chart"]}, ensure_ascii=False)))
    db.commit()

    return ChatResponse(
        conversation_id=result["conversation_id"], reply=result["reply"],
        chart=ChatChart(**result["chart"]) if result["chart"] else None,
        tools_used=result["tools_used"], provider=result["provider"],
        grounded=result["grounded"])


@router.get("/history")
def history(conversation_id: str = Query(...), limit: int = Query(50, ge=1, le=200),
            db: Session = Depends(get_db),
            user: User = Depends(get_current_user)) -> dict:
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id,
               ChatMessage.user_id == user.id)
        .order_by(ChatMessage.created_at).limit(limit)).all()
    out = []
    for r in rows:
        item = {"role": r.role, "content": r.content,
                "createdAt": r.created_at.isoformat()}
        if r.meta:
            try:
                item.update(json.loads(r.meta))
            except json.JSONDecodeError:
                pass
        out.append(item)
    return {"conversationId": conversation_id, "messages": out}


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def clear_history(conversation_id: str = Query(...), db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> None:
    for r in db.scalars(select(ChatMessage).where(
            ChatMessage.conversation_id == conversation_id,
            ChatMessage.user_id == user.id)):
        db.delete(r)
    db.commit()


@router.get("/suggestions")
def suggestions(user: User = Depends(get_current_user)) -> dict:
    """Suggested prompts — shown in the UI before the first message."""
    return {"suggestions": [
        "Which products will run out this week?",
        "What is my most profitable product?",
        "Which products are near expiry?",
        "Recommend a reorder list for tomorrow",
        "Summarize inventory health",
        "Which supplier delivers most reliably?",
        "How much revenue did we make in the last 30 days?",
        "Forecast demand for Panadol Extra",
    ]}
