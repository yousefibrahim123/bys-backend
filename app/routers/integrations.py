"""
POS / external-system integrations.

The old Integrations screen simulated everything in the browser — a fake
progress bar and a Math.random() "Connection Successful". Connections are now
stored per branch, and "Test Connection" opens a real TCP socket to the
configured host/port. The recorded status and the sync log reflect exactly
what happened — nothing is invented.

  GET    /api/integrations                 connections for the caller's branch
  POST   /api/integrations                 register a connection
  POST   /api/integrations/{id}/test       real reachability check + log entry
  DELETE /api/integrations/{id}            remove a connection
  GET    /api/integrations/logs            real activity log (?connectionId=)
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import logging
import socket
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from .. import permissions as perms
from ..models import PosConnection, SyncLog, User
from ..schemas import PosConnectionIn, PosConnectionOut, SyncLogOut
from ..security import get_current_user, resolve_branch

log = logging.getLogger("integrations")
router = APIRouter(prefix="/api/integrations", tags=["integrations"])

TEST_TIMEOUT_SECONDS = 4.0


def _require(db: Session, user: User, permission: str) -> None:
    if not perms.has(db, user, permission):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            f"You need the '{permission}' permission for this")


def _out(c: PosConnection) -> PosConnectionOut:
    return PosConnectionOut.model_validate(c)


@router.get("", response_model=List[PosConnectionOut])
def list_connections(branch: Optional[str] = None, db: Session = Depends(get_db),
                     user: User = Depends(get_current_user)) -> List[PosConnectionOut]:
    _require(db, user, "view_integrations")
    b = resolve_branch(user, branch)
    rows = db.scalars(select(PosConnection).where(PosConnection.branch == b)
                      .order_by(PosConnection.created_at.desc()))
    return [_out(c) for c in rows]


@router.post("", response_model=PosConnectionOut,
             status_code=status.HTTP_201_CREATED)
def create_connection(payload: PosConnectionIn, db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)) -> PosConnectionOut:
    _require(db, user, "connect_system")
    b = resolve_branch(user, payload.branch)
    if not b:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "A branch is required for a POS connection")
    name = payload.name.strip()
    host = payload.host.strip()
    if len(name) < 2:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Connection name must be at least 2 characters")
    if not host:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Host is required")
    if not 1 <= payload.port <= 65535:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Port must be between 1 and 65535")
    c = PosConnection(branch=b, name=name, system_type=payload.system_type.strip(),
                      host=host, port=payload.port,
                      database_name=payload.database_name.strip(),
                      status="unchecked", created_by=user.id)
    db.add(c)
    db.commit()
    db.refresh(c)
    log.info("POS connection #%d registered for %s (%s:%d)", c.id, b, host, payload.port)
    return _out(c)


@router.post("/{connection_id}/test", response_model=PosConnectionOut)
def test_connection(connection_id: int, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)) -> PosConnectionOut:
    """
    Real reachability check: attempts a TCP connection to host:port. The
    result — success or the actual error — is stored and logged. This never
    fakes a success.
    """
    _require(db, user, "run_manual_sync")
    c = db.get(PosConnection, connection_id)
    if not c:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Connection not found")
    resolve_branch(user, c.branch)

    started = time.monotonic()
    ok, detail = False, ""
    try:
        with socket.create_connection((c.host, c.port), timeout=TEST_TIMEOUT_SECONDS):
            ok = True
            detail = f"TCP connection to {c.host}:{c.port} succeeded"
    except OSError as e:
        detail = f"Could not reach {c.host}:{c.port} — {e}"
    duration = int((time.monotonic() - started) * 1000)

    c.status = "connected" if ok else "failed"
    c.last_checked_at = datetime.now(timezone.utc)
    c.last_error = "" if ok else detail
    db.add(SyncLog(connection_id=c.id, branch=c.branch, action="test",
                   ok=ok, detail=detail, duration_ms=duration))
    db.commit()
    db.refresh(c)
    log.info("POS test #%d: %s (%d ms)", c.id, "ok" if ok else detail, duration)
    return _out(c)


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT,
               response_model=None)
def delete_connection(connection_id: int, db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)) -> None:
    _require(db, user, "disconnect_system")
    c = db.get(PosConnection, connection_id)
    if not c:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Connection not found")
    resolve_branch(user, c.branch)
    db.query(SyncLog).filter(SyncLog.connection_id == c.id).delete()
    db.delete(c)
    db.commit()


@router.get("/logs", response_model=List[SyncLogOut])
def list_logs(connection_id: Optional[int] = Query(None), branch: Optional[str] = None,
              limit: int = Query(100, ge=1, le=500),
              db: Session = Depends(get_db),
              user: User = Depends(get_current_user)) -> List[SyncLogOut]:
    _require(db, user, "view_integrations")
    b = resolve_branch(user, branch)
    stmt = select(SyncLog).where(SyncLog.branch == b)
    if connection_id:
        stmt = stmt.where(SyncLog.connection_id == connection_id)
    stmt = stmt.order_by(SyncLog.created_at.desc()).limit(limit)
    return [SyncLogOut.model_validate(r) for r in db.scalars(stmt)]
