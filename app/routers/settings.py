"""
User settings, persisted in the database.

The Settings screen used to write everything to localStorage — "saved"
settings vanished on another machine or after clearing site data. Values now
live in `app_settings`, keyed by section, and every read reflects the latest
write immediately.

  GET /api/settings          -> {"organization": {...}, "notifications": {...}}
  PUT /api/settings          -> partial upsert, returns the merged result
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import json
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import AppSetting, User
from ..schemas import SettingsUpdate
from ..security import get_current_user

log = logging.getLogger("settings")
router = APIRouter(prefix="/api/settings", tags=["settings"])

# Known sections. Unknown keys are still stored (forward-compatible), but the
# payload has to be JSON-serialisable.
KNOWN_KEYS = {"organization", "notifications", "branch_info", "preferences"}


def _all_for(db: Session, user: User) -> dict:
    out: dict = {}
    for row in db.scalars(select(AppSetting).where(AppSetting.user_id == user.id)):
        try:
            out[row.key] = json.loads(row.value)
        except (TypeError, ValueError):
            out[row.key] = {}
    return out


@router.get("")
def get_settings(db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> dict:
    return {"settings": _all_for(db, user)}


@router.put("")
def put_settings(payload: SettingsUpdate, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> dict:
    """
    Upsert the sections that were sent; everything else is untouched.
    Returns the full merged settings so the UI can refresh in one step.
    """
    for key, value in (payload.settings or {}).items():
        key = str(key)[:60]
        row = db.scalar(select(AppSetting).where(AppSetting.user_id == user.id,
                                                 AppSetting.key == key))
        encoded = json.dumps(value if value is not None else {})
        if row:
            row.value = encoded
        else:
            db.add(AppSetting(user_id=user.id, key=key, value=encoded))
    db.commit()
    log.info("settings saved for %s: %s", user.email,
             ", ".join(payload.settings.keys()) or "(nothing)")
    return {"settings": _all_for(db, user)}
