"""Authentication: JWT, password hashing, and role-based access."""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from .models import RefreshToken, User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def hash_password(p: str) -> str:
    # bcrypt truncates at 72 bytes — do it ourselves so it doesn't raise
    return pwd_context.hash(p[:72])


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain[:72], hashed)
    except Exception:
        return False


def create_access_token(user: User) -> Tuple[str, int]:
    expires_in = settings.access_token_minutes * 60
    payload = {
        "sub": str(user.id), "email": user.email, "role": user.role,
        "branch": user.branch, "name": user.name,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm), expires_in


def issue_refresh_token(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(48)
    db.add(RefreshToken(
        user_id=user.id, token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_days)))
    db.commit()
    return token


def _unauthorized(detail: str = "Credentials are invalid or expired") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail,
                         headers={"WWW-Authenticate": "Bearer"})


def get_current_user(token: Optional[str] = Depends(oauth2),
                     db: Session = Depends(get_db)) -> User:
    if not token:
        raise _unauthorized("Sign-in required")
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        uid = int(payload.get("sub", 0))
    except (JWTError, ValueError):
        raise _unauthorized()
    user = db.get(User, uid)
    if not user or not user.is_active:
        raise _unauthorized("User not found or suspended")
    return user


def get_optional_user(token: Optional[str] = Depends(oauth2),
                      db: Session = Depends(get_db)) -> Optional[User]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return db.get(User, int(payload.get("sub", 0)))
    except (JWTError, ValueError):
        return None


def require_roles(*roles: str):
    def dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action is restricted to roles: {', '.join(roles)}")
        return user
    return dep


def _first_branch_name() -> str:
    """First real branch in the database — used only as the admin fallback.
    The old fallback was a hardcoded 'Giza', which silently filed data under a
    branch that may not exist at all."""
    from sqlalchemy import select

    from .database import SessionLocal
    from .models import Branch
    with SessionLocal() as s:
        name = s.scalar(select(Branch.name).order_by(Branch.id))
        return name or ""


def resolve_branch(user: User, requested: Optional[str]) -> str:
    """
    The branch this user is allowed to see.
    - admin/supplier: any branch (or the first real branch if unspecified)
    - branch: only their own; a request for another branch is rejected
    """
    if user.role in ("admin", "supplier"):
        return requested or user.branch or _first_branch_name()
    own = user.branch or _first_branch_name()
    if requested and requested != own:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"You only have access to branch {own}")
    return own
