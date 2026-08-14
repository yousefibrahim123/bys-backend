"""
User invites and account activation.

The "Add employee" screen used to call alert() and write to localStorage —
no account was created, no invite was sent, and the employee could never
sign in.

Now:
  POST /api/users/invite      creates a suspended account + activation token
  GET  /api/users/invite/{t}  validates the token before the password form
  POST /api/users/activate    activates the account and sets the password
  GET  /api/users             list users

Email: if SMTP is configured in .env it really sends. Without it the link is
returned in the response and written to the log — normal development
behaviour, and the point is that it's a working link, not a fake message.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..emailer import (invite_email, is_valid_email, send_email,
                       smtp_diagnosis)
from ..models import RefreshToken, User, UserInvite
from .. import permissions as perms
from ..schemas import (ActivateRequest, InviteCreate, InviteOut, InviteVerify,
                       PermissionsUpdate, TokenResponse, UserDetail, UserOut,
                       UserUpdate)
from ..security import (create_access_token, get_current_user, hash_password,
                        issue_refresh_token, require_roles)

log = logging.getLogger("users")
router = APIRouter(prefix="/api/users", tags=["users"])

INVITE_DAYS = 7


def _normalize(email: str) -> str:
    return (email or "").strip().lower()


@router.get("", response_model=List[UserDetail])
def list_users(db: Session = Depends(get_db),
               user: User = Depends(get_current_user)) -> List[UserDetail]:
    """
    Returns is_active and email_verified too — the screen needs them to show
    who is still pending activation, which it previously could not know.
    """
    stmt = select(User).order_by(User.created_at.desc())
    if user.role == "branch":
        stmt = stmt.where(User.branch == user.branch)
    out: List[UserDetail] = []
    for u in db.scalars(stmt):
        d = UserDetail.model_validate(u)
        d.permissions = sorted(perms.effective(db, u), key=perms.ALL.index)
        out.append(d)
    return out


@router.post("/invite", response_model=InviteOut, status_code=status.HTTP_201_CREATED)
def invite_user(payload: InviteCreate, db: Session = Depends(get_db),
                inviter: User = Depends(require_roles("admin", "branch", "supplier"))
                ) -> InviteOut:
    email = _normalize(payload.email)
    phone = (payload.phone or "").strip()

    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "That email is already registered")
    if phone and db.scalar(select(User).where(User.phone == phone)):
        raise HTTPException(status.HTTP_409_CONFLICT, "That mobile number is already registered")

    branch = payload.branch or inviter.branch
    if not is_valid_email(email):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Invalid email format")

    # Created suspended — can't sign in before activation
    invited = User(
        email=email, phone=phone, name=payload.name.strip(),
        role=payload.role, branch=branch,
        # An invited employee belongs to the same kind of organization as the
        # person inviting them — otherwise they land in the wrong sidebar.
        organization_type=("distributor" if payload.role == "supplier"
                           else (inviter.organization_type or "pharmacy")),
        password_hash=hash_password(secrets.token_urlsafe(32)),   # temporary and unguessable
        is_active=False,
        # Verified when they click the invite link — that link proves ownership
        email_verified=False,
    )
    db.add(invited)
    db.flush()

    token = secrets.token_urlsafe(32)
    db.add(UserInvite(
        user_id=invited.id, token=token, invited_by=inviter.id,
        job_title=payload.job_title, department=payload.department,
        expires_at=datetime.now(timezone.utc) + timedelta(days=INVITE_DAYS)))
    # Apply the wizard's checkboxes. Anything the inviter can't delegate is
    # dropped, so nobody can hand out authority they don't hold themselves.
    applied = perms.apply_grants(db, inviter, invited, payload.permissions)
    db.commit()
    db.refresh(invited)
    log.info("invited %s with %d permission(s)", email, len(applied))

    link = f"{settings.app_base_url.rstrip('/')}/activate?token={token}"
    subject, text, html = invite_email(
        invited.name, inviter.name, payload.role, branch, link, INVITE_DAYS)
    sent, reason = send_email(email, subject, text, html)
    if not sent:
        log.info("SMTP unavailable (%s) — activation link for %s: %s", reason, email, link)

    return InviteOut(
        user=UserOut.model_validate(invited),
        token=token,
        activation_url=link,
        expires_at=datetime.now(timezone.utc) + timedelta(days=INVITE_DAYS),
        email_sent=sent,
        note=(f"Invite sent to {email}." if sent else
              f"Email not sent ({reason}) — send this link to the employee manually."),
    )


@router.get("/invite/{token}", response_model=InviteVerify)
def verify_invite(token: str, db: Session = Depends(get_db)) -> InviteVerify:
    """Validate the token before showing the employee a password form."""
    inv = db.scalar(select(UserInvite).where(UserInvite.token == token))
    if not inv:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invalid activation link")
    if inv.used_at:
        raise HTTPException(status.HTTP_409_CONFLICT, "That link has already been used")
    if inv.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_410_GONE, "That link has expired — ask for a new invite")
    u = db.get(User, inv.user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return InviteVerify(email=u.email, name=u.name, role=u.role, branch=u.branch,
                        job_title=inv.job_title, expires_at=inv.expires_at)


@router.post("/activate", response_model=TokenResponse)
def activate(payload: ActivateRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Employee sets a password -> account activates and they're signed in."""
    inv = db.scalar(select(UserInvite).where(UserInvite.token == payload.token))
    if not inv:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invalid activation link")
    if inv.used_at:
        raise HTTPException(status.HTTP_409_CONFLICT, "That link has already been used")
    if inv.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_410_GONE, "That link has expired")

    if len(payload.password) < 8:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Password must be at least 8 characters")
    if not any(c.isalpha() for c in payload.password) or \
       not any(c.isdigit() for c in payload.password):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Password must contain a letter and a number")

    u = db.get(User, inv.user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    u.password_hash = hash_password(payload.password)
    u.is_active = True
    # Clicking the link sent to their inbox proves they own the address
    u.email_verified = True
    inv.used_at = datetime.now(timezone.utc)
    # An activated supplier must be selectable by pharmacies right away.
    from .auth import ensure_supplier_record
    ensure_supplier_record(db, u)
    db.commit()
    db.refresh(u)

    access, expires_in = create_access_token(u)
    return TokenResponse(access_token=access,
                         refresh_token=issue_refresh_token(db, u),
                         expires_in=expires_in, user=UserOut.model_validate(u))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_user(user_id: int, db: Session = Depends(get_db),
                admin: User = Depends(require_roles("admin"))) -> None:
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if u.id == admin.id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "You can't delete your own account")
    if u.role == "admin" and _admin_count(db) <= 1:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "This is the last admin — promote someone else first")
    db.query(UserInvite).filter(UserInvite.user_id == u.id).delete()
    db.query(RefreshToken).filter(RefreshToken.user_id == u.id).delete()
    db.delete(u)
    db.commit()


@router.post("/{user_id}/resend-invite", response_model=InviteOut)
def resend_invite(user_id: int, db: Session = Depends(get_db),
                  inviter: User = Depends(require_roles("admin", "branch", "supplier"))
                  ) -> InviteOut:
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if u.is_active:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "This account is already active")

    db.query(UserInvite).filter(UserInvite.user_id == u.id,
                               UserInvite.used_at.is_(None)).delete()
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=INVITE_DAYS)
    db.add(UserInvite(user_id=u.id, token=token, invited_by=inviter.id,
                      expires_at=expires))
    db.commit()

    link = f"{settings.app_base_url.rstrip('/')}/activate?token={token}"
    subject, text, html = invite_email(u.name, inviter.name, u.role, u.branch,
                                       link, INVITE_DAYS)
    sent, reason = send_email(u.email, subject, text, html)
    if not sent:
        log.info("SMTP unavailable (%s) — new link for %s: %s", reason, u.email, link)
    return InviteOut(user=UserOut.model_validate(u), token=token, activation_url=link,
                     expires_at=expires, email_sent=sent,
                     note=(f"Sent to {u.email}." if sent
                           else f"Email not sent ({reason}) — send the link manually."))


@router.get("/permissions/catalog")
def permission_catalog(target_role: str = "branch", db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)) -> dict:
    """
    What the signed-in user may grant to someone in `target_role`.

    The wizard used to render every permission for everyone. Now it renders
    only what this person actually holds, so the choices are real.
    """
    allowed = set(perms.grantable(db, user, target_role))
    return {
        "targetRole": target_role,
        "groups": [
            {"name": name,
             "permissions": [
                 {"key": k, "allowed": k in allowed,
                  "default": k in perms.defaults_for(target_role)}
                 for k in keys]}
            for name, keys in perms.CATALOG.items()
        ],
        "grantable": sorted(allowed, key=perms.ALL.index),
        "mine": sorted(perms.effective(db, user), key=perms.ALL.index),
    }


@router.get("/email-status")
def users_email_status(
        _: User = Depends(require_roles("admin", "branch", "supplier"))) -> dict:
    """Quick check from the user-management screen: will the invite actually
    arrive by email or not."""
    return smtp_diagnosis()


@router.get("/{user_id}", response_model=UserDetail)
def get_user(user_id: int, db: Session = Depends(get_db),
             viewer: User = Depends(get_current_user)) -> UserDetail:
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    # A branch user may only look at people in their own branch.
    if viewer.role == "branch" and u.branch != viewer.branch and u.id != viewer.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not visible from your branch")
    d = UserDetail.model_validate(u)
    d.permissions = sorted(perms.effective(db, u), key=perms.ALL.index)
    return d


@router.put("/{user_id}", response_model=UserDetail)
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db),
                editor: User = Depends(require_roles("admin", "branch", "supplier"))
                ) -> UserDetail:
    """
    Edit a teammate. The "Edit" button on the user-management screen had no
    handler because this endpoint did not exist.

    Guards: only an admin can grant the admin role or change roles at all, and
    nobody can suspend or demote themselves — otherwise the last admin could
    lock everyone out of the system.
    """
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if editor.role == "branch" and u.branch != editor.branch:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not visible from your branch")

    if payload.name is not None:
        name = payload.name.strip()
        if len(name) < 2:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "Name must be at least 2 characters")
        u.name = name

    if payload.phone is not None:
        phone = payload.phone.strip()
        if phone and db.scalar(select(User).where(User.phone == phone,
                                                  User.id != u.id)):
            raise HTTPException(status.HTTP_409_CONFLICT,
                                "That mobile number is already registered")
        u.phone = phone

    if payload.role is not None and payload.role != u.role:
        if editor.role != "admin":
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                "Only an admin can change roles")
        if u.id == editor.id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "You can't change your own role")
        if u.role == "admin" and _admin_count(db) <= 1:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "This is the last admin — promote someone else first")
        u.role = payload.role

    if payload.branch is not None:
        u.branch = payload.branch or None

    if payload.is_active is not None and payload.is_active != u.is_active:
        if u.id == editor.id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "You can't suspend your own account")
        if not payload.is_active and u.role == "admin" and _admin_count(db) <= 1:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "This is the last admin — promote someone else first")
        u.is_active = payload.is_active
        if not payload.is_active:
            # Suspending must actually end their sessions, not just flag them.
            db.query(RefreshToken).filter(RefreshToken.user_id == u.id).update(
                {RefreshToken.revoked: True})

    db.commit()
    db.refresh(u)
    return UserDetail.model_validate(u)


def _admin_count(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(User).where(
        User.role == "admin", User.is_active.is_(True))) or 0


@router.get("/{user_id}/permissions")
def get_permissions(user_id: int, db: Session = Depends(get_db),
                    viewer: User = Depends(get_current_user)) -> dict:
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if viewer.role == "branch" and u.branch != viewer.branch and u.id != viewer.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not visible from your branch")
    return {"userId": u.id, "role": u.role,
            "permissions": sorted(perms.effective(db, u), key=perms.ALL.index)}


@router.put("/{user_id}/permissions")
def set_permissions(user_id: int, payload: PermissionsUpdate,
                    db: Session = Depends(get_db),
                    editor: User = Depends(require_roles("admin", "branch"))) -> dict:
    """
    Replace a user's grants.

    `applied` may be shorter than what was sent: anything the editor can't
    delegate is dropped, and the response says which, so the UI can show it
    rather than silently pretending it worked.
    """
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if editor.role == "branch" and u.branch != editor.branch:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not visible from your branch")
    if u.id == editor.id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "You can't edit your own permissions")
    if u.role == "admin":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Admins hold every permission by definition")

    requested = list(payload.permissions)
    applied = perms.apply_grants(db, editor, u, requested)
    db.commit()
    return {"userId": u.id, "permissions": applied,
            "rejected": sorted(set(requested) - set(applied))}
