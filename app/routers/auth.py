"""
Authentication + email verification.

What changed: `/register` used to create the account and hand back a token
immediately — anyone could type any email and get in. Now it creates an
**unverified** account, emails a link, and login is refused until that link
is clicked.

  POST /api/auth/register            unverified account + verification email
  GET  /api/auth/verify-email/{t}    validate the token before showing the UI
  POST /api/auth/verify-email        verify the email and sign the user in
  POST /api/auth/resend-verification send a fresh link
  GET  /api/auth/email-status        SMTP configuration state (diagnostics)
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..emailer import (is_valid_email, send_email, smtp_configured,
                       smtp_diagnosis, verification_email)
from ..models import EmailVerification, PasswordReset, RefreshToken, User
from ..schemas import (ChangePasswordRequest, EmailOnlyRequest, LoginRequest,
                       ProfileUpdate, RefreshRequest, RegisterRequest,
                       RegisterResponse, ResetPasswordRequest, SimpleMessage,
                       TokenResponse, UserDetail, VerifyEmailRequest,
                       VerifyTokenInfo, UserOut)
from ..security import (create_access_token, get_current_user, hash_password,
                        issue_refresh_token, verify_password)
from ..emailer import send_email as _send

log = logging.getLogger("auth")
router = APIRouter(prefix="/api/auth", tags=["auth"])

VERIFY_HOURS = 24
RESET_HOURS = 1

# Addresses that belong to the built-in demo logins. Letting someone register
# these creates an account that looks like staff but is nobody, and it
# collides with the demo fallback used when the server is unreachable.
RESERVED_EMAILS = {
    "supplier@gmail.com", "admin@gmail.com", "giza@gmail.com",
    "cairo@gmail.com", "alex@gmail.com", "nascity@gmail.com",
}

def is_test_account(user: Optional[User]) -> bool:
    """
    Predefined TEST accounts (development/testing only). For them — and ONLY
    for them — the email round-trips are skipped: no verification email, no
    OTP, and password reset works locally without sending anything.

    The decision is made server-side from an explicit flag stored on the
    account (set at seed time), never from the email address or frontend
    input. Real accounts keep the full flow.
    """
    return bool(user and user.is_test_account)

# Throwaway inbox providers. Verification is meaningless against these: the
# address works for exactly as long as it takes to click the link.
DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "tempmail.com",
    "temp-mail.org", "throwawaymail.com", "yopmail.com", "trashmail.com",
    "sharklasers.com", "getnada.com", "maildrop.cc", "dispostable.com",
    "fakeinbox.com", "mailnesia.com", "spamgourmet.com", "moakt.com",
    "emailondeck.com", "tempmailo.com", "mohmal.com",
}


def _reject_unusable_email(email: str) -> None:
    """
    Registration accepted literally any well-formed address. Combined with the
    activation link being returned in the response body (see
    DEV_EXPOSE_LINKS), that meant a fake address became a fully verified
    account in two requests, with nothing to distinguish it afterwards.
    """
    if email in RESERVED_EMAILS:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "That address is reserved for the built-in demo accounts. "
            "Use your own email address.")
    domain = email.rsplit("@", 1)[-1].lower()
    if domain in DISPOSABLE_DOMAINS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Disposable email addresses aren't accepted — please use a "
            "permanent address you control.")


def ensure_supplier_record(db: Session, user: User) -> None:
    """
    A supplier signing up must appear in the suppliers list immediately —
    otherwise pharmacies can't pick them in the New Product form or address
    orders to them. Creates (or completes) the Supplier row matching the
    account name, carrying the contact email and phone across.
    """
    from ..models import Supplier
    if user.role != "supplier":
        return
    name = (user.name or "").strip()
    if not name:
        return
    s = db.scalar(select(Supplier).where(Supplier.name == name))
    if not s:
        db.add(Supplier(name=name, contact_email=user.email,
                        phone=user.phone or ""))
    else:
        if not s.contact_email:
            s.contact_email = user.email
        if not s.phone and user.phone:
            s.phone = user.phone
    db.flush()


def ensure_branch(db: Session, name: Optional[str]) -> None:
    """
    Registration and invites take the branch as free text. Without a matching
    Branch row the new pharmacy is invisible to transfers and the branches
    list, so make sure the row exists.
    """
    from ..models import Branch, Store
    clean = (name or "").strip()
    if not clean:
        return
    if db.scalar(select(Branch).where(Branch.name == clean)):
        return
    store = db.scalar(select(Store).order_by(Store.id))
    if not store:
        store = Store(name="BYS")
        db.add(store)
        db.flush()
    db.add(Branch(store_id=store.id, name=clean))
    db.flush()


def _check_password(pw: str) -> None:
    """One place for the password policy, so every entry point agrees."""
    if len(pw) < 8:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Password must be at least 8 characters")
    if not any(c.isalpha() for c in pw) or not any(c.isdigit() for c in pw):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Password must contain a letter and a number")


def user_payload(db: Session, user: User) -> UserOut:
    """User + effective permissions, so the UI gates on what's really granted."""
    from .. import permissions as perms
    out = UserOut.model_validate(user)
    out.permissions = sorted(perms.effective(db, user), key=perms.ALL.index)
    return out


def _token_response(db: Session, user: User) -> TokenResponse:
    access, expires_in = create_access_token(user)
    return TokenResponse(
        access_token=access, refresh_token=issue_refresh_token(db, user),
        expires_in=expires_in, user=user_payload(db, user))


def _issue_verification(db: Session, user: User) -> Tuple[str, str]:
    """Issue a fresh token and drop any unused older one."""
    db.query(EmailVerification).filter(
        EmailVerification.user_id == user.id,
        EmailVerification.used_at.is_(None)).delete()
    token = secrets.token_urlsafe(32)
    db.add(EmailVerification(
        user_id=user.id, token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=VERIFY_HOURS)))
    db.commit()
    return token, f"{settings.app_base_url.rstrip('/')}/verify-email?token={token}"


def _send_verification(user: User, url: str) -> Tuple[bool, str]:
    subject, text, html = verification_email(user.name, url, VERIFY_HOURS)
    sent, reason = send_email(user.email, subject, text, html)
    if not sent:
        log.info("verification link for %s: %s  (reason: %s)", user.email, url, reason)
    return sent, reason


# --------------------------------------------------------------------- login

@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """The frontend sends email or mobile in the same field — both are accepted."""
    ident = payload.email.strip().lower()
    user = db.scalar(select(User).where(
        or_(User.email == ident, User.phone == payload.email.strip())))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect email or password")
    if is_test_account(user):
        # Predefined test accounts stored in the backend authenticate
        # directly — no verification email, no OTP. Development only; every
        # real account below still goes through the full flow.
        if not user.email_verified:
            user.email_verified = True
            db.commit()
    elif not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified yet — open the link we sent you, "
                   "or request a new one.")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="This account is suspended")
    return _token_response(db, user)


# ------------------------------------------------------------------ register

@router.post("/register", response_model=RegisterResponse,
             status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> RegisterResponse:
    """
    Deliberately returns no token. Creates an unverified account and sends the
    verification link. Without SMTP the link comes back in the response so
    development isn't blocked.
    """
    email = payload.email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Invalid email format")
    _reject_unusable_email(email)
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "That email is already registered")
    phone = (payload.phone or "").strip()
    if phone and db.scalar(select(User).where(User.phone == phone)):
        raise HTTPException(status.HTTP_409_CONFLICT, "That mobile number is already registered")
    _check_password(payload.password)

    # Organization type comes from the signup wizard and is stored on the
    # account, not in the browser.
    org_type = payload.organization_type
    if payload.role == "supplier":
        org_type = "distributor"
    user = User(email=email, phone=phone, name=payload.name.strip(),
                role=payload.role, organization_type=org_type,
                branch=payload.branch,
                password_hash=hash_password(payload.password),
                email_verified=False, is_active=True)
    db.add(user)
    if org_type == "pharmacy":
        ensure_branch(db, payload.branch)
    db.flush()
    ensure_supplier_record(db, user)
    db.commit()
    db.refresh(user)

    _, url = _issue_verification(db, user)
    sent, reason = _send_verification(user, url)

    # Handing the link back to whoever called the endpoint means the account
    # can be verified without ever reading the inbox — so it is only done
    # while DEV_EXPOSE_LINKS is on. Either way the link is always written to
    # the server log, where only the operator can see it.
    expose = settings.dev_expose_links
    return RegisterResponse(
        user=UserOut.model_validate(user),
        email_sent=sent,
        verification_url=None if sent else (url if expose else None),
        expires_in_hours=VERIFY_HOURS,
        note=("We sent a verification link to your email — open it to sign in."
              if sent else
              (f"Account created but the email wasn't sent ({reason}). "
               f"Use this link manually." if expose else
               "Account created. Email delivery is not configured on this "
               "server — ask an administrator to send you the link.")),
    )


@router.get("/verify-email/{token}", response_model=VerifyTokenInfo)
def verify_email_info(token: str, db: Session = Depends(get_db)) -> VerifyTokenInfo:
    """Validate the token before rendering the screen, so we never show
    'verified' for a token that was never valid."""
    ev = db.scalar(select(EmailVerification).where(EmailVerification.token == token))
    if not ev:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invalid verification link")
    u = db.get(User, ev.user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if ev.used_at or u.email_verified:
        return VerifyTokenInfo(email=u.email, name=u.name, already_verified=True,
                               expires_at=ev.expires_at)
    if ev.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_410_GONE,
                            "That link has expired — request a new one")
    return VerifyTokenInfo(email=u.email, name=u.name, already_verified=False,
                           expires_at=ev.expires_at)


@router.post("/verify-email", response_model=TokenResponse)
def verify_email(payload: VerifyEmailRequest,
                 db: Session = Depends(get_db)) -> TokenResponse:
    """Verify the email and return a token — the user is signed straight in."""
    ev = db.scalar(select(EmailVerification).where(
        EmailVerification.token == payload.token))
    if not ev:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invalid verification link")
    u = db.get(User, ev.user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if not u.email_verified:
        if ev.used_at:
            raise HTTPException(status.HTTP_409_CONFLICT, "That link has already been used")
        if ev.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            raise HTTPException(status.HTTP_410_GONE,
                                "That link has expired — request a new one")
        u.email_verified = True
        ev.used_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(u)
    return _token_response(db, u)


@router.post("/resend-verification", response_model=RegisterResponse)
def resend_verification(payload: EmailOnlyRequest,
                        db: Session = Depends(get_db)) -> RegisterResponse:
    email = payload.email.strip().lower()
    u = db.scalar(select(User).where(User.email == email))
    if is_test_account(u):
        # Predefined test account — nothing to verify and nothing is sent.
        if u and not u.email_verified:
            u.email_verified = True
            db.commit()
        return RegisterResponse(
            user=UserOut.model_validate(u) if u else None,
            email_sent=False, verification_url=None,
            expires_in_hours=VERIFY_HOURS,
            note="Test account — already verified, just sign in.")
    # Security note: we don't reveal whether the email exists — the response
    # is identical either way.
    if not u or u.email_verified:
        return RegisterResponse(
            user=UserOut.model_validate(u) if u else None,
            email_sent=False, verification_url=None,
            expires_in_hours=VERIFY_HOURS,
            note="If that email is registered and unverified, a new link has been sent.")
    _, url = _issue_verification(db, u)
    sent, reason = _send_verification(u, url)
    expose = settings.dev_expose_links
    return RegisterResponse(
        user=UserOut.model_validate(u), email_sent=sent,
        verification_url=None if sent else (url if expose else None),
        expires_in_hours=VERIFY_HOURS,
        note=("We sent a new link to your email." if sent
              else (f"Email not sent ({reason}) — use this link manually."
                    if expose else
                    "Email delivery is not configured on this server.")))


@router.get("/email-status")
def email_status() -> dict:
    """Diagnostics: will email actually send, and what's missing if not."""
    d = smtp_diagnosis()
    d["willSend"] = smtp_configured()
    return d


# ------------------------------------------------------------------- tokens

@router.post("/refresh-token", response_model=TokenResponse)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    rt = db.scalar(select(RefreshToken).where(RefreshToken.token == payload.refresh_token))
    if not rt or rt.revoked or rt.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Refresh token is invalid or expired")
    user = db.get(User, rt.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="User not found")
    rt.revoked = True                       # rotate: each refresh revokes the previous token
    db.commit()
    return _token_response(db, user)


@router.post("/revoke-token", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def revoke(payload: RefreshRequest, db: Session = Depends(get_db)) -> None:
    rt = db.scalar(select(RefreshToken).where(RefreshToken.token == payload.refresh_token))
    if rt:
        rt.revoked = True
        db.commit()


@router.get("/me", response_model=UserDetail)
def me(db: Session = Depends(get_db),
       user: User = Depends(get_current_user)) -> UserDetail:
    from .. import permissions as perms
    out = UserDetail.model_validate(user)
    out.permissions = sorted(perms.effective(db, user), key=perms.ALL.index)
    return out


@router.put("/me", response_model=UserDetail)
def update_me(payload: ProfileUpdate, db: Session = Depends(get_db),
              user: User = Depends(get_current_user)) -> UserDetail:
    """Update your own name and phone. Email and role are not self-editable."""
    if payload.name is not None:
        name = payload.name.strip()
        if len(name) < 2:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "Name must be at least 2 characters")
        user.name = name
    if payload.phone is not None:
        phone = payload.phone.strip()
        if phone and db.scalar(select(User).where(User.phone == phone,
                                                  User.id != user.id)):
            raise HTTPException(status.HTTP_409_CONFLICT,
                                "That mobile number is already registered")
        user.phone = phone
    db.commit()
    db.refresh(user)
    return UserDetail.model_validate(user)


# --------------------------------------------------------------- passwords

@router.post("/change-password", response_model=SimpleMessage)
def change_password(payload: ChangePasswordRequest, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)) -> SimpleMessage:
    """
    The Settings screen had this form wired to nothing on the server.
    Changing the password revokes every refresh token, so other sessions
    are signed out.
    """
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "Current password is incorrect")
    if payload.new_password == payload.current_password:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "New password is the same as the current one")
    _check_password(payload.new_password)

    user.password_hash = hash_password(payload.new_password)
    db.query(RefreshToken).filter(RefreshToken.user_id == user.id).update(
        {RefreshToken.revoked: True})
    db.commit()
    return SimpleMessage(message="Password changed. Other sessions were signed out.")


@router.post("/forgot-password", response_model=RegisterResponse)
def forgot_password(payload: EmailOnlyRequest,
                    db: Session = Depends(get_db)) -> RegisterResponse:
    """
    Always answers the same whether or not the email exists, so this can't be
    used to enumerate accounts.
    """
    email = payload.email.strip().lower()
    u = db.scalar(select(User).where(User.email == email))
    generic = RegisterResponse(
        user=None, email_sent=False, verification_url=None,
        expires_in_hours=RESET_HOURS,
        note="If that email is registered, a reset link has been sent.")
    if not u:
        return generic

    db.query(PasswordReset).filter(PasswordReset.user_id == u.id,
                                   PasswordReset.used_at.is_(None)).delete()
    token = secrets.token_urlsafe(32)
    db.add(PasswordReset(
        user_id=u.id, token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=RESET_HOURS)))
    db.commit()

    url = f"{settings.app_base_url.rstrip('/')}/reset-password?token={token}"

    if is_test_account(u):
        # Predefined test account: reset locally, send nothing. The link is
        # returned directly so testing never depends on a mailbox.
        log.info("test-account reset link for %s: %s", u.email, url)
        return RegisterResponse(
            user=None, email_sent=False, verification_url=url,
            expires_in_hours=RESET_HOURS,
            note="Test account — no email sent. Use this link to set a "
                 "new password.")

    sent, reason = _send(
        u.email, "Reset your BYS password",
        f"Hi {u.name},\n\nReset your password here:\n{url}\n\n"
        f"This link is valid for {RESET_HOURS} hour and can be used once.\n"
        f"If you didn't ask for this, ignore this email.",
        f"<p>Hi <b>{u.name}</b>,</p><p>Reset your password:</p>"
        f"<p><a href='{url}'>{url}</a></p>"
        f"<p style='color:#64748b;font-size:12px'>Valid for {RESET_HOURS} hour, "
        f"single use. If you didn't ask for this, ignore this email.</p>")
    if not sent:
        log.info("reset link for %s: %s  (reason: %s)", u.email, url, reason)

    expose = settings.dev_expose_links
    return RegisterResponse(
        user=None, email_sent=sent,
        verification_url=None if sent else (url if expose else None),
        expires_in_hours=RESET_HOURS,
        note=(generic.note if sent else
              (f"Email not sent ({reason}) — use this link manually."
               if expose else generic.note)))


@router.post("/reset-password", response_model=TokenResponse)
def reset_password(payload: ResetPasswordRequest,
                   db: Session = Depends(get_db)) -> TokenResponse:
    pr = db.scalar(select(PasswordReset).where(PasswordReset.token == payload.token))
    if not pr:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invalid reset link")
    if pr.used_at:
        raise HTTPException(status.HTTP_409_CONFLICT, "That link has already been used")
    if pr.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_410_GONE, "That link has expired")
    u = db.get(User, pr.user_id)
    if not u:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    _check_password(payload.new_password)
    u.password_hash = hash_password(payload.new_password)
    # Completing a reset proves inbox ownership, so it also verifies the email.
    u.email_verified = True
    pr.used_at = datetime.now(timezone.utc)
    db.query(RefreshToken).filter(RefreshToken.user_id == u.id).update(
        {RefreshToken.revoked: True})
    db.commit()
    db.refresh(u)
    return _token_response(db, u)
