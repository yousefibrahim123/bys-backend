"""
Email service — shared by registration and employee invites.

Configured for Gmail by default. Required in .env:

    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your.account@gmail.com
    SMTP_PASSWORD=xxxx xxxx xxxx xxxx      # App Password, NOT your normal Google password
    SMTP_FROM=your.account@gmail.com
    SMTP_STARTTLS=true
    SMTP_SSL=false

Important for Gmail: enable 2-Step Verification on the account, then create
an App Password at https://myaccount.google.com/apppasswords — a normal
Google password is rejected by SMTP with a 535 error.

With no SMTP settings at all, send_email returns (False, clear reason) and
the caller returns the link in the response instead of pretending an email
was sent.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import logging
import re
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr

from .config import settings

log = logging.getLogger("emailer")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")


def is_valid_email(addr: str) -> bool:
    return bool(_EMAIL_RE.match((addr or "").strip()))


def smtp_configured() -> bool:
    return bool(settings.smtp_host and (settings.smtp_from or settings.smtp_user))


def smtp_diagnosis() -> dict:
    """Configuration state — surfaced by /api/auth/email-status."""
    missing = [k for k, v in {
        "SMTP_HOST": settings.smtp_host,
        "SMTP_USER": settings.smtp_user,
        "SMTP_PASSWORD": settings.smtp_password,
        "SMTP_FROM": settings.smtp_from or settings.smtp_user,
    }.items() if not v]
    return {
        "configured": smtp_configured() and not missing,
        "host": settings.smtp_host or None,
        "port": settings.smtp_port,
        "from": settings.smtp_from or settings.smtp_user or None,
        "missing": missing,
        "hint": ("Add these to backend/.env — for Gmail you need an App Password "
                 "from myaccount.google.com/apppasswords"),
    }


def send_email(to: str, subject: str, text_body: str,
               html_body: Optional[str] = None) -> Tuple[bool, str]:
    """
    Returns (sent, reason). Reason is empty on success and carries the real
    error message on failure, so the UI can tell the user exactly what went
    wrong instead of showing something generic.
    """
    if not is_valid_email(to):
        return False, f"Invalid email address: {to}"

    if not smtp_configured():
        return False, "SMTP is not configured in .env"

    sender = settings.smtp_from or settings.smtp_user

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr(("BYS — Buying Yield Stock", sender))
    msg["To"] = to
    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    try:
        ctx = ssl.create_default_context()
        if settings.smtp_ssl:
            server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port,
                                      timeout=25, context=ctx)
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=25)
            if settings.smtp_starttls:
                server.starttls(context=ctx)
        with server:
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)
        log.info("email sent to %s (%s)", to, subject)
        return True, ""

    except smtplib.SMTPAuthenticationError as e:
        reason = ("SMTP rejected the login. For Gmail you must use an App Password "
                  "(myaccount.google.com/apppasswords), not your normal account password. "
                  f"[{e.smtp_code}]")
        log.warning("SMTP auth failed for %s: %s", to, e)
        return False, reason
    except (smtplib.SMTPException, OSError) as e:
        log.warning("SMTP send failed for %s: %s", to, e)
        return False, f"Send failed: {type(e).__name__}: {e}"


# ----------------------------------------------------------------- templates

_SHELL = """\
<!doctype html><html lang="ar" dir="rtl"><body style="margin:0;padding:24px;
background:#f1f5f9;font-family:system-ui,-apple-system,'Segoe UI',Tahoma,sans-serif">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
   <tr><td align="center">
    <table role="presentation" width="100%" style="max-width:520px;background:#fff;
      border-radius:16px;border:1px solid #e2e8f0;overflow:hidden">
      <tr><td style="padding:24px 28px 8px">
        <div style="font-size:22px;font-weight:700;color:#0f172a">BYS</div>
        <div style="font-size:12px;color:#64748b">Buying Yield Stock</div>
      </td></tr>
      <tr><td style="padding:8px 28px 24px;color:#0f172a;font-size:15px;line-height:1.7">
        {body}
      </td></tr>
      <tr><td style="padding:16px 28px;background:#f8fafc;border-top:1px solid #e2e8f0;
        color:#64748b;font-size:11px">
        If you received this by mistake, ignore it — nothing will happen to your account.
      </td></tr>
    </table>
   </td></tr>
  </table>
</body></html>"""


def _button(url: str, label: str) -> str:
    return (f'<div style="margin:24px 0"><a href="{url}" style="display:inline-block;'
            f'background:#2563eb;color:#fff;text-decoration:none;padding:12px 24px;'
            f'border-radius:10px;font-weight:600;font-size:14px">{label}</a></div>'
            f'<div style="font-size:11px;color:#64748b;word-break:break-all">'
            f'Or copy this link: {url}</div>')


def verification_email(name: str, url: str, hours: int) -> Tuple[str, str, str]:
    subject = "Verify your BYS email"
    text = (f"Hi {name},\n\n"
            f"A new BYS account was created with this email address. "
            f"Confirm it's yours:\n{url}\n\n"
            f"This link is valid for {hours} hours.")
    html = _SHELL.format(body=(
        f"<p>Hi <b>{name}</b>,</p>"
        f"<p>A new BYS account was created with this email address. Click the "
        f"button to confirm it's yours and sign in.</p>"
        f"{_button(url, 'Verify email')}"
        f"<p style='color:#64748b;font-size:12px;margin-top:16px'>"
        f"This link is valid for {hours} hours.</p>"))
    return subject, text, html


def invite_email(name: str, inviter: str, role: str, branch: Optional[str],
                 url: str, days: int) -> Tuple[str, str, str]:
    where = f" at the {branch} branch" if branch else ""
    subject = "You've been invited to BYS"
    text = (f"Hi {name},\n\n{inviter} invited you to join BYS as {role}{where}.\n\n"
            f"Activate your account and choose a password:\n{url}\n\n"
            f"This link is valid for {days} days.")
    html = _SHELL.format(body=(
        f"<p>Hi <b>{name}</b>,</p>"
        f"<p><b>{inviter}</b> invited you to join BYS as <b>{role}</b>{where}.</p>"
        f"<p>Click the button to activate your account and choose a password.</p>"
        f"{_button(url, 'Activate account')}"
        f"<p style='color:#64748b;font-size:12px;margin-top:16px'>"
        f"Valid for {days} days, single use only.</p>"))
    return subject, text, html
