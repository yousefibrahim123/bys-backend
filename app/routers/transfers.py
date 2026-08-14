"""
Pharmacy-to-pharmacy stock transfer requests.

Flow (matches the product spec exactly):
  1. Source branch creates a request        -> POST /api/transfers
  2. Destination branch gets a notification
  3. Destination accepts or rejects         -> PUT /api/transfers/{id}/accept|reject
  4. Stock moves ONLY on acceptance, atomically on both sides
  5. The source branch is notified of the decision

Nothing here touches stock before approval — a pending request reserves
nothing and moves nothing.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..emailer import send_email
from ..models import Branch, Notification, Product, TransferRequest, User
from ..schemas import TransferCreate, TransferDecision, TransferOut
from ..security import get_current_user

log = logging.getLogger("transfers")
router = APIRouter(prefix="/api/transfers", tags=["transfers"])


def _out(t: TransferRequest) -> TransferOut:
    return TransferOut.model_validate(t)


def _email_branch(db: Session, branch: str, subject: str, body: str) -> None:
    """Email the active users of a branch, when SMTP is configured.
    Failures are logged — the in-app notification is the source of truth."""
    for u in db.scalars(select(User).where(User.branch == branch,
                                           User.is_active.is_(True),
                                           User.is_test_account.is_(False))):
        sent, reason = send_email(u.email, subject, body, f"<p>{body}</p>")
        if not sent:
            log.warning("transfer email to %s not sent: %s", u.email, reason)


def _email_user(db: Session, user_id: Optional[int], subject: str, body: str) -> None:
    u = db.get(User, user_id) if user_id else None
    if u and u.is_active and not u.is_test_account:
        sent, reason = send_email(u.email, subject, body, f"<p>{body}</p>")
        if not sent:
            log.warning("transfer email to %s not sent: %s", u.email, reason)


def _branch_exists(db: Session, name: str) -> bool:
    return bool(db.scalar(select(Branch).where(Branch.name == name)))


def _can_act_for(user: User, branch: str) -> bool:
    """Admins act for any branch; branch users only for their own."""
    if user.role == "admin":
        return True
    return (user.branch or "") == branch


@router.post("", response_model=TransferOut, status_code=status.HTTP_201_CREATED)
def create_transfer(payload: TransferCreate, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)) -> TransferOut:
    p = db.get(Product, payload.product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    if not _can_act_for(user, p.branch):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "You can only transfer stock out of your own branch")

    qty = payload.quantity
    if qty <= 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Quantity must be greater than zero")
    if qty > p.quantity:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Only {p.quantity} units of \"{p.name}\" are in stock")

    target = payload.to_branch.strip()
    if not target:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Destination branch is required")
    if target == p.branch:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Source and destination branch are the same")
    if not _branch_exists(db, target):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No branch named {target}")

    t = TransferRequest(
        product_id=p.id, sku=p.sku, product_name=p.name,
        from_branch=p.branch, to_branch=target, quantity=qty,
        note=payload.note.strip(), status="pending", requested_by=user.id)
    db.add(t)
    db.flush()

    # Notify the destination pharmacy — this is their cue to accept or reject.
    body = (f"{p.branch} wants to transfer {qty} units of {p.name} "
            f"(SKU {p.sku}) to {target}. "
            f"Open Stock Transfers to accept or reject."
            + (f" Note: {t.note}" if t.note else ""))
    db.add(Notification(
        branch=target, kind="info",
        title=f"Transfer request: {qty} × {p.name}", body=body))
    db.commit()
    _email_branch(db, target, f"Transfer request: {qty} × {p.name}", body)
    db.refresh(t)
    log.info("transfer request #%d: %d × %s from %s to %s",
             t.id, qty, p.name, p.branch, target)
    return _out(t)


@router.get("", response_model=List[TransferOut])
def list_transfers(direction: str = Query("all", pattern="^(all|incoming|outgoing)$"),
                   status_filter: Optional[str] = Query(None, alias="status"),
                   branch: Optional[str] = None,
                   limit: int = Query(100, ge=1, le=500),
                   db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)) -> List[TransferOut]:
    """
    Transfers visible to this user.

    A branch user sees requests where their branch is source or destination.
    An admin sees everything (optionally narrowed with ?branch=).
    """
    stmt = select(TransferRequest)
    if user.role == "admin":
        b = branch
        if b:
            if direction == "incoming":
                stmt = stmt.where(TransferRequest.to_branch == b)
            elif direction == "outgoing":
                stmt = stmt.where(TransferRequest.from_branch == b)
            else:
                stmt = stmt.where((TransferRequest.to_branch == b)
                                  | (TransferRequest.from_branch == b))
    else:
        own = user.branch or ""
        if direction == "incoming":
            stmt = stmt.where(TransferRequest.to_branch == own)
        elif direction == "outgoing":
            stmt = stmt.where(TransferRequest.from_branch == own)
        else:
            stmt = stmt.where((TransferRequest.to_branch == own)
                              | (TransferRequest.from_branch == own))
    if status_filter:
        stmt = stmt.where(TransferRequest.status == status_filter)
    stmt = stmt.order_by(TransferRequest.created_at.desc()).limit(limit)
    return [_out(t) for t in db.scalars(stmt)]


def _get_pending(db: Session, transfer_id: int) -> TransferRequest:
    t = db.get(TransferRequest, transfer_id)
    if not t:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Transfer request not found")
    if t.status != "pending":
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"This request is already {t.status}")
    return t


@router.put("/{transfer_id}/accept", response_model=TransferOut)
def accept_transfer(transfer_id: int, payload: Optional[TransferDecision] = None,
                    db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)) -> TransferOut:
    """Destination branch accepts — stock moves now, atomically."""
    t = _get_pending(db, transfer_id)
    if not _can_act_for(user, t.to_branch):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "Only the destination pharmacy can accept this request")

    src = db.get(Product, t.product_id)
    if not src or src.branch != t.from_branch:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "The source product no longer exists")
    if src.quantity < t.quantity:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Only {src.quantity} units remain at {t.from_branch} — "
            f"the request was for {t.quantity}. Ask them to resend it.")

    dest = db.scalar(select(Product).where(Product.sku == src.sku,
                                           Product.branch == t.to_branch))
    if dest:
        dest.quantity += t.quantity
    else:
        # Destination doesn't stock this item yet — create it there rather
        # than silently dropping the units.
        dest = Product(
            sku=src.sku, barcode=src.barcode, name=src.name,
            category=src.category, supplier=src.supplier,
            branch=t.to_branch, quantity=t.quantity,
            minimum_stock=src.minimum_stock, maximum_stock=src.maximum_stock,
            purchase_price=src.purchase_price, selling_price=src.selling_price,
            expiry_date=src.expiry_date, batch_number=src.batch_number,
            daily_sales=0.0)
        b = db.scalar(select(Branch).where(Branch.name == t.to_branch))
        if b:
            dest.branch_id = b.id
        db.add(dest)
    src.quantity -= t.quantity

    from ..seed import compute_inventory_status
    src.inventory_status = compute_inventory_status(src)
    src.last_updated_date = datetime.now(timezone.utc).isoformat()
    dest.inventory_status = compute_inventory_status(dest)
    dest.last_restock_date = datetime.now(timezone.utc).date().isoformat()
    dest.last_updated_date = datetime.now(timezone.utc).isoformat()

    t.status = "accepted"
    t.decided_by = user.id
    t.decided_at = datetime.now(timezone.utc)
    t.decision_note = (payload.note.strip() if payload else "")

    body = (f"{t.to_branch} accepted the transfer of {t.quantity} units of "
            f"{t.product_name}. Stock has been moved."
            + (f" Note: {t.decision_note}" if t.decision_note else ""))
    db.add(Notification(
        branch=t.from_branch, kind="success",
        title=f"Transfer accepted: {t.quantity} × {t.product_name}", body=body))
    db.commit()
    db.refresh(t)
    _email_user(db, t.requested_by,
                f"Transfer accepted: {t.quantity} × {t.product_name}", body)
    log.info("transfer #%d accepted by %s", t.id, user.email)
    return _out(t)


@router.put("/{transfer_id}/reject", response_model=TransferOut)
def reject_transfer(transfer_id: int, payload: Optional[TransferDecision] = None,
                    db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)) -> TransferOut:
    """Destination branch rejects — nothing moves."""
    t = _get_pending(db, transfer_id)
    if not _can_act_for(user, t.to_branch):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "Only the destination pharmacy can reject this request")

    t.status = "rejected"
    t.decided_by = user.id
    t.decided_at = datetime.now(timezone.utc)
    t.decision_note = (payload.note.strip() if payload else "")

    body = (f"{t.to_branch} rejected the transfer of {t.quantity} units of "
            f"{t.product_name}. Stock was not moved."
            + (f" Reason: {t.decision_note}" if t.decision_note else ""))
    db.add(Notification(
        branch=t.from_branch, kind="warning",
        title=f"Transfer rejected: {t.quantity} × {t.product_name}", body=body))
    db.commit()
    db.refresh(t)
    _email_user(db, t.requested_by,
                f"Transfer rejected: {t.quantity} × {t.product_name}", body)
    log.info("transfer #%d rejected by %s", t.id, user.email)
    return _out(t)


@router.put("/{transfer_id}/cancel", response_model=TransferOut)
def cancel_transfer(transfer_id: int, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)) -> TransferOut:
    """The requesting branch withdraws its own pending request."""
    t = _get_pending(db, transfer_id)
    if not _can_act_for(user, t.from_branch):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "Only the requesting pharmacy can cancel this request")
    t.status = "cancelled"
    t.decided_by = user.id
    t.decided_at = datetime.now(timezone.utc)
    db.add(Notification(
        branch=t.to_branch, kind="info",
        title=f"Transfer cancelled: {t.quantity} × {t.product_name}",
        body=f"{t.from_branch} withdrew the transfer request for "
             f"{t.quantity} units of {t.product_name}."))
    db.commit()
    db.refresh(t)
    return _out(t)
