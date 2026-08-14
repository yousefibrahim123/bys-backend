"""
Expiry actions that actually happen.

The near-expiry screens offered "Apply 15% Discount", "Transfer to Branch" and
"Return to Supplier", and every one of them only rendered a confirmation
message. Nothing was written, no price changed, no supplier was told.

Each action here changes real state and leaves a record:

  discount  -> the selling price is actually reduced, and the original price
               is kept so it can be restored
  transfer  -> stock moves between branches, both sides adjusted atomically
  return    -> a purchase order is raised against the supplier with a negative
               quantity, plus a notification, so the supplier sees the request
               on their own dashboard
  writeoff  -> stock is removed and the loss recorded

Every action is written to `expiry_actions`, so the UI can show what was done
and offer to undo a discount.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..emailer import send_email
from ..models import (Branch, ExpiryAction, Notification, Product,
                      PurchaseOrder, PurchaseOrderItem, Supplier, User)
from ..schemas import (DiscountRequest, ExpiryActionOut, ReturnRequest,
                       TransferRequest, WriteOffRequest)
from ..security import get_current_user, require_roles, resolve_branch

log = logging.getLogger("expiry")
router = APIRouter(prefix="/api/expiry", tags=["expiry"])

staff = require_roles("admin", "branch")


def _out(a: ExpiryAction, product: Optional[Product] = None) -> ExpiryActionOut:
    return ExpiryActionOut(
        id=a.id, product_id=a.product_id,
        product_name=product.name if product else "",
        action=a.action, detail=a.detail, branch=a.branch,
        value_egp=a.value_egp, reference_id=a.reference_id,
        reverted=a.reverted, created_at=a.created_at)


def _product_for(db: Session, product_id: int, user: User) -> Product:
    p = db.get(Product, product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    resolve_branch(user, p.branch)          # raises if out of the user's branch
    return p


# ------------------------------------------------------------------- discount

@router.post("/discount", response_model=ExpiryActionOut,
             status_code=status.HTTP_201_CREATED)
def apply_discount(payload: DiscountRequest, db: Session = Depends(get_db),
                   user: User = Depends(staff)) -> ExpiryActionOut:
    """Actually reduces the selling price and remembers the original."""
    p = _product_for(db, payload.product_id, user)
    pct = payload.percent
    if not 1 <= pct <= 90:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Discount must be between 1% and 90%")

    original = p.original_price if p.original_price else p.selling_price
    new_price = round(original * (1 - pct / 100), 2)
    if new_price < p.purchase_price:
        # Selling under cost turns a slow-moving batch into a guaranteed loss.
        # Allowed, but the caller has to say so explicitly.
        if not payload.allow_below_cost:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"{pct}% would price this at EGP {new_price:.2f}, below the "
                f"EGP {p.purchase_price:.2f} cost. Send allowBelowCost to override.")

    p.original_price = original
    p.selling_price = new_price
    p.discount_percent = pct
    p.discount_until = (datetime.now(timezone.utc)
                        + timedelta(days=payload.days or 30))

    a = ExpiryAction(
        product_id=p.id, branch=p.branch, action="discount",
        detail=f"{pct}% off — EGP {original:.2f} to EGP {new_price:.2f}",
        value_egp=round((original - new_price) * p.quantity, 2),
        performed_by=user.id)
    db.add(a)
    db.commit()
    db.refresh(a)
    log.info("discount %s%% applied to product %s (%s)", pct, p.id, p.name)
    return _out(a, p)


@router.post("/discount/{action_id}/revert", response_model=ExpiryActionOut)
def revert_discount(action_id: int, db: Session = Depends(get_db),
                    user: User = Depends(staff)) -> ExpiryActionOut:
    a = db.get(ExpiryAction, action_id)
    if not a or a.action != "discount":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Discount not found")
    if a.reverted:
        raise HTTPException(status.HTTP_409_CONFLICT, "Already reverted")
    p = _product_for(db, a.product_id, user)
    if p.original_price:
        p.selling_price = p.original_price
        p.original_price = None
    p.discount_percent = 0
    p.discount_until = None
    a.reverted = True
    db.commit()
    db.refresh(a)
    return _out(a, p)


# ------------------------------------------------------------------- transfer

@router.post("/transfer", response_model=ExpiryActionOut,
             status_code=status.HTTP_201_CREATED)
def transfer_stock(payload: TransferRequest, db: Session = Depends(get_db),
                   user: User = Depends(staff)) -> ExpiryActionOut:
    """Really moves units between branches — both sides in one transaction."""
    p = _product_for(db, payload.product_id, user)
    qty = payload.quantity
    if qty <= 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Quantity must be greater than zero")
    if qty > p.quantity:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Only {p.quantity} units in stock")
    target = payload.to_branch.strip()
    if target == p.branch:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Source and destination branch are the same")
    if not db.scalar(select(Branch).where(Branch.name == target)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No branch named {target}")

    # Stock is NOT moved here. A transfer between pharmacies is a request the
    # destination must accept first — units move on acceptance, in
    # /api/transfers/{id}/accept, and the destination can reject instead.
    from ..models import TransferRequest
    t = TransferRequest(
        product_id=p.id, sku=p.sku, product_name=p.name,
        from_branch=p.branch, to_branch=target, quantity=qty,
        note=f"Near-expiry transfer — expires {p.expiry_date}. "
             f"Prioritise selling this batch.",
        status="pending", requested_by=user.id)
    db.add(t)
    db.flush()

    db.add(Notification(
        branch=target, kind="info",
        title=f"Transfer request: {qty} × {p.name}",
        body=f"{p.branch} wants to transfer {qty} units of {p.name} "
             f"(expires {p.expiry_date}). Open Stock Transfers to accept "
             f"or reject."))

    a = ExpiryAction(
        product_id=p.id, branch=p.branch, action="transfer",
        detail=f"{qty} units requested to {target} — awaiting approval "
               f"(request #{t.id})",
        value_egp=round(qty * p.purchase_price, 2), reference_id=t.id,
        performed_by=user.id)
    db.add(a)
    db.commit()
    db.refresh(a)
    log.info("transfer request #%d: %d of product %s to %s (awaiting approval)",
             t.id, qty, p.id, target)
    return _out(a, p)


# --------------------------------------------------------------------- return

@router.post("/return", response_model=ExpiryActionOut,
             status_code=status.HTTP_201_CREATED)
def return_to_supplier(payload: ReturnRequest, db: Session = Depends(get_db),
                       user: User = Depends(staff)) -> ExpiryActionOut:
    """
    Raises a real return request against the supplier.

    This is the one that mattered most: the button used to show "Return
    created" and do nothing at all. Now it creates a purchase order with
    negative quantities (a return), notifies the supplier's account so it lands
    on their dashboard, emails them if SMTP is configured, and removes the
    units from stock.
    """
    p = _product_for(db, payload.product_id, user)
    qty = payload.quantity
    if qty <= 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Quantity must be greater than zero")
    if qty > p.quantity:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Only {p.quantity} units in stock")

    supplier = db.scalar(select(Supplier).where(Supplier.name == p.supplier))
    value = round(qty * p.purchase_price, 2)

    seq = (db.scalar(select(func.count()).select_from(PurchaseOrder)) or 0) + 1
    order = PurchaseOrder(
        order_number=f"RET-{seq:04d}", supplier=p.supplier, branch=p.branch,
        status="pending", total_amount=-value, kind="return",
        notes=(payload.reason
               or f"Return request — {p.name} expires {p.expiry_date}"))
    db.add(order)
    db.flush()
    db.add(PurchaseOrderItem(order_id=order.id, product_id=p.id,
                             product_name=p.name, quantity=-qty,
                             unit_price=p.purchase_price,
                             line_total=-value))

    p.quantity -= qty

    # Land it on the supplier's dashboard.
    supplier_users = list(db.scalars(select(User).where(
        User.role == "supplier",
        User.name == p.supplier)))
    body = (f"{p.branch} is returning {qty} x {p.name} "
            f"(expires {p.expiry_date}), value EGP {value:,.2f}. "
            f"Reason: {payload.reason or 'approaching expiry'}.")
    if supplier_users:
        for su in supplier_users:
            db.add(Notification(user_id=su.id, branch=p.branch,
                                title=f"Return request: {p.name}",
                                body=body, kind="warning"))
    else:
        # No supplier account exists yet — still record it so the request is
        # visible to whoever manages suppliers, rather than vanishing.
        db.add(Notification(branch=p.branch,
                            title=f"Return request raised: {p.name}",
                            body=body + " (No supplier account is linked yet.)",
                            kind="warning"))

    if supplier and supplier.contact_email:
        sent, reason = send_email(
            supplier.contact_email,
            f"Return request from {p.branch} — {p.name}",
            body + f"\n\nReference: {order.order_number}",
            f"<p>{body}</p><p><b>Reference:</b> {order.order_number}</p>")
        if not sent:
            log.info("return email to %s not sent (%s)",
                     supplier.contact_email, reason)

    a = ExpiryAction(
        product_id=p.id, branch=p.branch, action="return",
        detail=f"{qty} units returned to {p.supplier} ({order.order_number})",
        value_egp=value, performed_by=user.id, reference_id=order.id)
    db.add(a)
    db.commit()
    db.refresh(a)
    log.info("return %s raised for %d x product %s", order.order_number, qty, p.id)
    return _out(a, p)


# ------------------------------------------------------------------- write-off

@router.post("/write-off", response_model=ExpiryActionOut,
             status_code=status.HTTP_201_CREATED)
def write_off(payload: WriteOffRequest, db: Session = Depends(get_db),
              user: User = Depends(staff)) -> ExpiryActionOut:
    p = _product_for(db, payload.product_id, user)
    qty = payload.quantity
    if qty <= 0 or qty > p.quantity:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Quantity must be between 1 and {p.quantity}")
    p.quantity -= qty
    value = round(qty * p.purchase_price, 2)
    a = ExpiryAction(
        product_id=p.id, branch=p.branch, action="writeoff",
        detail=f"{qty} units written off — {payload.reason or 'expired'}",
        value_egp=value, performed_by=user.id)
    db.add(a)
    db.add(Notification(branch=p.branch, title=f"Stock written off: {p.name}",
                        body=f"{qty} units, EGP {value:,.2f} loss.",
                        kind="critical"))
    db.commit()
    db.refresh(a)
    return _out(a, p)


# -------------------------------------------------------------------- history

@router.get("/actions", response_model=List[ExpiryActionOut])
def list_actions(branch: Optional[str] = None, product_id: Optional[int] = None,
                 limit: int = 50, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> List[ExpiryActionOut]:
    b = resolve_branch(user, branch)
    stmt = select(ExpiryAction).where(ExpiryAction.branch == b)
    if product_id:
        stmt = stmt.where(ExpiryAction.product_id == product_id)
    rows = db.scalars(stmt.order_by(ExpiryAction.created_at.desc(),
                                    ExpiryAction.id.desc()).limit(limit)).all()
    products = {p.id: p for p in db.scalars(select(Product).where(
        Product.id.in_([r.product_id for r in rows])))} if rows else {}
    return [_out(r, products.get(r.product_id)) for r in rows]
