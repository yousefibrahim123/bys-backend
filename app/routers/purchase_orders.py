"""
Purchase orders — the purchase-request channel between a pharmacy branch and
its supplier.

Creating an order:
  * saves it in the database (it shows up in the list immediately),
  * sends a purchase request notification to the selected supplier,
  * confirms to the ordering branch.

Status tracking: pending -> approved -> shipped -> delivered, with
rejected/cancelled as terminal branches. Every change notifies the other
side. On 'delivered' the quantities are added to stock automatically — once.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..database import get_db
from ..emailer import send_email
from ..models import (Notification, OrderMessage, Product, PurchaseOrder,
                      PurchaseOrderItem, Supplier, SupplierListing, User)
from .. import permissions as perms
from ..schemas import (OrderMessageIn, OrderMessageOut, POItemOut,
                       POItemQtyUpdate, POStatusUpdate, PurchaseOrderCreate,
                       PurchaseOrderOut)
from ..security import get_current_user, resolve_branch
from .supplier_catalog import effective_price, offer_is_active

log = logging.getLogger("purchase-orders")
router = APIRouter(prefix="/api/purchase-orders", tags=["purchase-orders"])

# Allowed transitions — stops e.g. a cancelled order becoming "delivered"
ALLOWED: Dict[str, Set[str]] = {
    "pending": {"approved", "rejected", "cancelled"},
    "approved": {"shipped", "cancelled"},
    "shipped": {"delivered", "cancelled"},
    "delivered": set(),
    "rejected": set(),
    "cancelled": set(),
}

# Which permission each transition needs (admins hold everything).
TRANSITION_PERMISSION: Dict[str, str] = {
    "approved": "approve_orders",
    "rejected": "approve_orders",
    "shipped": "approve_orders",
    "delivered": "approve_orders",
    "cancelled": "cancel_orders",
}


def _require(db: Session, user: User, permission: str) -> None:
    if not perms.has(db, user, permission):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            f"You need the '{permission}' permission for this")


def _out(o: PurchaseOrder) -> PurchaseOrderOut:
    return PurchaseOrderOut(
        id=o.id, order_number=o.order_number, branch=o.branch, supplier=o.supplier,
        status=o.status, total_amount=o.total_amount, notes=o.notes,
        expected_delivery=o.expected_delivery, created_at=o.created_at,
        items=[POItemOut(id=i.id, product_id=i.product_id, product_name=i.product_name,
                         quantity=i.quantity, unit_price=i.unit_price,
                         line_total=i.line_total) for i in o.items])


def _next_order_number(db: Session) -> str:
    """
    Unique, human-readable order number.

    The old `count(*) + 1` collided as soon as an order was deleted — the
    unique constraint then rejected the brand-new order for no visible reason.
    """
    year = datetime.now().year
    seq = (db.scalar(select(func.max(PurchaseOrder.id))) or 0) + 1
    number = f"PO-{year}-{seq:04d}"
    while db.scalar(select(PurchaseOrder.id).where(
            PurchaseOrder.order_number == number)):
        seq += 1
        number = f"PO-{year}-{seq:04d}"
    return number


def _notify_suppliers(db: Session, supplier_name: str, title: str,
                      body: str, send_emails: bool = False) -> None:
    """
    Deliver a purchase-request notification to the supplier it concerns.

    Supplier users have no branch, so branch-scoped notifications never reach
    them — these are addressed to the user directly. Only accounts matching
    the order's supplier get it (test supplier accounts get everything so the
    demo portal stays populated).

    With send_emails=True the supplier is also emailed when SMTP is
    configured: their user accounts first, otherwise the supplier record's
    contact address. Email failures never fail the request — they are logged
    and the in-app notification still lands.
    """
    emailed: Set[str] = set()
    for u in db.scalars(select(User).where(User.role == "supplier",
                                           User.is_active.is_(True))):
        if u.name == supplier_name or u.is_test_account:
            db.add(Notification(user_id=u.id, branch="", kind="info",
                                title=title, body=body))
            if send_emails and u.name == supplier_name and u.email not in emailed:
                sent, reason = send_email(u.email, title, body,
                                          f"<p>{body}</p>")
                emailed.add(u.email)
                if not sent:
                    log.warning("supplier email to %s not sent: %s",
                                u.email, reason)
    if send_emails and not emailed:
        s = db.scalar(select(Supplier).where(Supplier.name == supplier_name))
        if s and s.contact_email:
            sent, reason = send_email(s.contact_email, title, body,
                                      f"<p>{body}</p>")
            if not sent:
                log.warning("supplier email to %s not sent: %s",
                            s.contact_email, reason)


@router.get("", response_model=List[PurchaseOrderOut])
def list_orders(branch: Optional[str] = None, status_filter: Optional[str] = Query(None, alias="status"),
                supplier: Optional[str] = None, limit: int = Query(100, ge=1, le=500),
                db: Session = Depends(get_db),
                user: User = Depends(get_current_user)) -> List[PurchaseOrderOut]:
    stmt = select(PurchaseOrder).options(selectinload(PurchaseOrder.items))
    if user.role == "supplier":
        if not user.is_test_account:
            # Supplier sees orders matching their name, branch, or email
            sup_name = user.name or ""
            sup_branch = user.branch or ""
            stmt = stmt.where((PurchaseOrder.supplier == sup_name) |
                              (PurchaseOrder.supplier == sup_branch) |
                              (PurchaseOrder.supplier == user.email))
        # The demo supplier account retains the platform-wide view.
    else:
        stmt = stmt.where(PurchaseOrder.branch == resolve_branch(user, branch))
    if status_filter:
        stmt = stmt.where(PurchaseOrder.status == status_filter)
    if supplier:
        stmt = stmt.where(PurchaseOrder.supplier == supplier)
    stmt = stmt.order_by(PurchaseOrder.created_at.desc()).limit(limit)
    return [_out(o) for o in db.scalars(stmt)]


@router.get("/{order_id}", response_model=PurchaseOrderOut)
def get_order(order_id: int, db: Session = Depends(get_db),
              user: User = Depends(get_current_user)) -> PurchaseOrderOut:
    o = db.scalar(select(PurchaseOrder).options(selectinload(PurchaseOrder.items))
                  .where(PurchaseOrder.id == order_id))
    if not o:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Purchase order not found")
    if user.role == "supplier":
        if not user.is_test_account and o.supplier != user.name:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                "This order belongs to a different supplier")
    else:
        resolve_branch(user, o.branch)
    return _out(o)


@router.post("", response_model=PurchaseOrderOut, status_code=status.HTTP_201_CREATED)
def create_order(payload: PurchaseOrderCreate, branch: Optional[str] = None,
                 db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> PurchaseOrderOut:
    b = resolve_branch(user, branch)
    _require(db, user, "create_orders")
    if not payload.items:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "A purchase order needs at least one line item")
    supplier_name = (payload.supplier or "").strip()
    if not supplier_name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "A supplier must be selected")

    o = PurchaseOrder(order_number=_next_order_number(db), branch=b,
                      supplier=supplier_name, notes=payload.notes,
                      expected_delivery=payload.expected_delivery, status="pending")
    total = 0.0
    # True while every line comes from this supplier's own catalogue listing.
    # Such an order is auto-approved: the listing IS the supplier's standing
    # offer, so there is nothing left for them to accept.
    all_from_listings = bool(payload.items)
    for it in payload.items:
        if it.quantity <= 0:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "Every line quantity must be greater than zero")
        name, price = it.product_name, it.unit_price
        if it.listing_id:
            l = db.get(SupplierListing, it.listing_id)
            if not l or not l.is_active:
                raise HTTPException(status.HTTP_404_NOT_FOUND,
                                    f"Supplier listing {it.listing_id} is not available")
            if l.supplier_name != supplier_name:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    f"Listing \"{l.product_name}\" belongs to {l.supplier_name}, "
                    f"not {supplier_name} — one order per supplier")
            if it.quantity < l.min_order_qty:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    f"\"{l.product_name}\" has a minimum order of {l.min_order_qty}")
            if it.quantity > l.available_qty:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    f"Only {l.available_qty} units of \"{l.product_name}\" are "
                    f"available from {l.supplier_name}")
            # The price always comes from the listing (offer applied
            # server-side) — never from the client.
            name = name or l.product_name
            price = effective_price(l)
            l.available_qty -= it.quantity
        else:
            all_from_listings = False
            if it.product_id:
                p = db.get(Product, it.product_id)
                if not p:
                    raise HTTPException(status.HTTP_404_NOT_FOUND,
                                        f"Product {it.product_id} not found")
                name = name or p.name
                price = price or p.purchase_price
        line = round(it.quantity * price, 2)
        total += line
        o.items.append(PurchaseOrderItem(product_id=it.product_id,
                                         listing_id=it.listing_id,
                                         product_name=name,
                                         quantity=it.quantity, unit_price=price,
                                         line_total=line))
    o.total_amount = round(total, 2)
    if all_from_listings:
        o.status = "approved"
    db.add(o)
    db.flush()

    # The purchase request goes to the supplier, and the branch gets a
    # confirmation it can see in its activity feed.
    lines = ", ".join(f"{i.quantity} × {i.product_name}" for i in o.items[:5])
    if o.status == "approved":
        _notify_suppliers(
            db, supplier_name,
            title=f"Order {o.order_number} placed on your offer",
            body=(f"{b} ordered from your published catalogue for "
                  f"EGP {o.total_amount:,.2f}: {lines}"
                  f"{'…' if len(o.items) > 5 else ''}. Auto-accepted — "
                  f"prepare it for shipping."),
            send_emails=True)
        db.add(Notification(
            branch=b, kind="success",
            title=f"Order {o.order_number} auto-accepted by {supplier_name}",
            body=f"EGP {o.total_amount:,.2f} · {len(o.items)} item(s). "
                 f"Ordered from the supplier's published offer — no approval "
                 f"needed."))
    else:
        _notify_suppliers(
            db, supplier_name,
            title=f"Purchase request {o.order_number} from {b}",
            body=(f"{b} placed a purchase request with {supplier_name} for "
                  f"EGP {o.total_amount:,.2f}: {lines}"
                  f"{'…' if len(o.items) > 5 else ''}. Status: pending."),
            send_emails=True)
        db.add(Notification(
            branch=b, kind="info",
            title=f"Purchase order {o.order_number} sent to {supplier_name}",
            body=f"EGP {o.total_amount:,.2f} · {len(o.items)} item(s). "
                 f"Awaiting supplier approval."))
    db.commit()
    db.refresh(o)
    log.info("PO %s created for %s -> %s (EGP %.2f)",
             o.order_number, b, supplier_name, o.total_amount)
    return _out(o)


@router.put("/{order_id}/items", response_model=PurchaseOrderOut)
def update_items(order_id: int, payload: POItemQtyUpdate,
                 db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> PurchaseOrderOut:
    """Edit line quantities while the order is still pending."""
    o = db.scalar(select(PurchaseOrder).options(selectinload(PurchaseOrder.items))
                  .where(PurchaseOrder.id == order_id))
    if not o:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Purchase order not found")
    if user.role != "supplier":
        resolve_branch(user, o.branch)
    _require(db, user, "create_orders")
    if o.status != "pending":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Only a pending order can be modified")
    total = 0.0
    for item in o.items:
        if item.id in payload.quantities:
            qty = int(payload.quantities[item.id])
            if qty <= 0:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    "Quantities must be greater than zero")
            item.quantity = qty
            item.line_total = round(qty * item.unit_price, 2)
        total += item.line_total
    o.total_amount = round(total, 2)
    db.commit()
    db.refresh(o)
    return _out(o)


@router.put("/{order_id}/status", response_model=PurchaseOrderOut)
def update_status(order_id: int, payload: POStatusUpdate,
                  db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> PurchaseOrderOut:
    """
    Change status, validating the transition, and notify the other side.
    On 'delivered' the quantities are added to stock automatically — once.
    """
    o = db.scalar(select(PurchaseOrder).options(selectinload(PurchaseOrder.items))
                  .where(PurchaseOrder.id == order_id))
    if not o:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Purchase order not found")
    if user.role == "supplier":
        # A supplier may only act on orders addressed to them.
        if not user.is_test_account and o.supplier != user.name:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                "This order belongs to a different supplier")
    else:
        resolve_branch(user, o.branch)

    new = payload.status
    if new == o.status:
        return _out(o)
    if new not in ALLOWED[o.status]:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Can't move from '{o.status}' to '{new}'. "
            f"Allowed: {', '.join(sorted(ALLOWED[o.status])) or 'none (final state)'}")
    needed = TRANSITION_PERMISSION.get(new)
    if needed:
        _require(db, user, needed)

    if new == "delivered":
        from ..seed import compute_inventory_status
        for it in o.items:
            if it.product_id:
                p = db.get(Product, it.product_id)
                if p:
                    p.quantity += it.quantity
                    p.last_restock_date = datetime.now(timezone.utc).date().isoformat()
                    p.inventory_status = compute_inventory_status(p)

    if new in ("cancelled", "rejected"):
        # Units reserved from supplier listings go back on the shelf.
        for it in o.items:
            if it.listing_id:
                l = db.get(SupplierListing, it.listing_id)
                if l:
                    l.available_qty += it.quantity

    o.status = new
    STATUS_TEXT = {
        "approved": ("success", "approved"),
        "rejected": ("warning", "rejected"),
        "shipped": ("info", "shipped"),
        "delivered": ("success", "delivered — stock has been updated"),
        "cancelled": ("warning", "cancelled"),
    }
    kind, text = STATUS_TEXT.get(new, ("info", new))
    # The branch always hears about progress on its order…
    db.add(Notification(
        branch=o.branch, kind=kind,
        title=f"Order {o.order_number} {text.split(' — ')[0]}",
        body=f"Purchase order {o.order_number} ({o.supplier}, "
             f"EGP {o.total_amount:,.2f}) is now {text}."))
    # …and the supplier hears when the branch cancels.
    if new == "cancelled":
        _notify_suppliers(
            db, o.supplier, title=f"Order {o.order_number} cancelled",
            body=f"{o.branch} cancelled purchase order {o.order_number} "
                 f"({o.supplier}, EGP {o.total_amount:,.2f}).")
    db.commit()
    db.refresh(o)
    return _out(o)


@router.delete("/{order_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_order(order_id: int, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> None:
    o = db.get(PurchaseOrder, order_id)
    if not o:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Purchase order not found")
    resolve_branch(user, o.branch)
    # Withdrawing your own pending order only needs create_orders; anything
    # further along requires the explicit cancel permission.
    if o.status == "pending":
        if not (perms.has(db, user, "create_orders")
                or perms.has(db, user, "cancel_orders")):
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                "You need the 'create_orders' permission for this")
    else:
        _require(db, user, "cancel_orders")
    if o.status == "delivered":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "A delivered purchase order can't be deleted")
    if o.status not in ("cancelled", "rejected"):
        # Units reserved from supplier listings go back on the shelf.
        items = db.scalars(select(PurchaseOrderItem).where(
            PurchaseOrderItem.order_id == o.id))
        for it in items:
            if it.listing_id:
                l = db.get(SupplierListing, it.listing_id)
                if l:
                    l.available_qty += it.quantity
    db.query(OrderMessage).filter(OrderMessage.order_id == o.id).delete()
    db.delete(o)
    db.commit()


# ------------------------------------------------------------- order chat

def _chat_access(db: Session, user: User, o: PurchaseOrder) -> str:
    """
    Who may read/write this order's chat, and which side they speak for.
    Pharmacy side: admins and the ordering branch. Supplier side: the supplier
    the order is addressed to (test supplier accounts see everything).
    """
    if user.role == "supplier":
        if not user.is_test_account and o.supplier != user.name:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                "This order belongs to a different supplier")
        return "supplier"
    resolve_branch(user, o.branch)
    return "admin" if user.role == "admin" else "pharmacy"


@router.get("/{order_id}/messages", response_model=List[OrderMessageOut])
def list_messages(order_id: int, db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> List[OrderMessageOut]:
    o = db.get(PurchaseOrder, order_id)
    if not o:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Purchase order not found")
    _chat_access(db, user, o)
    rows = db.scalars(select(OrderMessage)
                      .where(OrderMessage.order_id == o.id)
                      .order_by(OrderMessage.created_at, OrderMessage.id))
    return [OrderMessageOut.model_validate(m) for m in rows]


@router.post("/{order_id}/messages", response_model=OrderMessageOut,
             status_code=status.HTTP_201_CREATED)
def send_message(order_id: int, payload: OrderMessageIn,
                 db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> OrderMessageOut:
    """
    Post a message on the order thread. The other side gets an in-app
    notification so the conversation is never missed.
    """
    o = db.get(PurchaseOrder, order_id)
    if not o:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Purchase order not found")
    side = _chat_access(db, user, o)
    body = payload.body.strip()
    if not body:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Message can't be empty")
    if len(body) > 2000:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Message is too long (2000 characters max)")

    m = OrderMessage(order_id=o.id, sender_id=user.id, sender_name=user.name,
                     sender_side=side, body=body)
    db.add(m)

    preview = body if len(body) <= 120 else body[:117] + "…"
    if side == "supplier":
        db.add(Notification(
            branch=o.branch, kind="info",
            title=f"Message on order {o.order_number} from {user.name}",
            body=preview))
    else:
        _notify_suppliers(db, o.supplier,
                          title=f"Message on order {o.order_number} from {o.branch}",
                          body=preview)
    db.commit()
    db.refresh(m)
    return OrderMessageOut.model_validate(m)
