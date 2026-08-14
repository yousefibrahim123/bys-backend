"""
Supplier catalogue — the medicines a supplier stocks and offers.

Suppliers manage their own listings (drug name, price, available stock) and
can attach offers (a percentage discount, optionally until a date).
Pharmacies search the catalogue by drug name when placing orders; a purchase
order placed against a listing is auto-approved because the listing is the
supplier's standing offer.

  GET    /api/supplier/listings           supplier: own · pharmacy/admin: active ones (?q= search)
  POST   /api/supplier/listings           supplier only — add a medicine
  PUT    /api/supplier/listings/{id}      owner only — price, stock, offer …
  DELETE /api/supplier/listings/{id}      owner only
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import SupplierListing, User
from ..schemas import ListingIn, ListingOut
from ..security import get_current_user, require_roles

log = logging.getLogger("supplier-catalog")
router = APIRouter(prefix="/api/supplier/listings", tags=["supplier-catalog"])

supplier_only = require_roles("supplier", "admin")


def offer_is_active(l: SupplierListing) -> bool:
    if l.offer_percent <= 0:
        return False
    if l.offer_until is None:
        return True
    until = l.offer_until.replace(tzinfo=timezone.utc)
    return until >= datetime.now(timezone.utc)


def effective_price(l: SupplierListing) -> float:
    if offer_is_active(l):
        return round(l.unit_price * (1 - l.offer_percent / 100), 2)
    return round(l.unit_price, 2)


def _out(db: Session, l: SupplierListing) -> ListingOut:
    owner = db.get(User, l.supplier_user_id)
    return ListingOut(
        id=l.id, supplier_user_id=l.supplier_user_id,
        supplier_name=l.supplier_name,
        supplier_phone=(owner.phone if owner else "") or "",
        supplier_email=(owner.email if owner else "") or "",
        product_name=l.product_name, category=l.category,
        unit_price=l.unit_price, available_qty=l.available_qty,
        min_order_qty=l.min_order_qty, offer_percent=l.offer_percent,
        offer_until=l.offer_until, offer_active=offer_is_active(l),
        effective_price=effective_price(l), notes=l.notes,
        is_active=l.is_active, created_at=l.created_at, updated_at=l.updated_at)


def _validate(payload: ListingIn) -> None:
    if len(payload.product_name.strip()) < 2:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Medicine name must be at least 2 characters")
    if payload.unit_price < 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Price can't be negative")
    if payload.available_qty < 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Available quantity can't be negative")
    if payload.min_order_qty < 1:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Minimum order quantity must be at least 1")
    if not 0 <= payload.offer_percent <= 90:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Offer must be between 0% and 90%")


def _parse_until(raw: Optional[str]) -> Optional[datetime]:
    if not raw or not raw.strip():
        return None
    try:
        d = datetime.fromisoformat(raw.strip()[:19])
        return d
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "offerUntil must be an ISO date (YYYY-MM-DD)")


@router.get("", response_model=List[ListingOut])
def list_listings(q: Optional[str] = None, supplier: Optional[str] = None,
                  include_inactive: bool = Query(False),
                  limit: int = Query(100, ge=1, le=500),
                  db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> List[ListingOut]:
    """
    Suppliers see their own catalogue (including paused listings when asked);
    pharmacies and admins see every active listing, searchable by drug name,
    so an offer is visible the moment the supplier publishes it.
    """
    stmt = select(SupplierListing)
    if user.role == "supplier":
        stmt = stmt.where(SupplierListing.supplier_user_id == user.id)
        if not include_inactive:
            stmt = stmt.where(SupplierListing.is_active.is_(True))
    else:
        stmt = stmt.where(SupplierListing.is_active.is_(True))
        if supplier:
            stmt = stmt.where(SupplierListing.supplier_name == supplier)
    if q and q.strip():
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(SupplierListing.product_name.ilike(like),
                              SupplierListing.category.ilike(like),
                              SupplierListing.supplier_name.ilike(like)))
    stmt = stmt.order_by(SupplierListing.product_name).limit(limit)
    return [_out(db, l) for l in db.scalars(stmt)]


@router.post("", response_model=ListingOut, status_code=status.HTTP_201_CREATED)
def create_listing(payload: ListingIn, db: Session = Depends(get_db),
                   user: User = Depends(supplier_only)) -> ListingOut:
    _validate(payload)
    name = payload.product_name.strip()
    dup = db.scalar(select(SupplierListing).where(
        SupplierListing.supplier_user_id == user.id,
        SupplierListing.product_name == name))
    if dup:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"You already list \"{name}\" — edit it instead")
    l = SupplierListing(
        supplier_user_id=user.id, supplier_name=user.name,
        product_name=name, category=payload.category.strip(),
        unit_price=round(payload.unit_price, 2),
        available_qty=payload.available_qty,
        min_order_qty=payload.min_order_qty,
        offer_percent=round(payload.offer_percent, 1),
        offer_until=_parse_until(payload.offer_until),
        notes=payload.notes.strip(), is_active=payload.is_active)
    db.add(l)
    db.commit()
    db.refresh(l)
    log.info("listing #%d created by %s: %s", l.id, user.email, name)
    return _out(db, l)


def _own_listing(db: Session, listing_id: int, user: User) -> SupplierListing:
    l = db.get(SupplierListing, listing_id)
    if not l:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Listing not found")
    if user.role != "admin" and l.supplier_user_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "You can only manage your own listings")
    return l


@router.put("/{listing_id}", response_model=ListingOut)
def update_listing(listing_id: int, payload: ListingIn,
                   db: Session = Depends(get_db),
                   user: User = Depends(supplier_only)) -> ListingOut:
    l = _own_listing(db, listing_id, user)
    _validate(payload)
    name = payload.product_name.strip()
    if name != l.product_name:
        dup = db.scalar(select(SupplierListing).where(
            SupplierListing.supplier_user_id == l.supplier_user_id,
            SupplierListing.product_name == name,
            SupplierListing.id != l.id))
        if dup:
            raise HTTPException(status.HTTP_409_CONFLICT,
                                f"You already list \"{name}\"")
    l.product_name = name
    l.category = payload.category.strip()
    l.unit_price = round(payload.unit_price, 2)
    l.available_qty = payload.available_qty
    l.min_order_qty = payload.min_order_qty
    l.offer_percent = round(payload.offer_percent, 1)
    l.offer_until = _parse_until(payload.offer_until)
    l.notes = payload.notes.strip()
    l.is_active = payload.is_active
    db.commit()
    db.refresh(l)
    return _out(db, l)


@router.delete("/{listing_id}", status_code=status.HTTP_204_NO_CONTENT,
               response_model=None)
def delete_listing(listing_id: int, db: Session = Depends(get_db),
                   user: User = Depends(supplier_only)) -> None:
    l = _own_listing(db, listing_id, user)
    db.delete(l)
    db.commit()
