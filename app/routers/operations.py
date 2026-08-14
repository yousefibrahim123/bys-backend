"""Inventory, sales, notifications, suppliers, and analytics."""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (Branch, Category, Notification, Product, PurchaseOrder,
                      Sale, Supplier, User)
from ..schemas import (InventorySummary, NotificationOut, ProductOut, SaleCreate,
                       SaleOut, SalesSummary, SupplierOut)
from ..security import get_current_user, resolve_branch

inventory = APIRouter(prefix="/api/inventory", tags=["inventory"])
sales = APIRouter(prefix="/api/sales", tags=["sales"])
misc = APIRouter(prefix="/api", tags=["reference"])
analytics = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _days_left(p: Product) -> Optional[int]:
    if not p.expiry_date:
        return None
    try:
        d = datetime.fromisoformat(p.expiry_date[:10]).date()
        return (d - datetime.now(timezone.utc).date()).days
    except ValueError:
        return None


# -------------------------------------------------------------- inventory

@inventory.get("/summary", response_model=InventorySummary)
def inventory_summary(branch: Optional[str] = None, db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)) -> InventorySummary:
    b = resolve_branch(user, branch)
    ps = list(db.scalars(select(Product).where(Product.branch == b)))
    cost = sum(p.quantity * p.purchase_price for p in ps)
    retail = sum(p.quantity * p.selling_price for p in ps)
    near = expired = 0
    for p in ps:
        d = _days_left(p)
        if d is None:
            continue
        if d < 0:
            expired += 1
        elif d <= 120:
            near += 1
    return InventorySummary(
        branch=b, total_products=len(ps), total_units=sum(p.quantity for p in ps),
        inventory_value=round(cost, 2), retail_value=round(retail, 2),
        potential_profit=round(retail - cost, 2),
        low_stock_count=sum(1 for p in ps if 0 < p.quantity <= p.minimum_stock),
        out_of_stock_count=sum(1 for p in ps if p.quantity <= 0),
        near_expiry_count=near, expired_count=expired,
        categories=len({p.category for p in ps if p.category}),
        suppliers=len({p.supplier for p in ps if p.supplier}))


@inventory.get("/low-stock", response_model=List[ProductOut])
def low_stock(branch: Optional[str] = None, limit: int = Query(50, ge=1, le=500),
              db: Session = Depends(get_db),
              user: User = Depends(get_current_user)) -> List[ProductOut]:
    b = resolve_branch(user, branch)
    ps = db.scalars(select(Product).where(
        Product.branch == b, Product.quantity <= Product.minimum_stock)
        .order_by(Product.quantity).limit(limit))
    return [ProductOut.from_orm_product(p) for p in ps]


@inventory.get("/near-expiry", response_model=List[ProductOut])
def near_expiry(branch: Optional[str] = None, days: int = Query(120, ge=1, le=1000),
                db: Session = Depends(get_db),
                user: User = Depends(get_current_user)) -> List[ProductOut]:
    b = resolve_branch(user, branch)
    out = []
    for p in db.scalars(select(Product).where(Product.branch == b)):
        d = _days_left(p)
        if d is not None and d <= days:
            out.append(p)
    out.sort(key=lambda p: p.expiry_date or "9999")
    return [ProductOut.from_orm_product(p) for p in out]


@inventory.post("/adjust/{product_id}", response_model=ProductOut)
def adjust_stock(product_id: int, delta: int = Query(..., description="positive adds stock, negative removes"),
                 db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> ProductOut:
    p = db.get(Product, product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    resolve_branch(user, p.branch)
    if p.quantity + delta < 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Current stock is {p.quantity} — can't deduct {abs(delta)}")
    p.quantity += delta
    if delta > 0:
        p.last_restock_date = datetime.now(timezone.utc).date().isoformat()
    from ..seed import compute_inventory_status
    p.inventory_status = compute_inventory_status(p)
    p.last_updated_date = datetime.now(timezone.utc).isoformat()
    db.commit()
    db.refresh(p)
    return ProductOut.from_orm_product(p)


# ------------------------------------------------------------------ sales

@sales.post("", response_model=SaleOut, status_code=status.HTTP_201_CREATED)
def record_sale(payload: SaleCreate, db: Session = Depends(get_db),
                user: User = Depends(get_current_user)) -> SaleOut:
    p = db.get(Product, payload.product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    b = resolve_branch(user, p.branch)
    if payload.quantity <= 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Quantity must be greater than zero")
    if p.quantity < payload.quantity:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Only {p.quantity} units of \"{p.name}\" are available")

    price = payload.unit_price if payload.unit_price is not None else p.selling_price
    s = Sale(branch=b, product_id=p.id, quantity=payload.quantity, unit_price=price,
             total=round(payload.quantity * price, 2), customer_name=payload.customer_name)
    p.quantity -= payload.quantity
    from ..seed import compute_inventory_status
    p.inventory_status = compute_inventory_status(p)
    db.add(s)
    db.commit()
    db.refresh(s)
    return SaleOut(id=s.id, branch=s.branch, product_id=s.product_id, product_name=p.name,
                   quantity=s.quantity, unit_price=s.unit_price, total=s.total,
                   sold_at=s.sold_at, customer_name=s.customer_name)


@sales.get("", response_model=List[SaleOut])
def list_sales(branch: Optional[str] = None, days: int = Query(30, ge=1, le=365),
               limit: int = Query(100, ge=1, le=1000),
               db: Session = Depends(get_db),
               user: User = Depends(get_current_user)) -> List[SaleOut]:
    b = resolve_branch(user, branch)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute(
        select(Sale, Product.name).join(Product, Product.id == Sale.product_id)
        .where(Sale.branch == b, Sale.sold_at >= since)
        .order_by(Sale.sold_at.desc()).limit(limit)).all()
    return [SaleOut(id=s.id, branch=s.branch, product_id=s.product_id, product_name=n,
                    quantity=s.quantity, unit_price=s.unit_price, total=s.total,
                    sold_at=s.sold_at, customer_name=s.customer_name) for s, n in rows]


@sales.get("/summary", response_model=SalesSummary)
def sales_summary(branch: Optional[str] = None, days: int = Query(30, ge=1, le=365),
                  db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> SalesSummary:
    b = resolve_branch(user, branch)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = list(db.scalars(select(Sale).where(Sale.branch == b, Sale.sold_at >= since)))

    per_day: Dict[str, float] = defaultdict(float)
    per_product: Dict[int, dict] = {}
    for s in rows:
        per_day[s.sold_at.date().isoformat()] += s.total
        d = per_product.setdefault(s.product_id, {"revenue": 0.0, "units": 0})
        d["revenue"] += s.total
        d["units"] += s.quantity

    names = dict(db.execute(select(Product.id, Product.name).where(
        Product.id.in_(per_product.keys()))).all()) if per_product else {}
    top = sorted(({"name": names.get(pid, f"#{pid}"),
                   "revenue": round(v["revenue"], 2), "units": v["units"]}
                  for pid, v in per_product.items()),
                 key=lambda r: -r["revenue"])[:10]

    revenue = sum(s.total for s in rows)
    return SalesSummary(
        branch=b, days=days, total_revenue=round(revenue, 2),
        total_units=sum(s.quantity for s in rows), transactions=len(rows),
        average_basket=round(revenue / len(rows), 2) if rows else 0.0,
        daily=[{"date": d, "revenue": round(v, 2)} for d, v in sorted(per_day.items())],
        top_products=top)


@sales.get("/monthly")
def sales_monthly(branch: Optional[str] = None, months: int = Query(6, ge=1, le=24),
                  db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> dict:
    """
    Real revenue and profit per calendar month.

    The Reports screen drew a six-month "Revenue & Profit" chart by taking one
    month's figure and multiplying it by a fixed array of six numbers — and
    when there were no sales at all it fell back to a hardcoded 50,000, which
    is why a pharmacy with an empty inventory still saw six months of bars.
    This returns the months that actually happened, and an empty list when
    nothing has been sold.
    """
    b = resolve_branch(user, branch)
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=31 * months)

    rows = db.execute(
        select(Sale, Product.purchase_price)
        .join(Product, Product.id == Sale.product_id)
        .where(Sale.branch == b, Sale.sold_at >= since)).all()

    buckets: Dict[str, dict] = {}
    for sale, cost in rows:
        key = sale.sold_at.strftime("%Y-%m")
        d = buckets.setdefault(key, {"revenue": 0.0, "profit": 0.0, "units": 0})
        d["revenue"] += sale.total
        d["profit"] += sale.total - (cost or 0.0) * sale.quantity
        d["units"] += sale.quantity

    out = [{"month": k,
            "label": datetime.strptime(k, "%Y-%m").strftime("%b"),
            "revenue": round(v["revenue"], 2),
            "profit": round(v["profit"], 2),
            "units": v["units"]}
           for k, v in sorted(buckets.items())]
    return {"branch": b, "months": months, "series": out,
            "hasData": bool(out)}


# ------------------------------------------------- notifications and lookups


def _notification_scope(user: User, branch: Optional[str]):
    """The visibility condition for this user's notification feed."""
    if user.role == "supplier":
        return Notification.user_id == user.id
    b = resolve_branch(user, branch)
    return (Notification.branch == b) | (Notification.user_id == user.id)


@misc.get("/notifications", response_model=List[NotificationOut])
def notifications(branch: Optional[str] = None, unread_only: bool = False,
                  limit: int = Query(50, ge=1, le=200),
                  db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> List[NotificationOut]:
    stmt = select(Notification).where(_notification_scope(user, branch))
    if unread_only:
        stmt = stmt.where(Notification.is_read.is_(False))
    stmt = stmt.order_by(Notification.created_at.desc()).limit(limit)
    return [NotificationOut.model_validate(n) for n in db.scalars(stmt)]


@misc.get("/notifications/unread-count")
def unread_count(branch: Optional[str] = None, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> dict:
    n = db.scalar(select(func.count(Notification.id)).where(
        _notification_scope(user, branch),
        Notification.is_read.is_(False))) or 0
    return {"unreadCount": int(n)}


@misc.put("/notifications/{notification_id}/read", response_model=NotificationOut)
def mark_read(notification_id: int, db: Session = Depends(get_db),
              user: User = Depends(get_current_user)) -> NotificationOut:
    n = db.get(Notification, notification_id)
    if not n:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    n.is_read = True
    db.commit()
    db.refresh(n)
    return NotificationOut.model_validate(n)


@misc.put("/notifications/read-all")
def mark_all_read(branch: Optional[str] = None, db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> dict:
    rows = db.scalars(select(Notification).where(
        _notification_scope(user, branch),
        Notification.is_read.is_(False))).all()
    for n in rows:
        n.is_read = True
    db.commit()
    return {"updated": len(rows)}


@misc.delete("/notifications/{notification_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_notification(notification_id: int, db: Session = Depends(get_db),
                        user: User = Depends(get_current_user)) -> None:
    n = db.get(Notification, notification_id)
    if n:
        db.delete(n)
        db.commit()


@misc.get("/suppliers", response_model=List[SupplierOut])
def suppliers(db: Session = Depends(get_db),
              user: User = Depends(get_current_user)) -> List[SupplierOut]:
    return [SupplierOut.model_validate(s)
            for s in db.scalars(select(Supplier).order_by(Supplier.name))]


@misc.get("/categories", response_model=List[dict])
def categories(db: Session = Depends(get_db),
               user: User = Depends(get_current_user)) -> List[dict]:
    return [{"id": c.id, "name": c.name, "description": c.description}
            for c in db.scalars(select(Category).order_by(Category.name))]


@misc.get("/branches", response_model=List[dict])
def branches(db: Session = Depends(get_db),
             user: User = Depends(get_current_user)) -> List[dict]:
    rows = db.scalars(select(Branch).order_by(Branch.name))
    return [{"id": b.id, "name": b.name, "address": b.address, "phone": b.phone}
            for b in rows]


# -------------------------------------------------------------- analytics

@analytics.get("/owner")
def owner_analytics(db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)) -> dict:
    """Cross-branch comparison — admins only."""
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admins only")
    since = datetime.now(timezone.utc) - timedelta(days=30)
    out = []
    for b in db.scalars(select(Branch).order_by(Branch.name)):
        ps = list(db.scalars(select(Product).where(Product.branch == b.name)))
        rev = db.scalar(select(func.sum(Sale.total)).where(
            Sale.branch == b.name, Sale.sold_at >= since)) or 0.0
        pending = db.scalar(select(func.count(PurchaseOrder.id)).where(
            PurchaseOrder.branch == b.name,
            PurchaseOrder.status == "pending")) or 0
        near_expiry = 0
        for p in ps:
            d = _days_left(p)
            if d is not None and 0 <= d <= 120:
                near_expiry += 1
        out.append({
            "branch": b.name, "products": len(ps),
            "units": sum(p.quantity for p in ps),
            "inventoryValue": round(sum(p.quantity * p.purchase_price for p in ps), 2),
            "revenue30d": round(float(rev), 2),
            "lowStock": sum(1 for p in ps if 0 < p.quantity <= p.minimum_stock),
            "outOfStock": sum(1 for p in ps if p.quantity <= 0),
            "nearExpiry": near_expiry,
            "pendingOrders": int(pending),
        })
    return {
        "branches": out,
        "totals": {
            "revenue30d": round(sum(b["revenue30d"] for b in out), 2),
            "inventoryValue": round(sum(b["inventoryValue"] for b in out), 2),
            "products": sum(b["products"] for b in out),
            "lowStock": sum(b["lowStock"] for b in out),
        },
    }


@analytics.get("/supplier")
def supplier_analytics(db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)) -> dict:
    if user.role not in ("supplier", "admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Suppliers and admins only")
    stmt = select(PurchaseOrder)
    if user.role == "supplier" and not user.is_test_account:
        # A real distributor sees their own performance only — never the
        # whole platform's order book. (The seeded demo supplier keeps the
        # platform-wide view so the demo portal stays populated.)
        stmt = stmt.where(PurchaseOrder.supplier == user.name)
    agg: Dict[str, dict] = {}
    for o in db.scalars(stmt):
        a = agg.setdefault(o.supplier, {"supplier": o.supplier, "orders": 0,
                                        "revenue": 0.0, "delivered": 0, "pending": 0})
        a["orders"] += 1
        a["revenue"] += o.total_amount
        a["delivered"] += int(o.status == "delivered")
        a["pending"] += int(o.status == "pending")
    rows = []
    for a in agg.values():
        a["revenue"] = round(a["revenue"], 2)
        a["deliveryRate"] = round(a["delivered"] / a["orders"] * 100, 1) if a["orders"] else 0.0
        rows.append(a)
    rows.sort(key=lambda r: -r["revenue"])
    return {"suppliers": rows,
            "totalRevenue": round(sum(r["revenue"] for r in rows), 2),
            "totalOrders": sum(r["orders"] for r in rows)}
