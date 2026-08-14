"""
AI assistant tools.

Every number in an assistant reply must originate here — from the database or
from the models. The LLM picks the tool and phrases the answer, but never
invents a figure.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal, Any, Callable

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..models import (Customer, Product, PurchaseOrder, Sale, SupplierListing,
                      TransferRequest)
from ..ml.expiry import score_expiry
from ..ml.forecast import forecast_series, reorder_math, stockout_date

# Tool descriptions handed to the LLM so it can choose
TOOL_SPECS = [
    {"name": "inventory_overview",
     "description": "Inventory summary: item count, value, low stock, out of stock, near expiry",
     "args": {}},
    {"name": "low_stock",
     "description": "Products at or near their minimum level that will run out soon",
     "args": {"limit": "count (default 10)"}},
    {"name": "near_expiry",
     "description": "Products nearing expiry, with expected loss from the expiry model",
     "args": {"days": "days (default 120)", "limit": "count (default 10)"}},
    {"name": "top_products",
     "description": "Top products by profit, revenue, or units sold",
     "args": {"metric": "profit|revenue|units", "limit": "count (default 5)"}},
    {"name": "forecast_product",
     "description": "28-day demand forecast for one product + recommended order qty",
     "args": {"product_name": "product name or part of it"}},
    {"name": "reorder_list",
     "description": "Suggested reorder list for the whole branch, based on the forecast",
     "args": {"limit": "count (default 10)"}},
    {"name": "sales_summary",
     "description": "Sales summary for a recent period: revenue, transactions, average basket",
     "args": {"days": "days (default 30)"}},
    {"name": "supplier_performance",
     "description": "Supplier performance: order count, value, delivery rate",
     "args": {}},
    {"name": "search_product",
     "description": "Search the branch inventory for a product by name",
     "args": {"query": "search text"}},
    {"name": "orders_summary",
     "description": "Purchase orders: counts by status (pending/approved/shipped/"
                    "delivered/rejected/cancelled), total value, and the latest orders",
     "args": {"limit": "recent orders to include (default 5)"}},
    {"name": "transfers_summary",
     "description": "Stock transfer requests involving this branch: pending/accepted/"
                    "rejected counts and the latest requests",
     "args": {"limit": "recent transfers to include (default 5)"}},
    {"name": "customers_summary",
     "description": "Branch customers: how many, and top customers by spending",
     "args": {"limit": "top customers to include (default 5)"}},
    {"name": "supplier_offers",
     "description": "Medicines published by suppliers on the platform, including "
                    "active discount offers — searchable by drug name",
     "args": {"query": "drug name or part of it (optional)"}},
]


def _products(db: Session, branch: str) -> List[Product]:
    return list(db.scalars(select(Product).where(Product.branch == branch)))


def _days_to_expiry(p: Product) -> Optional[int]:
    if not p.expiry_date:
        return None
    try:
        d = datetime.fromisoformat(p.expiry_date[:10]).date()
        return (d - datetime.now(timezone.utc).date()).days
    except ValueError:
        return None


# -------------------------------------------------------------------- tools

def inventory_overview(db: Session, branch: str, **_) -> dict:
    ps = _products(db, branch)
    cost = sum(p.quantity * p.purchase_price for p in ps)
    retail = sum(p.quantity * p.selling_price for p in ps)
    near = [p for p in ps if (d := _days_to_expiry(p)) is not None and 0 <= d <= 120]
    return {
        "branch": branch,
        "total_products": len(ps),
        "total_units": sum(p.quantity for p in ps),
        "inventory_value_egp": round(cost, 2),
        "retail_value_egp": round(retail, 2),
        "potential_profit_egp": round(retail - cost, 2),
        "low_stock_count": sum(1 for p in ps if 0 < p.quantity <= p.minimum_stock),
        "out_of_stock_count": sum(1 for p in ps if p.quantity <= 0),
        "near_expiry_count": len(near),
        "categories": len({p.category for p in ps if p.category}),
        "suppliers": len({p.supplier for p in ps if p.supplier}),
    }


def low_stock(db: Session, branch: str, limit: int = 10, **_) -> dict:
    ps = [p for p in _products(db, branch) if p.quantity <= p.minimum_stock]
    ps.sort(key=lambda p: (p.quantity - p.minimum_stock,
                           -(p.daily_sales or 0)))
    rows = []
    for p in ps[:int(limit)]:
        rate = max(p.daily_sales, 0.01)
        rows.append({
            "name": p.name, "current_stock": p.quantity,
            "minimum_stock": p.minimum_stock, "daily_sales": p.daily_sales,
            "days_until_empty": round(p.quantity / rate, 1),
            "supplier": p.supplier,
        })
    return {"branch": branch, "count": len(ps), "items": rows}


def near_expiry(db: Session, branch: str, days: int = 120, limit: int = 10, **_) -> dict:
    rows = []
    for p in _products(db, branch):
        d = _days_to_expiry(p)
        if d is None or d > int(days):
            continue
        risk = score_expiry(quantity=p.quantity, days_to_expiry=max(d, 0),
                            daily_sales_rate=p.daily_sales, unit_cost=p.purchase_price)
        rows.append({
            "name": p.name, "quantity": p.quantity, "days_to_expiry": d,
            "expiry_date": p.expiry_date,
            "risk_probability": risk["risk_probability"],
            "risk_level": risk["risk_level"],
            "expected_loss_egp": risk["expected_loss_egp"],
            "action": risk["action"],
        })
    rows.sort(key=lambda r: -r["expected_loss_egp"])
    return {"branch": branch, "count": len(rows), "items": rows[:int(limit)],
            "total_value_at_risk_egp": round(sum(r["expected_loss_egp"] for r in rows), 2),
            "model": "lightgbm-binary"}


def top_products(db: Session, branch: str, metric: str = "profit",
                 limit: int = 5, **_) -> dict:
    ps = _products(db, branch)

    def key(p: Product) -> float:
        if metric == "revenue":
            return p.monthly_sales * p.selling_price
        if metric == "units":
            return p.monthly_sales
        return p.monthly_sales * (p.selling_price - p.purchase_price)

    ps.sort(key=key, reverse=True)
    rows = []
    for p in ps[:int(limit)]:
        margin = ((p.selling_price - p.purchase_price) / p.selling_price * 100
                  if p.selling_price else 0)
        rows.append({
            "name": p.name,
            "monthly_units": p.monthly_sales,
            "monthly_revenue_egp": round(p.monthly_sales * p.selling_price, 2),
            "monthly_profit_egp": round(p.monthly_sales *
                                        (p.selling_price - p.purchase_price), 2),
            "margin_pct": round(margin, 1),
        })
    return {"branch": branch, "metric": metric, "items": rows}


def search_product(db: Session, branch: str, query: str = "", **_) -> dict:
    q = (query or "").strip().lower()
    if not q:
        return {"branch": branch, "items": []}
    ps = [p for p in _products(db, branch) if q in p.name.lower()
          or q in (p.category or "").lower() or q in (p.sku or "").lower()]
    return {"branch": branch, "count": len(ps), "items": [
        {"id": p.id, "name": p.name, "quantity": p.quantity,
         "selling_price": p.selling_price, "category": p.category,
         "expiry_date": p.expiry_date, "status": p.inventory_status}
        for p in ps[:10]]}


def forecast_product(db: Session, branch: str, product_name: str = "", **_) -> dict:
    q = (product_name or "").strip().lower()
    ps = _products(db, branch)
    match = next((p for p in ps if q and q in p.name.lower()), None)
    if match is None:
        return {"error": f"No product with that name in the {branch} branch",
                "available_examples": [p.name for p in ps[:5]]}

    r = forecast_series(match.ml_pharmacy_id, match.ml_product_id, 28,
                        match.daily_sales, match.quantity)
    ro = reorder_math(r["total_p50"], r["total_p90"], match.quantity)
    return {
        "product": match.name, "branch": branch, "current_stock": match.quantity,
        "forecast_28d_p50": round(r["total_p50"], 1),
        "forecast_28d_p90": round(r["total_p90"], 1),
        "daily_average": ro["daily_p50"],
        "reorder_point": ro["reorder_point"],
        "safety_stock": ro["safety_stock"],
        "recommended_order_qty": ro["order_qty"],
        "days_of_cover": ro["days_of_cover"],
        "should_order_now": ro["should_order_now"],
        "expected_stockout_date": stockout_date(r["daily"], match.quantity),
        "model": r["model"],
        "note": r.get("note", ""),
    }


def reorder_list(db: Session, branch: str, limit: int = 10, **_) -> dict:
    rows = []
    for p in _products(db, branch):
        r = forecast_series(p.ml_pharmacy_id, p.ml_product_id, 28,
                            p.daily_sales, p.quantity)
        ro = reorder_math(r["total_p50"], r["total_p90"], p.quantity)
        if not ro["should_order_now"] or ro["order_qty"] <= 0:
            continue
        cover = ro["days_of_cover"]
        rows.append({
            "name": p.name, "supplier": p.supplier, "current_stock": p.quantity,
            "forecast_28d": round(r["total_p50"], 1),
            "recommended_qty": ro["order_qty"],
            "estimated_cost_egp": round(ro["order_qty"] * p.purchase_price, 2),
            "days_of_cover": cover,
            "urgency": "critical" if cover < 3 else "high" if cover < 7 else "normal",
        })
    rows.sort(key=lambda r: r["days_of_cover"])
    top = rows[:int(limit)]
    return {"branch": branch, "count": len(rows), "items": top,
            "total_estimated_cost_egp": round(
                sum(r["estimated_cost_egp"] for r in rows), 2),
            "model": "lightgbm-quantile-p50/p90"}


def sales_summary(db: Session, branch: str, days: int = 30, **_) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=int(days))
    rows = db.execute(
        select(func.count(Sale.id), func.sum(Sale.total), func.sum(Sale.quantity))
        .where(Sale.branch == branch, Sale.sold_at >= since)).one()
    n, revenue, units = rows[0] or 0, float(rows[1] or 0), int(rows[2] or 0)

    top = db.execute(
        select(Product.name, func.sum(Sale.total).label("rev"),
               func.sum(Sale.quantity).label("units"))
        .join(Product, Product.id == Sale.product_id)
        .where(Sale.branch == branch, Sale.sold_at >= since)
        .group_by(Product.name).order_by(func.sum(Sale.total).desc()).limit(5)).all()

    return {
        "branch": branch, "days": int(days), "transactions": n,
        "total_revenue_egp": round(revenue, 2), "total_units": units,
        "average_basket_egp": round(revenue / n, 2) if n else 0,
        "daily_average_egp": round(revenue / max(int(days), 1), 2),
        "top_products": [{"name": t[0], "revenue_egp": round(float(t[1]), 2),
                          "units": int(t[2])} for t in top],
    }


def supplier_performance(db: Session, branch: str, **_) -> dict:
    orders = list(db.scalars(select(PurchaseOrder).where(PurchaseOrder.branch == branch)))
    agg: Dict[str, Dict[str, Any]] = {}
    for o in orders:
        a = agg.setdefault(o.supplier, {"supplier": o.supplier, "orders": 0,
                                        "total_egp": 0.0, "delivered": 0})
        a["orders"] += 1
        a["total_egp"] += o.total_amount
        a["delivered"] += int(o.status == "delivered")
    rows = []
    for a in agg.values():
        a["total_egp"] = round(a["total_egp"], 2)
        a["delivery_rate_pct"] = round(a["delivered"] / a["orders"] * 100, 1) if a["orders"] else 0
        rows.append(a)
    rows.sort(key=lambda r: -r["total_egp"])
    return {"branch": branch, "suppliers": rows}


def orders_summary(db: Session, branch: str, limit: int = 5, **_) -> dict:
    orders = list(db.scalars(select(PurchaseOrder)
                             .where(PurchaseOrder.branch == branch)
                             .order_by(PurchaseOrder.created_at.desc())))
    by_status: Dict[str, int] = {}
    for o in orders:
        by_status[o.status] = by_status.get(o.status, 0) + 1
    return {
        "branch": branch, "total_orders": len(orders),
        "by_status": by_status,
        "total_value_egp": round(sum(o.total_amount for o in orders), 2),
        "open_value_egp": round(sum(o.total_amount for o in orders
                                    if o.status in ("pending", "approved", "shipped")), 2),
        "recent": [{
            "order_number": o.order_number, "supplier": o.supplier,
            "status": o.status, "total_egp": round(o.total_amount, 2),
            "created": o.created_at.date().isoformat(),
        } for o in orders[:int(limit)]],
    }


def transfers_summary(db: Session, branch: str, limit: int = 5, **_) -> dict:
    rows = list(db.scalars(select(TransferRequest)
                           .where(or_(TransferRequest.from_branch == branch,
                                      TransferRequest.to_branch == branch))
                           .order_by(TransferRequest.created_at.desc())))
    by_status: Dict[str, int] = {}
    for t in rows:
        by_status[t.status] = by_status.get(t.status, 0) + 1
    return {
        "branch": branch, "total_transfers": len(rows), "by_status": by_status,
        "recent": [{
            "product": t.product_name, "quantity": t.quantity,
            "from": t.from_branch, "to": t.to_branch, "status": t.status,
            "created": t.created_at.date().isoformat(),
        } for t in rows[:int(limit)]],
    }


def customers_summary(db: Session, branch: str, limit: int = 5, **_) -> dict:
    total = db.scalar(select(func.count(Customer.id))
                      .where(Customer.branch == branch)) or 0
    top = db.execute(
        select(Sale.customer_name, func.sum(Sale.total).label("spent"),
               func.count(Sale.id).label("purchases"))
        .where(Sale.branch == branch, Sale.customer_name != "")
        .group_by(Sale.customer_name)
        .order_by(func.sum(Sale.total).desc()).limit(int(limit))).all()
    return {
        "branch": branch, "total_customers": int(total),
        "top_customers": [{"name": t[0], "total_spent_egp": round(float(t[1]), 2),
                           "purchases": int(t[2])} for t in top],
    }


def supplier_offers(db: Session, branch: str, query: str = "", **_) -> dict:
    stmt = select(SupplierListing).where(SupplierListing.is_active.is_(True))
    q = (query or "").strip()
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(SupplierListing.product_name.ilike(like),
                              SupplierListing.category.ilike(like),
                              SupplierListing.supplier_name.ilike(like)))
    rows = list(db.scalars(stmt.order_by(
        SupplierListing.offer_percent.desc(),
        SupplierListing.product_name).limit(15)))
    now = datetime.now(timezone.utc)

    def active_offer(l: SupplierListing) -> bool:
        if l.offer_percent <= 0:
            return False
        return l.offer_until is None or l.offer_until.replace(tzinfo=timezone.utc) >= now

    return {"query": q, "count": len(rows), "items": [{
        "product": l.product_name, "supplier": l.supplier_name,
        "category": l.category, "unit_price_egp": l.unit_price,
        "offer_percent": l.offer_percent if active_offer(l) else 0,
        "price_after_offer_egp": (round(l.unit_price * (1 - l.offer_percent / 100), 2)
                                  if active_offer(l) else l.unit_price),
        "available_qty": l.available_qty, "min_order_qty": l.min_order_qty,
    } for l in rows]}


TOOLS: Dict[str, Callable[..., dict]] = {
    "inventory_overview": inventory_overview,
    "low_stock": low_stock,
    "near_expiry": near_expiry,
    "top_products": top_products,
    "forecast_product": forecast_product,
    "reorder_list": reorder_list,
    "sales_summary": sales_summary,
    "supplier_performance": supplier_performance,
    "search_product": search_product,
    "orders_summary": orders_summary,
    "transfers_summary": transfers_summary,
    "customers_summary": customers_summary,
    "supplier_offers": supplier_offers,
}


def run_tool(name: str, db: Session, branch: str, args: Optional[dict] = None) -> dict:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(db, branch, **(args or {}))
    except TypeError:
        # The model passed an unknown argument — run with defaults instead of failing
        return fn(db, branch)
    except Exception as e:  # pragma: no cover
        return {"error": f"Tool {name} failed: {type(e).__name__}: {e}"}
