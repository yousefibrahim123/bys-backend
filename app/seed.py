"""
Database seeding.

Loads the same data the frontend ships (176 products across 4 branches) so the
UI stays identical when it switches from localStorage to the API.

It also builds a "bridge" between each app product and a training time series:
without it the forecaster has no history to work from. Matching is by nearest
daily sales rate and stored in two columns on the product. Once there's enough
real sales history (90+ days) the bridge is dropped and the model trains on
the real data.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import ML_ASSETS
from .database import Base, SessionLocal, engine
from .models import (Branch, Category, Notification, Product, PurchaseOrder,
                     PurchaseOrderItem, Sale, Store, Supplier, User)
from .security import hash_password

log = logging.getLogger("seed")
SEED_JSON = ML_ASSETS / "data" / "seed_products.json"
SERIES_CACHE = ML_ASSETS / "data" / "series_cache.parquet"

BRANCH_KEYS = {"giza": "Giza", "cairo": "Cairo", "nascity": "Nasr City", "alex": "Alexandria"}

DEMO_USERS = [
    ("admin@gmail.com", "01067890123", "123123", "admin", "Multi-Branch Admin", None),
]


def _snake(p: dict) -> dict:
    ai = p.get("aiPredictionData") or {}
    return dict(
        sku=p.get("sku") or f"SKU-{p['id']}", barcode=p.get("barcode", ""),
        name=p["name"], category=p.get("category", ""), supplier=p.get("supplier", ""),
        branch=p["branch"],
        purchase_price=float(p.get("purchasePrice", 0)),
        selling_price=float(p.get("sellingPrice", p.get("price", 0))),
        quantity=int(p.get("quantity", p.get("qty", 0))),
        minimum_stock=int(p.get("minimumStock", 0)),
        maximum_stock=int(p.get("maximumStock", p.get("maxQty", 0))),
        expiry_date=p.get("expiryDate", p.get("expiry", "")),
        batch_number=p.get("batchNumber", ""),
        daily_sales=float(p.get("dailySales", 0)),
        weekly_sales=float(p.get("weeklySales", 0)),
        monthly_sales=float(p.get("monthlySales", 0)),
        predicted_demand=float(ai.get("predictedDemand", 0)),
        confidence_score=float(ai.get("confidenceScore", 0)),
        recommended_action=ai.get("recommendedAction", ""),
        expected_stockout_date=ai.get("expectedStockoutDate", ""),
        risk_level=p.get("riskLevel", "low"), status=p.get("status", "active"),
        inventory_status=p.get("inventoryStatus", "normal"),
        last_restock_date=p.get("lastRestockDate", ""),
        last_updated_date=p.get("lastUpdatedDate", ""),
    )


def compute_inventory_status(p: Product) -> str:
    """Inventory status derived from the numbers — not stored and forgotten."""
    today = datetime.now(timezone.utc).date()
    if p.quantity <= 0:
        return "out_of_stock"
    try:
        if p.expiry_date:
            days = (datetime.fromisoformat(p.expiry_date[:10]).date() - today).days
            if days <= 90:
                return "expiring"
    except ValueError:
        pass
    if p.quantity <= p.minimum_stock:
        return "low"
    if p.daily_sales < 0.15 and p.quantity > max(p.minimum_stock, 1) * 2:
        return "stale"
    return "normal"


def _build_series_bridge(db: Session) -> int:
    """
    Picks the nearest time series by sales rate for each product and writes a
    trimmed copy (only what's needed) to parquet, so startup doesn't read 62 MB.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas unavailable — skipped building the time-series bridge")
        return 0

    sales_csv = ML_ASSETS / "data" / "sales_daily.csv"
    if not sales_csv.exists():
        log.warning("sales_daily.csv missing — the forecaster will use its fallback")
        return 0

    log.info("building ML series bridge ...")
    sales = pd.read_csv(sales_csv, parse_dates=["date"],
                        usecols=["pharmacy_id", "product_id", "date", "qty",
                                 "stock_out_flag"])

    # Mean daily sales over the last 90 days per series + series length
    cutoff = sales["date"].max() - pd.Timedelta(days=90)
    recent = sales[sales["date"] >= cutoff]
    stats = (recent.groupby(["pharmacy_id", "product_id"])["qty"]
             .agg(["mean", "count"]).reset_index())
    lengths = sales.groupby(["pharmacy_id", "product_id"]).size().rename("n_days")
    stats = stats.merge(lengths, on=["pharmacy_id", "product_id"])
    stats = stats[stats["n_days"] >= 400].reset_index(drop=True)   # series long enough to be useful
    if stats.empty:
        log.warning("no series long enough — skipped")
        return 0

    means = stats["mean"].to_numpy()
    products = db.scalars(select(Product)).all()

    used: Set[Tuple[int, int]] = set()
    chosen: List[Tuple[int, int]] = []
    for p in sorted(products, key=lambda x: x.id):
        target = max(float(p.daily_sales), 0.0)
        order = abs(means - target).argsort()
        pick = None
        for oi in order[:80]:                      # nearest one not yet taken
            key = (int(stats.at[oi, "pharmacy_id"]), int(stats.at[oi, "product_id"]))
            if key not in used:
                pick = key
                break
        if pick is None:
            oi = int(order[0])
            pick = (int(stats.at[oi, "pharmacy_id"]), int(stats.at[oi, "product_id"]))
        used.add(pick)
        p.ml_pharmacy_id, p.ml_product_id = pick
        chosen.append(pick)

    db.commit()

    keys = pd.DataFrame(sorted(used), columns=["pharmacy_id", "product_id"])
    subset = sales.merge(keys, on=["pharmacy_id", "product_id"], how="inner")
    SERIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(SERIES_CACHE, index=False)
    log.info("series cache: %d series · %d rows · %.1f MB",
             len(keys), len(subset), SERIES_CACHE.stat().st_size / 1e6)
    return len(keys)


def _seed_sales_history(db: Session, products: List[Product], days: int = 60) -> int:
    """Production ready: no fake sales history is generated."""
    return 0


def _seed_orders_and_alerts(db: Session, products: List[Product]) -> None:
    # Production ready: no fake orders or fake notifications are seeded
    pass


def _ensure_test_accounts(db: Session) -> int:
    """
    The predefined test accounts sign in directly (no verification email, no
    OTP). They carry an explicit server-side flag — the bypass is never decided
    by email address alone or by frontend logic.
    """
    created_or_fixed = 0
    for email, phone, pwd, role, name, branch in DEMO_USERS:
        u = db.scalar(select(User).where(User.email == email))
        if u is None:
            db.add(User(email=email, phone=phone, name=name, role=role,
                        branch=branch, password_hash=hash_password(pwd),
                        organization_type=("distributor" if role == "supplier"
                                           else "pharmacy"),
                        email_verified=True, is_active=True,
                        is_test_account=True))
            created_or_fixed += 1
        elif not (u.email_verified and u.is_active and u.is_test_account):
            u.email_verified = True
            u.is_active = True
            u.is_test_account = True
            created_or_fixed += 1
    if created_or_fixed:
        db.commit()
    return created_or_fixed


def seed(force: bool = False) -> dict:
    """
    Production databases start empty: no demo products, suppliers, branches,
    orders or notifications. The only seeded rows are the predefined test
    accounts (explicitly flagged as such). Set SEED_DEMO_DATA=true to load the
    demo catalogue for local development.
    """
    from .config import settings

    Base.metadata.create_all(engine)
    db: Session = SessionLocal()
    try:
        if not force and db.scalar(select(func.count(User.id))):
            fixed = _ensure_test_accounts(db)
            return {"skipped": True, "reason": "database already populated",
                    "test_accounts_ensured": fixed}

        if force:
            for m in (Sale, PurchaseOrderItem, PurchaseOrder, Notification,
                      Product, Branch, Store, User, Supplier, Category):
                db.query(m).delete()
            db.commit()

        _ensure_test_accounts(db)

        if not settings.seed_demo_data:
            return {"skipped": False, "demo_data": False,
                    "users": len(DEMO_USERS),
                    "note": "clean database — no demo data seeded"}

        # ---- Everything below is demo/sample data, loaded only on opt-in.
        store = Store(name="BYS Pharmacy Group")
        db.add(store)
        db.flush()

        for name in BRANCH_KEYS.values():
            db.add(Branch(store_id=store.id, name=name,
                          address=f"{name}, Egypt", phone="+20 100 000 0000"))
        db.commit()

        raw = json.loads(Path(SEED_JSON).read_text(encoding="utf-8"))
        branches = {b.name: b.id for b in db.scalars(select(Branch)).all()}

        cats: Set[str] = set()
        sups: Set[str] = set()
        created: List[Product] = []
        for key, items in raw.items():
            bname = BRANCH_KEYS.get(key, key)
            for item in items:
                item["branch"] = bname
                p = Product(**_snake(item))
                p.branch_id = branches.get(bname)
                p.inventory_status = compute_inventory_status(p)
                db.add(p)
                created.append(p)
                if p.category:
                    cats.add(p.category)
                if p.supplier:
                    sups.add(p.supplier)
        db.commit()

        db.add_all([Category(name=c) for c in sorted(cats)])
        rng = random.Random(3)
        db.add_all([Supplier(name=s, contact_email=f"orders@{s.split()[0].lower()}.example",
                             phone="+20 111 000 0000",
                             lead_time_days=rng.randint(1, 7),
                             reliability_score=round(rng.uniform(0.72, 0.98), 2),
                             min_order_value=rng.choice([0, 500, 1000, 2500]))
                    for s in sorted(sups)])
        db.commit()

        n_sales = _seed_sales_history(db, created)
        _seed_orders_and_alerts(db, created)
        n_series = _build_series_bridge(db)

        return {
            "skipped": False, "demo_data": True,
            "branches": len(BRANCH_KEYS), "users": len(DEMO_USERS),
            "products": len(created), "categories": len(cats), "suppliers": len(sups),
            "sales_rows": n_sales,
            "purchase_orders": db.scalar(select(func.count(PurchaseOrder.id))),
            "notifications": db.scalar(select(func.count(Notification.id))),
            "ml_series_linked": n_series,
        }
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    print(json.dumps(seed(force="--force" in sys.argv), indent=2, ensure_ascii=False))
