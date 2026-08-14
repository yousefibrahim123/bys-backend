"""ML models exposed as API endpoints."""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..ml.expiry import score_expiry
from ..ml.forecast import explain, forecast_series, reorder_math, stockout_date
from ..ml.matcher import match_product
from ..ml.registry import registry
from ..models import Product, User
from ..schemas import (ExpiryRequest, ExpiryResponse, ForecastDay, ForecastRequest,
                       ForecastResponse, MatchCandidate, MatchRequest, MatchResponse,
                       ReorderLine, ReorderResponse)
from ..security import get_current_user, resolve_branch

router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.get("/status")
def ml_status() -> dict:
    """Per-model status — shows what loaded and what didn't, without failing."""
    return registry.status()


@router.post("/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest, db: Session = Depends(get_db),
             user: User = Depends(get_current_user)) -> ForecastResponse:
    p = db.get(Product, payload.product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    resolve_branch(user, p.branch)

    on_hand = payload.current_stock if payload.current_stock is not None else p.quantity
    r = forecast_series(p.ml_pharmacy_id, p.ml_product_id, payload.horizon_days,
                        p.daily_sales, on_hand)
    ro = reorder_math(r["total_p50"], r["total_p90"], on_hand, horizon=payload.horizon_days)

    return ForecastResponse(
        product_id=p.id, product_name=p.name, branch=p.branch,
        horizon_days=payload.horizon_days,
        daily=[ForecastDay(**d) for d in r["daily"]],
        total_p50=round(r["total_p50"], 1), total_p90=round(r["total_p90"], 1),
        safety_stock=ro["safety_stock"], reorder_point=ro["reorder_point"],
        current_stock=on_hand, recommended_order_qty=ro["order_qty"],
        days_of_cover=ro["days_of_cover"],
        stockout_date=stockout_date(r["daily"], on_hand),
        model=r["model"],
        explanation=explain(p.name, r["total_p50"], r["total_p90"], on_hand, ro,
                            payload.horizon_days, r.get("note", "")))


@router.post("/expiry-risk", response_model=ExpiryResponse)
def expiry_risk(payload: ExpiryRequest,
                user: User = Depends(get_current_user)) -> ExpiryResponse:
    if payload.quantity < 0 or payload.days_to_expiry < 0 or payload.daily_sales_rate < 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Values must be zero or greater")
    return ExpiryResponse(**score_expiry(
        quantity=payload.quantity, days_to_expiry=payload.days_to_expiry,
        daily_sales_rate=payload.daily_sales_rate, unit_cost=payload.unit_cost))


@router.get("/expiry-risk/product/{product_id}", response_model=ExpiryResponse)
def expiry_for_product(product_id: int, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)) -> ExpiryResponse:
    """Same model, but pulls the numbers from the product instead of manual input."""
    p = db.get(Product, product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    resolve_branch(user, p.branch)
    days = 0
    if p.expiry_date:
        try:
            days = max(0, (datetime.fromisoformat(p.expiry_date[:10]).date()
                           - datetime.now(timezone.utc).date()).days)
        except ValueError:
            days = 0
    return ExpiryResponse(**score_expiry(
        quantity=p.quantity, days_to_expiry=days,
        daily_sales_rate=p.daily_sales, unit_cost=p.purchase_price))


@router.post("/match", response_model=MatchResponse)
def match(payload: MatchRequest, user: User = Depends(get_current_user)) -> MatchResponse:
    if not payload.query.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Query text is required")
    r = match_product(payload.query, payload.limit, payload.catalog)
    if r["decision"] == "UNAVAILABLE":
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "The matcher model is not available right now")
    return MatchResponse(
        query=r["query"], decision=r["decision"], confidence=r["confidence"],
        best=MatchCandidate(**r["best"]) if r["best"] else None,
        alternatives=[MatchCandidate(**c) for c in r["alternatives"]],
        catalog=r["catalog"], catalog_size=r["catalog_size"])


@router.get("/match", response_model=MatchResponse)
def match_get(q: str = Query(..., min_length=1), limit: int = Query(5, ge=1, le=20),
              catalog: str = Query("real", pattern="^(real|synthetic)$"),
              user: User = Depends(get_current_user)) -> MatchResponse:
    """Same thing over GET — easier for the search box in the UI."""
    return match(MatchRequest(query=q, limit=limit, catalog=catalog), user)


@router.get("/reorder-list", response_model=ReorderResponse)
def reorder_list(branch: Optional[str] = None, limit: int = Query(25, ge=1, le=200),
                 db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> ReorderResponse:
    b = resolve_branch(user, branch)
    lines: List[ReorderLine] = []
    for p in db.scalars(select(Product).where(Product.branch == b)):
        r = forecast_series(p.ml_pharmacy_id, p.ml_product_id, 28,
                            p.daily_sales, p.quantity)
        ro = reorder_math(r["total_p50"], r["total_p90"], p.quantity)
        if not ro["should_order_now"] or ro["order_qty"] <= 0:
            continue
        cover = ro["days_of_cover"]
        lines.append(ReorderLine(
            product_id=p.id, name=p.name, supplier=p.supplier,
            current_stock=p.quantity, minimum_stock=p.minimum_stock,
            forecast_28d=round(r["total_p50"], 1),
            recommended_qty=ro["order_qty"],
            estimated_cost=round(ro["order_qty"] * p.purchase_price, 2),
            urgency="critical" if cover < 3 else "high" if cover < 7 else "normal",
            reason=(f"Stock of {p.quantity} covers only {cover:.0f} days; "
                    f"reorder point is {ro['reorder_point']:.0f}.")))

    lines.sort(key=lambda l: (l.current_stock / max(l.forecast_28d, 1e-6)))
    top = lines[:limit]
    return ReorderResponse(
        branch=b, generated_at=datetime.now(timezone.utc), lines=top,
        total_estimated_cost=round(sum(l.estimated_cost for l in lines), 2),
        model="lightgbm-quantile-p50/p90")
