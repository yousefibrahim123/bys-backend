"""
Demand forecasting service.

Mirrors the feature-building logic in predict.py exactly. If features are
built differently from training, the model returns wrong numbers without
raising anything — so order and names are pinned to `booster.feature_name()`.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .registry import HORIZON, RAMADANS, registry


class ForecastUnavailable(RuntimeError):
    pass


def _feature_row(hist: pd.Series, target: pd.Timestamp, meta: dict, promo: int) -> dict:
    def lag(d: int) -> float:
        return float(hist.get(target - pd.Timedelta(days=d), np.nan))

    base_end = target - pd.Timedelta(days=HORIZON)
    win = hist[hist.index <= base_end]
    f: Dict[str, float] = {}
    for L in [HORIZON, HORIZON + 1, HORIZON + 7, HORIZON + 14, HORIZON + 28, HORIZON + 364]:
        f[f"lag_{L}"] = lag(L)
    for w in (7, 14, 28, 91):
        f[f"rmean_{w}"] = float(win.tail(w).mean()) if len(win) else np.nan
    f["rstd_28"] = float(win.tail(28).std()) if len(win) >= 3 else np.nan
    f["rmax_28"] = float(win.tail(28).max()) if len(win) else np.nan
    f["zero_ratio_28"] = float((win.tail(28) == 0).mean()) if len(win) >= 5 else np.nan
    f["zero_ratio_91"] = float((win.tail(91) == 0).mean()) if len(win) >= 10 else np.nan
    f["trend_ratio"] = f["rmean_7"] / (f["rmean_28"] + 0.1)
    f["cv"] = f["rstd_28"] / (f["rmean_28"] + 0.1)
    f["on_promo"] = promo
    f["dow"] = target.dayofweek
    f["month"] = target.month
    f["day_of_month"] = target.day
    f["is_weekend"] = int(target.dayofweek in (4, 5))
    f["is_flu_season"] = int(target.month in (11, 12, 1, 2))
    f["is_allergy_season"] = int(target.month in (3, 4, 5))
    f["is_ramadan"] = int(any(pd.Timestamp(s) <= target <= pd.Timestamp(e)
                              for s, e in RAMADANS))
    f["stockout_rate_28"] = meta["stockout_rate"]
    f.update(meta["static"])
    return f


def reorder_math(total_p50: float, total_p90: float, on_hand: float,
                 lead_time_days: int = 2, review_days: int = 7,
                 horizon: int = HORIZON) -> dict:
    """
    Safety stock = the gap between P90 and P50 over the (lead time + review)
    window. This derives safety from the data's own distribution rather than a
    z-score formula that assumes normality — and pharmacy sales are not
    normally distributed.
    """
    d50, d90 = total_p50 / horizon, total_p90 / horizon
    window = lead_time_days + review_days
    safety = max(0.0, (d90 - d50) * window)
    rop = d50 * window + safety
    return {
        "reorder_point": round(rop, 1),
        "safety_stock": round(safety, 1),
        "order_qty": int(np.ceil(max(0.0, d50 * horizon + safety - on_hand))),
        "days_of_cover": round(on_hand / max(d50, 1e-6), 1),
        "should_order_now": bool(on_hand <= rop),
        "daily_p50": round(d50, 3),
    }


def _fallback(daily_sales: float, horizon: int, on_hand: float) -> dict:
    """
    If the product has no linked time series (a new product, say) we fall back
    to its recorded sales rate with conservative variance, and say so
    explicitly in the response.
    """
    d50 = max(float(daily_sales), 0.0)
    d90 = d50 * 1.6 + 0.5
    today = date.today()
    daily = [{"date": (today + timedelta(days=i + 1)).isoformat(),
              "p50": round(d50, 2), "p90": round(d90, 2)} for i in range(horizon)]
    t50, t90 = d50 * horizon, d90 * horizon
    return {"daily": daily, "total_p50": t50, "total_p90": t90,
            "model": "moving-average-fallback",
            "note": "Not enough sales history for this product — used its recorded sales rate."}


def forecast_series(pharmacy_id: Optional[int], series_product_id: Optional[int],
                    horizon: int = HORIZON, daily_sales_hint: float = 0.0,
                    on_hand: float = 0.0) -> dict:
    fc = registry.forecast
    sales = registry.series

    if fc is None or sales is None or pharmacy_id is None or series_product_id is None:
        return _fallback(daily_sales_hint, horizon, on_hand)

    g = sales[(sales.pharmacy_id == pharmacy_id)
              & (sales.product_id == series_product_id)].sort_values("date")
    if g.empty or len(g) < 120:
        return _fallback(daily_sales_hint, horizon, on_hand)

    hist = pd.Series(g["qty"].values.astype(float), index=pd.DatetimeIndex(g["date"].values))
    last = pd.Timestamp(g["date"].max())

    try:
        p = fc["products"].loc[series_product_id]
    except KeyError:
        return _fallback(daily_sales_hint, horizon, on_hand)

    meta = {
        "stockout_rate": float(g["stock_out_flag"].tail(28).mean())
        if "stock_out_flag" in g.columns else 0.0,
        "static": {
            "category_code": fc["categories"].index(p["category"])
            if p["category"] in fc["categories"] else 0,
            "form_code": fc["forms"].index(p["form"]) if p["form"] in fc["forms"] else 0,
            "is_chronic": int(p["is_chronic"]),
            "unit_price_base": float(p["unit_price"]),
            "pharmacy_id": pharmacy_id,
        },
    }

    rows = [_feature_row(hist, last + pd.Timedelta(days=d), meta, 0)
            for d in range(1, horizon + 1)]
    X = pd.DataFrame(rows).reindex(columns=fc["features"]).to_numpy(np.float32)
    p50 = np.clip(fc["p50"].predict(X), 0, None)
    p90 = np.clip(fc["p90"].predict(X), 0, None)

    today = date.today()
    daily = [{"date": (today + timedelta(days=i + 1)).isoformat(),
              "p50": round(float(a), 2), "p90": round(float(b), 2)}
             for i, (a, b) in enumerate(zip(p50, p90))]
    return {"daily": daily, "total_p50": float(p50.sum()), "total_p90": float(p90.sum()),
            "model": "lightgbm-quantile-p50/p90", "note": ""}


def stockout_date(daily: List[dict], on_hand: float) -> Optional[str]:
    running = on_hand
    for d in daily:
        running -= d["p50"]
        if running <= 0:
            return d["date"]
    return None


def explain(name: str, total_p50: float, total_p90: float, on_hand: float,
            ro: dict, horizon: int, note: str) -> str:
    parts = [
        f"Forecast for \"{name}\" over {horizon} days: "
        f"{total_p50:.0f} units (likely), {total_p90:.0f} units (safe scenario).",
        f"Current stock of {on_hand:.0f} units covers about "
        f"{ro['days_of_cover']:.0f} days.",
    ]
    if ro["should_order_now"]:
        parts.append(
            f"Stock is below the reorder point ({ro['reorder_point']:.0f}) — "
            f"order {ro['order_qty']} units, of which {ro['safety_stock']:.0f} "
            f"is safety stock.")
    else:
        parts.append(
            f"Stock is above the reorder point ({ro['reorder_point']:.0f}) — "
            f"no need to order right now.")
    if note:
        parts.append(note)
    return " ".join(parts)
