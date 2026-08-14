"""Expiry-risk service."""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import numpy as np

from .registry import registry


def score_expiry(quantity: float, days_to_expiry: float, daily_sales_rate: float,
                 unit_cost: float = 0.0, moq: int = 1, volatility: float = 0.5,
                 zero_ratio: float = 0.3, chronic: int = 0,
                 seasonal_ok: int = 1) -> dict:
    expected_sales = max(daily_sales_rate, 0.0) * max(days_to_expiry, 0.0)
    unsold = max(0.0, quantity - expected_sales)

    row = {
        "qty_received": quantity, "moq": moq, "avg_daily_sales": daily_sales_rate,
        "days_to_expiry_at_receipt": days_to_expiry,
        "weeks_of_cover_ordered": quantity / max(daily_sales_rate * 7, 1e-6),
        "expected_sales": expected_sales, "cover_gap": quantity - expected_sales,
        "cover_ratio": expected_sales / max(quantity, 1), "sales_volatility": volatility,
        "zero_days_ratio": zero_ratio, "is_chronic": chronic,
        "unit_cost": unit_cost, "seasonal_alignment": seasonal_ok,
    }

    m = registry.expiry
    if m is None:
        # Transparent rule-based fallback — same logic the model learned, no model
        risk = 0.0 if quantity <= 0 else min(1.0, unsold / max(quantity, 1))
        model_name = "rule-based-fallback"
    else:
        X = np.array([[row[f] for f in m["features"]]], dtype=np.float32)
        risk = float(m["model"].predict(X)[0])
        model_name = "lightgbm-binary"

    action = ("RETURN_TO_SUPPLIER" if risk > 0.6
              else "PROMOTE" if risk > 0.25 else "MONITOR")
    level = "HIGH" if risk > 0.6 else "MEDIUM" if risk > 0.25 else "LOW"

    if action == "RETURN_TO_SUPPLIER":
        advice = "Return the batch to the supplier before it becomes a certain loss."
    elif action == "PROMOTE":
        advice = "Run a promotion or discount to clear it before expiry."
    else:
        advice = "The sales rate is enough to clear this quantity — monitor as usual."

    explanation = (
        f"You have {quantity:g} units with {days_to_expiry:g} days to expiry. "
        f"At {daily_sales_rate:g} units/day you would sell about "
        f"{expected_sales:.0f}, leaving {unsold:.0f} units worth "
        f"EGP {unsold * unit_cost:,.0f}. {advice}"
    )

    return {
        "risk_probability": round(risk, 3),
        "risk_level": level,
        "expected_leftover_units": round(unsold, 1),
        "expected_loss_egp": round(unsold * unit_cost, 2),
        "action": action,
        "explanation": explanation,
        "model": model_name,
    }
