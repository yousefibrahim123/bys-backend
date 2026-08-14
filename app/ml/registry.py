"""
ML model registry.

Each model loads once on first use (lazily) and stays in memory. If a model
is missing, the registry keeps working and everything else runs normally — it
reports `available=False` for that one instead of taking the whole app down.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal, Any

from ..config import ML_ASSETS

log = logging.getLogger("ml.registry")

MODELS_DIR = ML_ASSETS / "models"
DATA_DIR = ML_ASSETS / "data"
SERIES_CACHE = DATA_DIR / "series_cache.parquet"

HORIZON = 28
RAMADANS = [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30"),
            ("2026-02-18", "2026-03-19")]

EXPIRY_FEATURES = [
    "qty_received", "moq", "avg_daily_sales", "days_to_expiry_at_receipt",
    "weeks_of_cover_ordered", "expected_sales", "cover_gap", "cover_ratio",
    "sales_volatility", "zero_days_ratio", "is_chronic", "unit_cost",
    "seasonal_alignment",
]


class _Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._forecast: Optional[Dict[str, Any]] = None
        self._expiry: Optional[Dict[str, Any]] = None
        self._matcher: Dict[str, Dict[str, Any]] = {}
        self._series = None
        self._errors: Dict[str, str] = {}

    # ------------------------------------------------------------- forecasting
    @property
    def forecast(self) -> Optional[Dict[str, Any]]:
        if self._forecast is None and "forecast" not in self._errors:
            with self._lock:
                if self._forecast is None and "forecast" not in self._errors:
                    try:
                        import lightgbm as lgb
                        import pandas as pd
                        p50 = lgb.Booster(model_file=str(MODELS_DIR / "forecast_p50.txt"))
                        p90 = lgb.Booster(model_file=str(MODELS_DIR / "forecast_p90.txt"))
                        products = pd.read_csv(DATA_DIR / "products.csv")
                        self._forecast = {
                            "p50": p50, "p90": p90,
                            "features": p50.feature_name(),
                            "products": products.set_index("product_id"),
                            "categories": sorted(products["category"].unique()),
                            "forms": sorted(products["form"].unique()),
                            "metrics": _metrics(MODELS_DIR / "forecast_metrics.json", "forecast"),
                        }
                        log.info("forecast models loaded (%d features)",
                                 len(self._forecast["features"]))
                    except Exception as e:  # pragma: no cover
                        self._errors["forecast"] = str(e)
                        log.error("forecast load failed: %s", e)
        return self._forecast

    # ------------------------------------------------------------ sales history
    @property
    def series(self):
        """Cache of only the needed series — not the whole 62 MB file."""
        if self._series is None and "series" not in self._errors:
            with self._lock:
                if self._series is None and "series" not in self._errors:
                    try:
                        import pandas as pd
                        if SERIES_CACHE.exists():
                            df = pd.read_parquet(SERIES_CACHE)
                        else:
                            log.warning("series cache missing — reading the full sales_daily.csv")
                            df = pd.read_csv(
                                DATA_DIR / "sales_daily.csv", parse_dates=["date"],
                                usecols=["pharmacy_id", "product_id", "date", "qty",
                                         "stock_out_flag"])
                        df["date"] = pd.to_datetime(df["date"])
                        self._series = df.sort_values("date")
                        log.info("series cache loaded: %d rows", len(df))
                    except Exception as e:  # pragma: no cover
                        self._errors["series"] = str(e)
                        log.error("series load failed: %s", e)
        return self._series

    # ------------------------------------------------------------------ expiry
    @property
    def expiry(self) -> Optional[Dict[str, Any]]:
        if self._expiry is None and "expiry" not in self._errors:
            with self._lock:
                if self._expiry is None and "expiry" not in self._errors:
                    try:
                        import lightgbm as lgb
                        self._expiry = {
                            "model": lgb.Booster(model_file=str(MODELS_DIR / "expiry_risk.txt")),
                            "features": EXPIRY_FEATURES,
                            "metrics": _metrics(MODELS_DIR / "expiry_metrics.json", "expiry"),
                        }
                        log.info("expiry model loaded")
                    except Exception as e:  # pragma: no cover
                        self._errors["expiry"] = str(e)
                        log.error("expiry load failed: %s", e)
        return self._expiry

    # ----------------------------------------------------------------- matcher
    def matcher(self, catalog: str = "real") -> Optional[Dict[str, Any]]:
        key = "real" if catalog == "real" else "synthetic"
        if key in self._matcher:
            return self._matcher[key]
        if f"matcher_{key}" in self._errors:
            return None
        with self._lock:
            if key in self._matcher:
                return self._matcher[key]
            try:
                import joblib
                import pandas as pd
                tag = "matcher_real" if key == "real" else "matcher"
                csv = "products_real.csv" if key == "real" else "products.csv"
                r = joblib.load(MODELS_DIR / f"{tag}_tfidf.joblib")
                rr = joblib.load(MODELS_DIR / f"{tag}_reranker.joblib")
                cat = pd.read_csv(DATA_DIR / csv).set_index("product_id")
                self._matcher[key] = {
                    "vectorizer": r["vectorizer"], "matrix": r["catalog_matrix"],
                    "product_ids": r["product_ids"], "texts": r["catalog_texts"],
                    "strengths": r["catalog_strengths"],
                    "scaler": rr["scaler"], "clf": rr["model"],
                    "thresholds": rr["thresholds"],
                    "blend_weight": float(rr.get("blend_weight", 0.5)),
                    "top_k": int(rr.get("top_k", 15)),
                    "catalog": cat, "size": len(cat),
                    "metrics": _load_json(MODELS_DIR / f"{tag}_metrics.json"),
                }
                log.info("matcher '%s' loaded (%d products)", key, len(cat))
                return self._matcher[key]
            except Exception as e:  # pragma: no cover
                self._errors[f"matcher_{key}"] = str(e)
                log.error("matcher %s load failed: %s", key, e)
                return None

    # ------------------------------------------------------------------ status
    def status(self) -> Dict[str, Any]:
        return {
            "forecast": {
                "available": self.forecast is not None,
                "error": self._errors.get("forecast"),
                "metrics": (self.forecast or {}).get("metrics", {}),
            },
            "expiry": {
                "available": self.expiry is not None,
                "error": self._errors.get("expiry"),
                "metrics": (self.expiry or {}).get("metrics", {}),
            },
            "matcher_real": _matcher_status(self.matcher("real"),
                                            self._errors.get("matcher_real")),
            "matcher_synthetic": _matcher_status(self.matcher("synthetic"),
                                                 self._errors.get("matcher_synthetic")),
            "series_cache": {
                "available": self.series is not None,
                "rows": 0 if self.series is None else int(len(self.series)),
                "error": self._errors.get("series"),
            },
        }

    def warmup(self) -> None:
        """Preload at startup so the first request isn't slow."""
        _ = self.forecast, self.expiry, self.series
        self.matcher("real")


def _matcher_status(m: Optional[dict], err: Optional[str]) -> dict:
    if m is None:
        return {"available": False, "error": err}
    return {"available": True, "catalog_size": m["size"],
            "top1_accuracy": m.get("metrics", {}).get(
                "hybrid_ensemble", {}).get("top1_accuracy"),
            "thresholds": m["thresholds"]}


VERIFIED = MODELS_DIR / "verified_metrics.json"


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("metrics file unreadable (%s): %s", p.name, e)
        return {}


def _metrics(primary: Path, verified_key: str) -> dict:
    """
forecast_metrics.json and expiry_metrics.json arrived **truncated** from
    source (214 and 78 bytes, cut off mid-value), so json.loads raised on them
    and they returned {} — which is why /api/ml/status showed empty metrics
    even though the models themselves load fine.

    verified_metrics.json is computed from the shipped models directly by
    recompute_metrics.py, so we use it as the fallback rather than show
    nothing.
    """
    m = _load_json(primary)
    if m:
        return m
    v = _load_json(VERIFIED).get(verified_key, {})
    if v:
        v = {**v, "source": "verified_metrics.json (original file arrived truncated)"}
    return v


registry = _Registry()
