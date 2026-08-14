"""
Recompute model metrics from the artefacts actually shipped.
============================================================
forecast_metrics.json and expiry_metrics.json arrived truncated from source
(214 and 78 bytes, cut off mid-value). This script recomputes the numbers from
the models themselves rather than copying figures from an unverified source.

Important note on forecasting: train_forecast.py only saves the fold-0 model.
Fold 0's test window is the last 28 days and training stops before it, so that
is a genuine holdout for the shipped model. Folds 1 and 2 use **older**
windows that are inside fold 0's training data, so scoring against them would
leak. We therefore measure fold 0 only, and say so explicitly.

    python recompute_metrics.py            # everything
    python recompute_metrics.py forecast   # forecasting only
    python recompute_metrics.py expiry     # expiry only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = HERE / "ml_assets" / "models"
DATA = HERE / "ml_assets" / "data"
OUT = MODELS / "verified_metrics.json"

HORIZON = 28
FOLD_LEN = 28
RAMADANS = [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30"),
            ("2026-02-18", "2026-03-19")]


def wape(y, p):
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))


def mase(y, p, naive_mae):
    return float(np.abs(y - p).mean() / max(naive_mae, 1e-9))


def bias(y, p):
    return float((p - y).sum() / max(y.sum(), 1e-9))



def build_features(df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Copied from train_forecast.py — must stay identical, line for line."""
    df = df.sort_values(["pharmacy_id", "product_id", "date"]).reset_index(drop=True)
    g = df.groupby(["pharmacy_id", "product_id"])["qty"]

    for lag in [HORIZON, HORIZON + 1, HORIZON + 7, HORIZON + 14, HORIZON + 28, HORIZON + 364]:
        df[f"lag_{lag}"] = g.shift(lag).astype(np.float32)

    base = g.shift(HORIZON)
    for w in (7, 14, 28, 91):
        df[f"rmean_{w}"] = base.rolling(w, min_periods=1).mean().astype(np.float32)
    df["rstd_28"] = base.rolling(28, min_periods=3).std().astype(np.float32)
    df["rmax_28"] = base.rolling(28, min_periods=1).max().astype(np.float32)
    df["zero_ratio_28"] = base.eq(0).rolling(28, min_periods=5).mean().astype(np.float32)
    df["zero_ratio_91"] = base.eq(0).rolling(91, min_periods=10).mean().astype(np.float32)
    df["trend_ratio"] = (df["rmean_7"] / (df["rmean_28"] + 0.1)).astype(np.float32)
    df["cv"] = (df["rstd_28"] / (df["rmean_28"] + 0.1)).astype(np.float32)

    d = df["date"]
    df["dow"] = d.dt.dayofweek.astype(np.int8)
    df["month"] = d.dt.month.astype(np.int8)
    df["day_of_month"] = d.dt.day.astype(np.int8)
    df["is_weekend"] = d.dt.dayofweek.isin([4, 5]).astype(np.int8)
    df["is_flu_season"] = d.dt.month.isin([11, 12, 1, 2]).astype(np.int8)
    df["is_allergy_season"] = d.dt.month.isin([3, 4, 5]).astype(np.int8)
    ram = np.zeros(len(df), dtype=np.int8)
    for a, b in RAMADANS:
        ram |= ((d >= a) & (d <= b)).values.astype(np.int8)
    df["is_ramadan"] = ram

    so = df.groupby(["pharmacy_id", "product_id"])["stock_out_flag"]
    df["stockout_rate_28"] = so.shift(HORIZON).rolling(28, min_periods=5).mean().astype(np.float32)

    cats = sorted(products["category"].unique())
    forms = sorted(products["form"].unique())
    meta = products.set_index("product_id")
    df["category_code"] = df["product_id"].map(
        meta["category"].map({c: i for i, c in enumerate(cats)})).astype(np.float32)
    df["form_code"] = df["product_id"].map(
        meta["form"].map({f: i for i, f in enumerate(forms)})).astype(np.float32)
    df["is_chronic"] = df["product_id"].map(meta["is_chronic"]).astype(np.float32)
    df["unit_price_base"] = df["product_id"].map(meta["unit_price"]).astype(np.float32)
    return df


def eval_forecast() -> dict:
    print("Reading sales ...", flush=True)
    sales = pd.read_csv(DATA / "sales_daily.csv", parse_dates=["date"],
                        usecols=["pharmacy_id", "product_id", "date", "qty",
                                 "on_promo", "stock_out_flag"])
    products = pd.read_csv(DATA / "products.csv")
    print(f"  {len(sales):,} rows · {sales.date.min().date()} -> {sales.date.max().date()}")

    print("Building features ...", flush=True)
    df = build_features(sales, products)
    df = df.dropna(subset=["rmean_28", "lag_28"]).reset_index(drop=True)
    print(f"  {len(df):,} usable rows")

    p50 = lgb.Booster(model_file=str(MODELS / "forecast_p50.txt"))
    p90 = lgb.Booster(model_file=str(MODELS / "forecast_p90.txt"))
    feats = p50.feature_name()

    max_date = df["date"].max()
    test_end = max_date                      # fold 0
    test_start = test_end - pd.Timedelta(days=FOLD_LEN - 1)
    te = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
    print(f"Test window (fold 0): {test_start.date()} -> {test_end.date()} "
          f"({len(te):,} points)")

    X = te[feats].to_numpy(np.float32)
    y = te["qty"].values.astype(float)
    pred50 = np.clip(p50.predict(X), 0, None)
    pred90 = np.clip(p90.predict(X), 0, None)

    naive = np.nan_to_num(te[f"lag_{HORIZON}"].values.astype(float))
    seasonal = np.nan_to_num(te[f"lag_{HORIZON + 7}"].values.astype(float))
    ma28 = np.nan_to_num(te["rmean_28"].values.astype(float))
    naive_mae = float(np.abs(y - naive).mean())

    baselines = {
        "naive_lag28": {"wape": wape(y, naive), "mase": 1.0},
        "seasonal_naive_lag35": {"wape": wape(y, seasonal),
                                 "mase": mase(y, seasonal, naive_mae)},
        "moving_avg_28": {"wape": wape(y, ma28), "mase": mase(y, ma28, naive_mae)},
    }
    best_base = min(v["wape"] for v in baselines.values())
    w50 = wape(y, pred50)

    res = {
        "model": "LightGBM Quantile (P50/P90)",
        "artifacts": ["forecast_p50.txt", "forecast_p90.txt"],
        "n_features": len(feats),
        "n_trees_p50": p50.num_trees(),
        "n_trees_p90": p90.num_trees(),
        "evaluation": {
            "method": "holdout — fold 0 window (last 28 days), outside the shipped model's training data",
            "test_window": f"{test_start.date()} → {test_end.date()}",
            "test_points": int(len(te)),
            "series": int(te.groupby(["pharmacy_id", "product_id"]).ngroups),
        },
        "p50": {"wape": round(w50, 4), "mase": round(mase(y, pred50, naive_mae), 3),
                "bias": round(bias(y, pred50), 4),
                "mae": round(float(np.abs(y - pred50).mean()), 3)},
        "p90": {"service_level": round(float((pred90 >= y).mean()), 4),
                "overstock_ratio": round(float(pred90.sum() / max(y.sum(), 1)), 3)},
        "baselines": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                      for k, v in baselines.items()},
        "best_baseline_wape": round(best_base, 4),
        "improvement_vs_best_baseline_pct": round(100 * (best_base - w50) / best_base, 2),
        "note": ("The shipped model is the fold-0 model. Folds 1 and 2 use older "
                 "windows that fall inside its training data, so scoring against "
                 "them would leak. For all 3 folds run "
                 "ml-training/train_forecast.py all, which trains an independent "
                 "model per fold."),
    }
    (MODELS / "forecast_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  WAPE={res['p50']['wape']}  MASE={res['p50']['mase']}  "
          f"service@P90={res['p90']['service_level']}  "
          f"gain={res['improvement_vs_best_baseline_pct']}%")
    return res



EXP_FEATURES = ["qty_received", "moq", "avg_daily_sales", "days_to_expiry_at_receipt",
                "weeks_of_cover_ordered", "expected_sales", "cover_gap", "cover_ratio",
                "sales_volatility", "zero_days_ratio", "is_chronic", "unit_cost",
                "seasonal_alignment"]


def eval_expiry() -> dict:
    """Copied from train_expiry_risk.py — same feature build, same split."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    batches = pd.read_csv(DATA / "batches.csv",
                          parse_dates=["received_date", "expiry_date"])
    products = pd.read_csv(DATA / "products.csv")
    sales = pd.read_csv(DATA / "sales_daily.csv", parse_dates=["date"],
                        usecols=["pharmacy_id", "product_id", "date", "qty"])

    stats = (sales.groupby(["pharmacy_id", "product_id"])["qty"]
             .agg(mean_qty="mean", std_qty="std",
                  zero_ratio=lambda s: float((s == 0).mean())).reset_index())
    df = batches.merge(stats, on=["pharmacy_id", "product_id"], how="left")
    df = df.merge(products[["product_id", "is_chronic", "cost_price", "seasonality"]],
                  on="product_id", how="left")

    df["expected_sales"] = df["avg_daily_sales"] * df["days_to_expiry_at_receipt"]
    df["cover_gap"] = df["qty_received"] - df["expected_sales"]
    df["cover_ratio"] = df["expected_sales"] / df["qty_received"].clip(lower=1)
    df["sales_volatility"] = (df["std_qty"] / df["mean_qty"].clip(lower=0.01)).fillna(0)
    df["zero_days_ratio"] = df["zero_ratio"].fillna(1.0)
    df["unit_cost"] = df["cost_price"]
    recv_month = df["received_date"].dt.month
    season_ok = np.where(
        df["seasonality"] == "winter", recv_month.isin([9, 10, 11, 12, 1]),
        np.where(df["seasonality"] == "spring", recv_month.isin([1, 2, 3, 4]),
                 np.where(df["seasonality"] == "summer", recv_month.isin([4, 5, 6, 7]), True)))
    df["seasonal_alignment"] = season_ok.astype(int)

    df = df.sort_values("received_date").reset_index(drop=True)
    y = df["expired_unsold"].values.astype(int)
    split = int(len(df) * 0.75)
    yte = y[split:]

    out = {
        "model": "LightGBM Binary",
        "n_batches": int(len(df)),
        "positive_rate": round(float(y.mean()), 4),
        "split": "temporal — oldest 75% train / newest 25% test",
        "test_batches": int(len(yte)),
        "test_positives": int(yte.sum()),
    }

    for name, fname in (("with_derived_features", "expiry_risk.txt"),
                        ("honest_no_leaky_features", "expiry_risk_honest.txt")):
        path = MODELS / fname
        if not path.exists():
            continue
        m = lgb.Booster(model_file=str(path))
        cols = m.feature_name()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            out[name] = {"error": f"missing features: {missing}"}
            continue
        pred = m.predict(df[cols].astype(np.float32).iloc[split:])
        out[name] = {
            "artifact": fname,
            "roc_auc": round(float(roc_auc_score(yte, pred)), 4),
            "pr_auc": round(float(average_precision_score(yte, pred)), 4),
            "n_features": len(cols),
            "features": cols,
        }
        print(f"  {name}: AUC={out[name]['roc_auc']} PR-AUC={out[name]['pr_auc']}")

    expected_rule = df["avg_daily_sales"] * df["days_to_expiry_at_receipt"]
    rule_score = (df["qty_received"] - expected_rule).values[split:]
    rule_pred = (expected_rule < df["qty_received"]).astype(int).values
    out["leakage_probe"] = {
        "one_line_rule": {
            "definition": "qty_received - (avg_daily_sales × days_to_expiry_at_receipt)",
            "roc_auc": round(float(roc_auc_score(yte, rule_score)), 4),
            "pr_auc": round(float(average_precision_score(yte, rule_score)), 4),
        },
        "coverage_ratio_column_alone": {
            "roc_auc": round(float(roc_auc_score(yte, -df["coverage_ratio"].values[split:])), 4),
        },
        "rule_label_agreement_pct": round(100 * float((rule_pred == y).mean()), 2),
        "disagreements": int((rule_pred != y).sum()),
        "verdict": ("The label is fully determined by the inputs via a one-line rule. "
                    "The model isn't learning any additional signal — it's recovering "
                    "the rule. The AUC here measures label reconstruction, not "
                    "predictive power."),
    }
    print(f"  leakage probe: one-line rule AUC="
          f"{out['leakage_probe']['one_line_rule']['roc_auc']} · "
          f"agreement {out['leakage_probe']['rule_label_agreement_pct']}%")

    out["leakage_warning"] = (
        "In the synthetic data the label is generated by a coverage rule, and "
        "cover_ratio/cover_gap encode that same rule. The \"honest\" variant is "
        "**not actually honest** — it still has qty_received, avg_daily_sales and "
        "days_to_expiry, from which the same rule can be rebuilt, which is why it "
        "scores almost the same AUC. Neither number indicates real performance. "
        "The only fix is labels from real inventory.")
    (MODELS / "expiry_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out



def load_matcher() -> dict:
    out = {}
    for tag, key in (("matcher_real", "real_catalog_25k"),
                     ("matcher", "synthetic_catalog_600")):
        p = MODELS / f"{tag}_metrics.json"
        if p.exists():
            try:
                out[key] = json.loads(p.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as e:
                out[key] = {"error": f"corrupt file: {e}"}
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    result: dict = {"generated_by": "recompute_metrics.py",
                    "source": "computed directly from the shipped models"}

    if which in ("all", "forecast"):
        print("\n=== Forecasting ===")
        result["forecast"] = eval_forecast()
    if which in ("all", "expiry"):
        print("\n=== Expiry risk ===")
        result["expiry"] = eval_expiry()
    if which in ("all", "matcher"):
        result["matcher"] = load_matcher()

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
