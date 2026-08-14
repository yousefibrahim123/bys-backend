"""
Drug-name matching service.

The normalisation and feature-extraction functions are copied verbatim from
train_matcher.py. Any small divergence here means inference features differ
from training features, and the model returns wrong results without raising
anything. Do not change these functions in isolation.

The Arabic character tables below are data processing, not UI: the Egyptian
drug catalogue (25,065 items) is in Arabic, so matching a name requires
normalising Arabic letter variants, digits, and unit abbreviations.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz

from .registry import registry

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")
NOISE = {"علبة", "عبوة", "شريط", "box", "strip", "pcs", "x", "*"}
UNITS = [(r"\bمجم\b|\bملجم\b|\bم\s*ج\b", "mg"), (r"\bجم\b|\bجرام\b", "g"),
         (r"\bمل\b", "ml"), (r"\bمكجم\b", "mcg")]
STRENGTH = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg)")
UNIT_SCALE = {"mg": 1, "g": 1000, "ml": 1, "mcg": 0.001}

FEATURE_NAMES = ["tfidf_cos", "token_set", "partial", "wratio", "token_sort",
                 "strength_match", "strength_missing", "len_diff", "token_overlap",
                 "numeric_overlap"]


def normalize(t: str) -> str:
    s = unicodedata.normalize("NFKC", str(t)).strip().lower().translate(AR_DIGITS)
    s = DIACRITICS.sub("", s)
    for a, b in [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه"),
                 ("ؤ", "و"), ("ئ", "ي")]:
        s = s.replace(a, b)
    for p, r in UNITS:
        s = re.sub(p, r, s)
    s = re.sub(r"(\d)\s*(mg|g|ml|mcg)", r"\1\2", s)
    s = re.sub(r"[^\w\u0600-\u06FF]+", " ", s)
    return " ".join(w for w in s.split() if w not in NOISE).strip()


def strength_of(s: str) -> Optional[float]:
    m = STRENGTH.search(s)
    if not m:
        return None
    return float(m.group(1)) * UNIT_SCALE[m.group(2)]


def _num_overlap(a: str, b: str) -> float:
    na = set(re.findall(r"\d+(?:\.\d+)?", a))
    nb = set(re.findall(r"\d+(?:\.\d+)?", b))
    if not na or not nb:
        return 0.5
    return len(na & nb) / len(na | nb)


def pair_features(raw: str, cand: str, cos: float,
                  raw_strength, cand_strength) -> List[float]:
    return [
        cos,
        fuzz.token_set_ratio(raw, cand) / 100,
        fuzz.partial_ratio(raw, cand) / 100,
        fuzz.WRatio(raw, cand) / 100,
        fuzz.token_sort_ratio(raw, cand) / 100,
        1.0 if (raw_strength is not None and cand_strength is not None
                and abs(raw_strength - cand_strength) < 1e-6) else 0.0,
        1.0 if (raw_strength is None or cand_strength is None) else 0.0,
        abs(len(raw) - len(cand)) / 40,
        len(set(raw.split()) & set(cand.split())) / max(1, len(set(raw.split()))),
        _num_overlap(raw, cand),
    ]


def match_product(query: str, limit: int = 5, catalog: str = "real") -> dict:
    m = registry.matcher(catalog)
    if m is None:
        return {"query": query, "decision": "UNAVAILABLE", "confidence": 0.0,
                "best": None, "alternatives": [], "catalog": catalog, "catalog_size": 0}

    n = normalize(query)
    if not n:
        return {"query": query, "decision": "REJECT", "confidence": 0.0,
                "best": None, "alternatives": [], "catalog": catalog,
                "catalog_size": m["size"]}

    q = m["vectorizer"].transform([n])
    denom = float(np.sqrt(q.multiply(q).sum())) or 1e-9
    q = q.multiply(1 / denom).tocsr()
    sims = np.asarray((q @ m["matrix"].T).todense()).ravel()

    k = min(m["top_k"], len(sims))
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    rs = strength_of(n)
    feats = np.array([pair_features(n, m["texts"][i], float(sims[i]), rs, m["strengths"][i])
                      for i in idx], dtype=np.float32)
    probs = m["clf"].predict_proba(m["scaler"].transform(feats))[:, 1]

    w = m["blend_weight"]
    blended = w * probs + (1 - w) * (sims[idx] / max(sims[idx].max(), 1e-9))
    order = np.argsort(-blended)

    cands = []
    for rank, o in enumerate(order[:max(limit, 1)]):
        pid = int(m["product_ids"][idx[o]])
        try:
            row = m["catalog"].loc[pid]
        except KeyError:
            continue
        cands.append({
            "product_id": pid,
            "name_ar": str(row.get("name_ar", "")),
            "name_en": str(row.get("name_en", "")),
            "confidence": round(float(probs[o]), 3),
            "price_egp": (float(row["price_egp"]) if "price_egp" in row.index
                          and row["price_egp"] == row["price_egp"] else None),
            "manufacturer": (str(row["manufacturer"]) if "manufacturer" in row.index
                             else None),
        })

    if not cands:
        return {"query": query, "decision": "REJECT", "confidence": 0.0, "best": None,
                "alternatives": [], "catalog": catalog, "catalog_size": m["size"]}

    conf = cands[0]["confidence"]
    th = m["thresholds"]
    decision = ("AUTO_ACCEPT" if conf >= th["auto"]
                else "NEEDS_REVIEW" if conf >= th["review"] else "REJECT")

    return {
        "query": query, "decision": decision, "confidence": conf,
        "best": cands[0], "alternatives": cands[1:],
        "catalog": catalog, "catalog_size": m["size"],
    }
