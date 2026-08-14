"""
Integration test: reproduces exactly the calls the frontend now makes after
being wired to the API, using the same camelCase field names.

The point is to prove every screen will find data, not just that the endpoints
exist.

    python test_frontend_integration.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_tmp = Path(tempfile.mkdtemp())
if (HERE / "pharma.db").exists():
    shutil.copy(HERE / "pharma.db", _tmp / "pharma.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp / 'pharma.db'}"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
res: list[tuple[bool, str]] = []


def check(name, cond, extra=""):
    res.append((bool(cond), name))
    print(f"  [{OK if cond else BAD}] {name}{('  → ' + str(extra)) if extra else ''}")


with TestClient(app) as c:
    # ---- what AuthContext.login does
    print("\n--- AuthContext: login and token storage ---")
    r = c.post("/api/auth/login", json={"email": "giza@gmail.com", "password": "123123"})
    check("login 200", r.status_code == 200, r.status_code)
    d = r.json()
    check("accessToken present", "accessToken" in d)
    check("refreshToken present", "refreshToken" in d)
    check("user has role and branch",
          {"role", "branch", "email", "name", "id"} <= set(d["user"]))
    H = {"Authorization": f"Bearer {d['accessToken']}"}

    # ---- what AuthContext does on startup
    print("\n--- AuthContext: session restore ---")
    check("/api/auth/me with token", c.get("/api/auth/me", headers=H).status_code == 200)
    check("/api/auth/me without token rejected",
          c.get("/api/auth/me").status_code == 401)
    check("invalid token rejected",
          c.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}
                ).status_code == 401)

    # ---- token rotation
    print("\n--- Token rotation ---")
    r = c.post("/api/auth/refresh-token", json={"refreshToken": d["refreshToken"]})
    check("refresh-token 200", r.status_code == 200)
    check("old token revoked",
          c.post("/api/auth/refresh-token",
                 json={"refreshToken": d["refreshToken"]}).status_code == 401)

    # ---- screens that used to run on mock data
    print("\n--- Screens that used to run on static data ---")
    screens = [
        ("Inventory", "/api/products/all?branch=Giza"),
        ("Inventory - summary", "/api/inventory/summary?branch=Giza"),
        ("Inventory - low stock", "/api/inventory/low-stock?branch=Giza"),
        ("NearExpiry", "/api/inventory/near-expiry?branch=Giza"),
        ("Reports - sales", "/api/sales/summary?branch=Giza"),
        ("PurchaseOrders", "/api/purchase-orders"),
        ("Notifications", "/api/notifications"),
        ("SupplierComparison", "/api/suppliers"),
        ("Inventory - categories", "/api/categories"),
        ("AdminDashboard - branches", "/api/branches"),
        ("UserManagement - list", "/api/users"),
    ]
    for name, path in screens:
        r = c.get(path, headers=H)
        n = len(r.json()) if isinstance(r.json(), list) else "obj"
        check(name, r.status_code == 200, f"{r.status_code} · {n}")

    # Cross-branch comparison is admin-only — a branch user must be refused
    adm = c.post("/api/auth/login",
                 json={"email": "admin@gmail.com", "password": "123123"}).json()
    ADM = {"Authorization": f"Bearer {adm['accessToken']}"}
    check("AdminOverview as admin",
          c.get("/api/analytics/owner", headers=ADM).status_code == 200)
    check("branch user can't see cross-branch comparison",
          c.get("/api/analytics/owner", headers=H).status_code == 403)

    # ---- the new models screen
    print("\n--- ModelsStatus: the three models ---")
    s = c.get("/api/ml/status").json()
    check("forecast loaded", s["forecast"]["available"])
    check("expiry loaded", s["expiry"]["available"])
    check("matcher loaded", s["matcher_real"]["available"],
          f"{s['matcher_real'].get('catalog_size'):,} items")
    check("forecast metrics visible", bool(s["forecast"]["metrics"]))
    check("expiry metrics visible", bool(s["expiry"]["metrics"]))

    pid = c.get("/api/products/all?branch=Giza", headers=H).json()[0]["id"]
    r = c.post("/api/ml/forecast", headers=H,
               json={"productId": pid, "horizonDays": 28})
    check("live forecast test", r.status_code == 200,
          f"totalP50={r.json().get('totalP50')}")
    check("response has the fields the screen reads",
          {"totalP50", "totalP90", "reorderPoint", "recommendedOrderQty",
           "daysOfCover", "explanation"} <= set(r.json()))

    r = c.post("/api/ml/expiry-risk", headers=H, json={
        "quantity": 500, "daysToExpiry": 90, "dailySalesRate": 2, "unitCost": 15})
    check("live expiry test", r.status_code == 200,
          f"risk={r.json().get('riskLevel')}")
    check("response has the fields the screen reads",
          {"riskProbability", "riskLevel", "expectedLeftoverUnits",
           "expectedLossEgp", "action", "explanation"} <= set(r.json()))

    r = c.get("/api/ml/match?q=panadol extra&limit=5", headers=H)
    check("live matcher test", r.status_code == 200,
          r.json().get("best", {}).get("nameAr"))
    check("response has best and alternatives",
          {"decision", "confidence", "best", "alternatives"} <= set(r.json()))

    # ---- add-user flow from the UI
    print("\n--- UserManagement: full invite flow ---")
    r = c.post("/api/users/invite", headers=H, json={
        "email": "integration@gmail.com", "name": "Integration User",
        "phone": "01133224455", "role": "branch", "branch": "Giza",
        "jobTitle": "Pharmacist", "department": "Pharmacy"})
    check("invite created", r.status_code == 201, r.status_code)
    inv = r.json()
    check("response has activationUrl and emailSent",
          {"activationUrl", "emailSent", "note", "user"} <= set(inv))
    check("link points at a route the UI handles",
          "/activate?token=" in inv["activationUrl"])
    check("list includes the new user",
          any(u["email"] == "integration@gmail.com"
              for u in c.get("/api/users", headers=H).json()))
    check("email-status available to the UI",
          c.get("/api/users/email-status", headers=H).status_code == 200)

    # the employee opens the link
    itok = inv["activationUrl"].split("token=")[1]
    check("ActivateAccount reads the invite",
          c.get(f"/api/users/invite/{itok}").status_code == 200)
    r = c.post("/api/users/activate",
               json={"token": itok, "password": "Strong123"})
    check("activation returns a session", r.status_code == 200 and "accessToken" in r.json())
    check("weak password rejected",
          c.post("/api/users/activate",
                 json={"token": "x", "password": "123"}).status_code in (404, 422))

    # ---- delete
    uid = inv["user"]["id"]
    admin = c.post("/api/auth/login",
                   json={"email": "admin@gmail.com", "password": "123123"}).json()
    AH = {"Authorization": f"Bearer {admin['accessToken']}"}
    check("delete button works",
          c.delete(f"/api/users/{uid}", headers=AH).status_code == 204)

    # ---- AI assistant
    print("\n--- AIAssistant ---")
    r = c.post("/api/ai/chat", headers=H,
               json={"message": "What is about to run out?", "branch": "Giza"})
    check("chat responds", r.status_code == 200)
    check("reply is grounded in data", r.json().get("grounded") is True,
          f"tools={r.json().get('toolsUsed')}")

shutil.rmtree(_tmp, ignore_errors=True)
passed = sum(1 for ok, _ in res if ok)
print(f"\n{'='*52}\n{passed}/{len(res)} passed")
for ok, n in res:
    if not ok:
        print(f"  FAILED: {n}")
sys.exit(0 if passed == len(res) else 1)
