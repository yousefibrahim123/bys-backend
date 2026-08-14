"""
Proves the expiry actions really happen and permissions are really enforced.

    python test_actions_permissions.py

The point of every assertion here is state, not a response code: a price that
actually moved, stock that actually left one branch and arrived in another, a
return order that actually exists for the supplier to see.
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
os.environ["AI_REPLY_BUDGET_SECONDS"] = "1"      # keep the suite fast

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
res: list[tuple[bool, str]] = []


def check(name, cond, extra=""):
    res.append((bool(cond), name))
    print(f"  [{OK if cond else BAD}] {name}{('  -> ' + str(extra)) if extra else ''}")


def section(t):
    print(f"\n--- {t} ---")


with TestClient(app) as c:
    from sqlalchemy import select
    from app.database import SessionLocal
    from app.models import User
    from app.security import hash_password

    with SessionLocal() as s:
        if not s.scalar(select(User).where(User.email == "giza@gmail.com")):
            s.add(User(email="giza@gmail.com", phone="01012345678", name="Giza Branch",
                       role="branch", branch="Giza", password_hash=hash_password("123123"),
                       organization_type="pharmacy", email_verified=True, is_active=True))
        if not s.scalar(select(User).where(User.email == "cairo@gmail.com")):
            s.add(User(email="cairo@gmail.com", phone="01023456789", name="Cairo Branch",
                       role="branch", branch="Cairo", password_hash=hash_password("123123"),
                       organization_type="pharmacy", email_verified=True, is_active=True))
        if not s.scalar(select(User).where(User.email == "supplier@gmail.com")):
            s.add(User(email="supplier@gmail.com", phone="01056789012", name="Supplier",
                       role="supplier", branch=None, password_hash=hash_password("123123"),
                       organization_type="distributor", email_verified=True, is_active=True))
        s.commit()

    adm = c.post("/api/auth/login",
                 json={"email": "admin@gmail.com", "password": "123123"}).json()
    A = {"Authorization": f"Bearer {adm['accessToken']}"}
    giz = c.post("/api/auth/login",
                 json={"email": "giza@gmail.com", "password": "123123"}).json()
    G = {"Authorization": f"Bearer {giz['accessToken']}"}

    products = c.get("/api/products/all?branch=Giza", headers=G).json()
    prod = products[0]
    pid = prod["id"]

    # ------------------------------------------------------------- discount
    section("Discount actually changes the price")
    before = c.get(f"/api/products/{pid}", headers=G).json()
    r = c.post("/api/expiry/discount", headers=G,
               json={"productId": pid, "percent": 15, "allowBelowCost": True})
    check("discount 201", r.status_code == 201, r.status_code)
    act = r.json()
    after = c.get(f"/api/products/{pid}", headers=G).json()
    expected = round(before["sellingPrice"] * 0.85, 2)
    check("selling price really dropped 15%",
          abs(after["sellingPrice"] - expected) < 0.02,
          f"{before['sellingPrice']} -> {after['sellingPrice']}")
    check("the change is not just a message",
          after["sellingPrice"] != before["sellingPrice"])
    check("an action record exists",
          any(a["id"] == act["id"]
              for a in c.get("/api/expiry/actions?branch=Giza", headers=G).json()))
    check("percent out of range rejected",
          c.post("/api/expiry/discount", headers=G,
                 json={"productId": pid, "percent": 99}).status_code == 422)

    r = c.post(f"/api/expiry/discount/{act['id']}/revert", headers=G)
    check("revert 200", r.status_code == 200, r.status_code)
    restored = c.get(f"/api/products/{pid}", headers=G).json()
    check("original price restored",
          abs(restored["sellingPrice"] - before["sellingPrice"]) < 0.02,
          f"{restored['sellingPrice']} vs {before['sellingPrice']}")
    check("cannot revert twice",
          c.post(f"/api/expiry/discount/{act['id']}/revert",
                 headers=G).status_code == 409)

    section("Below-cost discount needs an explicit override")
    check("blocked without the override",
          c.post("/api/expiry/discount", headers=G,
                 json={"productId": pid, "percent": 90}).status_code == 422)

    # ------------------------------------------------------------- transfer
    section("Transfer actually moves stock between branches")
    src = c.get(f"/api/products/{pid}", headers=G).json()
    qty = min(5, src["quantity"])
    r = c.post("/api/expiry/transfer", headers=A,
               json={"productId": pid, "quantity": qty, "toBranch": "Cairo"})
    check("transfer 201", r.status_code == 201, r.text[:120])
    t_id = r.json().get("reference_id") or r.json().get("referenceId")
    if t_id:
        c.put(f"/api/transfers/{t_id}/accept", headers=A)
    src_after = c.get(f"/api/products/{pid}", headers=G).json()
    check("source branch stock decreased",
          src_after["quantity"] == src["quantity"] - qty,
          f"{src['quantity']} -> {src_after['quantity']}")
    cairo = c.get("/api/products/all?branch=Cairo", headers=A).json()
    match = [p for p in cairo if p["sku"] == src["sku"]]
    check("destination branch has the units", bool(match),
          match[0]["quantity"] if match else "not found")
    check("destination was notified",
          any("Transfer" in n["title"]
              for n in c.get("/api/notifications?branch=Cairo", headers=A).json()))
    check("transferring more than stock is rejected",
          c.post("/api/expiry/transfer", headers=A,
                 json={"productId": pid, "quantity": 99999,
                       "toBranch": "Cairo"}).status_code == 422)
    check("unknown branch rejected",
          c.post("/api/expiry/transfer", headers=A,
                 json={"productId": pid, "quantity": 1,
                       "toBranch": "Atlantis"}).status_code == 404)
    check("same-branch transfer rejected",
          c.post("/api/expiry/transfer", headers=A,
                 json={"productId": pid, "quantity": 1,
                       "toBranch": "Giza"}).status_code == 422)

    # --------------------------------------------------------------- return
    section("Return really reaches the supplier")
    before_orders = len(c.get("/api/purchase-orders", headers=A).json())
    src = c.get(f"/api/products/{pid}", headers=G).json()
    rqty = min(3, src["quantity"])
    r = c.post("/api/expiry/return", headers=G,
               json={"productId": pid, "quantity": rqty,
                     "reason": "expires in 12 days"})
    check("return 201", r.status_code == 201, r.text[:140])
    orders = c.get("/api/purchase-orders", headers=A).json()
    check("a purchase order was actually raised",
          len(orders) == before_orders + 1, f"{before_orders} -> {len(orders)}")
    ret = [o for o in orders if str(o.get("orderNumber", "")).startswith("RET-")]
    check("it is recorded as a return", bool(ret),
          ret[0]["orderNumber"] if ret else "none")
    check("the return total is negative",
          ret and ret[0]["totalAmount"] < 0,
          ret[0]["totalAmount"] if ret else "")
    after = c.get(f"/api/products/{pid}", headers=G).json()
    check("returned units left stock",
          after["quantity"] == src["quantity"] - rqty,
          f"{src['quantity']} -> {after['quantity']}")
    notes = c.get("/api/notifications?branch=Giza", headers=G).json()
    check("a return notification was raised",
          any("Return request" in n["title"] for n in notes))

    section("Supplier sees the request on their own dashboard")
    sup = c.post("/api/auth/login",
                 json={"email": "supplier@gmail.com", "password": "123123"}).json()
    S = {"Authorization": f"Bearer {sup['accessToken']}"}
    sup_orders = c.get("/api/purchase-orders", headers=S).json()
    check("supplier can list orders", isinstance(sup_orders, list),
          len(sup_orders))

    # ------------------------------------------------------------ write-off
    section("Write-off removes stock and records the loss")
    src = c.get(f"/api/products/{pid}", headers=G).json()
    r = c.post("/api/expiry/write-off", headers=G,
               json={"productId": pid, "quantity": 2, "reason": "expired"})
    check("write-off 201", r.status_code == 201, r.status_code)
    check("stock really reduced",
          c.get(f"/api/products/{pid}", headers=G).json()["quantity"]
          == src["quantity"] - 2)
    check("loss value recorded", r.json()["valueEgp"] > 0, r.json()["valueEgp"])

    section("Branch isolation still holds")
    cairo_prod = [p for p in c.get("/api/products/all?branch=Cairo",
                                   headers=A).json()][0]
    check("a Giza user cannot discount a Cairo product",
          c.post("/api/expiry/discount", headers=G,
                 json={"productId": cairo_prod["id"],
                       "percent": 10}).status_code == 403)

    # ---------------------------------------------------------- permissions
    section("Permissions are stored and enforced")
    r = c.get("/api/users/permissions/catalog?targetRole=branch", headers=A)
    check("catalog 200", r.status_code == 200)
    cat = r.json()
    check("catalog is grouped", len(cat["groups"]) >= 8, len(cat["groups"]))
    check("an admin can grant everything",
          all(p["allowed"] for g in cat["groups"] for p in g["permissions"]))

    inv = c.post("/api/users/invite", headers=A, json={
        "email": "limited@gmail.com", "name": "Limited User", "role": "branch",
        "branch": "Giza",
        "permissions": ["view_dashboard", "view_inventory", "view_orders"]}).json()
    uid = inv["user"]["id"]
    got = c.get(f"/api/users/{uid}/permissions", headers=A).json()["permissions"]
    check("only the ticked permissions were stored",
          set(got) == {"view_dashboard", "view_inventory", "view_orders"}, got)

    section("You cannot grant what you do not hold")
    c.put(f"/api/users/{uid}/permissions", headers=A,
          json={"permissions": ["view_dashboard", "view_users", "add_users"]})
    itok = inv["activationUrl"].split("token=")[1]
    c.post("/api/users/activate", json={"token": itok, "password": "Limited123"})
    lim = c.post("/api/auth/login",
                 json={"email": "limited@gmail.com",
                       "password": "Limited123"}).json()
    L = {"Authorization": f"Bearer {lim['accessToken']}"}

    cat2 = c.get("/api/users/permissions/catalog?targetRole=branch",
                 headers=L).json()
    allowed = set(cat2["grantable"])
    check("their catalog is limited to what they hold",
          "delete_users" not in allowed and "view_dashboard" in allowed,
          sorted(allowed))
    check("delete_users is not offered to them",
          not any(p["allowed"] for g in cat2["groups"]
                  for p in g["permissions"] if p["key"] == "delete_users"))

    inv2 = c.post("/api/users/invite", headers=L, json={
        "email": "escalate@gmail.com", "name": "Escalation Test",
        "role": "branch", "branch": "Giza",
        "permissions": ["view_dashboard", "delete_users", "billing_settings"]})
    check("invite still succeeds", inv2.status_code == 201, inv2.status_code)
    uid2 = inv2.json()["user"]["id"]
    got2 = set(c.get(f"/api/users/{uid2}/permissions",
                     headers=A).json()["permissions"])
    check("escalated permissions were dropped",
          "delete_users" not in got2 and "billing_settings" not in got2, sorted(got2))
    check("legitimate ones survived", "view_dashboard" in got2)

    r = c.put(f"/api/users/{uid2}/permissions", headers=L,
              json={"permissions": ["view_dashboard", "delete_users"]})
    check("PUT reports what it refused",
          r.status_code == 200 and "delete_users" in r.json()["rejected"],
          r.json().get("rejected"))
    check("an admin always holds everything",
          set(c.get(f"/api/users/{adm['user']['id']}/permissions",
                    headers=A).json()["permissions"]) ==
          set(c.get("/api/users/permissions/catalog?targetRole=admin",
                    headers=A).json()["mine"]))
    check("you cannot edit your own permissions",
          c.put(f"/api/users/{adm['user']['id']}/permissions", headers=A,
                json={"permissions": []}).status_code == 422)

shutil.rmtree(_tmp, ignore_errors=True)
passed = sum(1 for ok, _ in res if ok)
print(f"\n{'=' * 56}\n{passed}/{len(res)} passed")
for ok, n in res:
    if not ok:
        print(f"  FAILED: {n}")
sys.exit(0 if passed == len(res) else 1)
