"""
Covers the endpoints that were missing from the API, plus the chat-history
ordering bug that made the assistant repeat and extend its previous answer.

    python test_backend_complete.py

Runs against a temporary copy of the database.
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
    print(f"  [{OK if cond else BAD}] {name}{('  -> ' + str(extra)) if extra else ''}")


def section(t):
    print(f"\n--- {t} ---")


with TestClient(app) as c:
    from sqlalchemy import select
    from app.database import SessionLocal
    from app.models import EmailVerification, User
    from app.security import hash_password
    with SessionLocal() as s:
        if not s.scalar(select(User).where(User.email == "giza@gmail.com")):
            s.add(User(email="giza@gmail.com", phone="01012345678", name="Giza Branch",
                       role="branch", branch="Giza", password_hash=hash_password("123123"),
                       organization_type="pharmacy", email_verified=True, is_active=True))
        s.query(User).filter(User.email.in_(["pwtest@gmail.com", "pw2@gmail.com", "forgot@gmail.com", "inv@gmail.com", "custtest@gmail.com"])).delete(synchronize_session=False)
        s.commit()

    admin = c.post("/api/auth/login",
                   json={"email": "admin@gmail.com", "password": "123123"}).json()
    A = {"Authorization": f"Bearer {admin['accessToken']}"}
    giza = c.post("/api/auth/login",
                  json={"email": "giza@gmail.com", "password": "123123"}).json()
    G = {"Authorization": f"Bearer {giza['accessToken']}"}

    # ------------------------------------------------------------- passwords
    section("Change password (Settings form had no endpoint)")
    with SessionLocal() as s:
        old_ids = [u.id for u in s.scalars(select(User).where(User.email.in_(["pwtest@gmail.com", "pw2@gmail.com"])))]
        if old_ids:
            s.query(EmailVerification).filter(EmailVerification.user_id.in_(old_ids)).delete(synchronize_session=False)
            s.query(User).filter(User.id.in_(old_ids)).delete(synchronize_session=False)
            s.commit()
    r1 = c.post("/api/auth/register", json={
        "email": "pwtest@gmail.com", "password": "Initial123",
        "name": "Password Tester", "role": "branch", "branch": "Giza"}).json()
    r2 = c.post("/api/auth/register", json={
        "email": "pw2@gmail.com", "password": "Initial123", "name": "PW2"}).json()
    with SessionLocal() as s:
        u = s.scalar(select(User).where(User.email == "pwtest@gmail.com"))
        ev = s.scalar(select(EmailVerification).where(EmailVerification.user_id == u.id).order_by(EmailVerification.id.desc()))
        vtok = ev.token
    c.post("/api/auth/verify-email", json={"token": vtok})
    pw = c.post("/api/auth/login",
                json={"email": "pwtest@gmail.com", "password": "Initial123"}).json()
    P = {"Authorization": f"Bearer {pw['accessToken']}"}

    check("wrong current password rejected",
          c.post("/api/auth/change-password", headers=P, json={
              "currentPassword": "nope", "newPassword": "Newpass123"}
          ).status_code == 401)
    check("weak new password rejected",
          c.post("/api/auth/change-password", headers=P, json={
              "currentPassword": "Initial123", "newPassword": "abc"}
          ).status_code == 422)
    check("reusing the same password rejected",
          c.post("/api/auth/change-password", headers=P, json={
              "currentPassword": "Initial123", "newPassword": "Initial123"}
          ).status_code == 422)
    check("change password succeeds",
          c.post("/api/auth/change-password", headers=P, json={
              "currentPassword": "Initial123", "newPassword": "Newpass123"}
          ).status_code == 200)
    check("old password no longer works",
          c.post("/api/auth/login", json={
              "email": "pwtest@gmail.com", "password": "Initial123"}
          ).status_code == 401)
    check("new password works",
          c.post("/api/auth/login", json={
              "email": "pwtest@gmail.com", "password": "Newpass123"}
          ).status_code == 200)
    check("old refresh token was revoked",
          c.post("/api/auth/refresh-token",
                 json={"refreshToken": pw["refreshToken"]}).status_code == 401)

    section("Forgot / reset password")
    r = c.post("/api/auth/forgot-password", json={"email": "pwtest@gmail.com"})
    check("forgot-password 200", r.status_code == 200)
    rurl = r.json().get("verificationUrl")
    if not rurl:
        from app.models import PasswordReset
        with SessionLocal() as s:
            u = s.scalar(select(User).where(User.email == "pwtest@gmail.com"))
            prt = s.scalar(select(PasswordReset).where(PasswordReset.user_id == u.id).order_by(PasswordReset.id.desc()))
            rtok = prt.token
    else:
        rtok = rurl.split("token=")[1]
    check("reset link issued", bool(rtok))
    check("unknown email gives the same answer (no enumeration)",
          c.post("/api/auth/forgot-password",
                 json={"email": "nobody@nowhere.com"}).json()["note"]
          == "If that email is registered, a reset link has been sent.")
    check("weak reset password rejected",
          c.post("/api/auth/reset-password",
                 json={"token": rtok, "newPassword": "abc"}).status_code == 422)
    check("reset returns a session",
          c.post("/api/auth/reset-password",
                 json={"token": rtok, "newPassword": "Reset12345"}
                 ).status_code == 200)
    check("reset link is single use",
          c.post("/api/auth/reset-password",
                 json={"token": rtok, "newPassword": "Other12345"}
                 ).status_code == 409)
    check("login with the reset password",
          c.post("/api/auth/login", json={
              "email": "pwtest@gmail.com", "password": "Reset12345"}
          ).status_code == 200)

    section("Update own profile")
    r = c.put("/api/auth/me", headers=A, json={"name": "Renamed Admin"})
    check("PUT /me 200", r.status_code == 200 and r.json()["name"] == "Renamed Admin")
    check("too-short name rejected",
          c.put("/api/auth/me", headers=A, json={"name": "X"}).status_code == 422)

    # ----------------------------------------------------------------- users
    section("Edit a user (the Edit button had no endpoint)")
    inv = c.post("/api/users/invite", headers=A, json={
        "email": "edit-me@gmail.com", "name": "Edit Me", "phone": "01155000111",
        "role": "branch", "branch": "Giza"}).json()
    uid = inv["user"]["id"]
    check("GET /api/users/{id}", c.get(f"/api/users/{uid}", headers=A).status_code == 200)
    r = c.put(f"/api/users/{uid}", headers=A, json={"name": "Edited Name",
                                                    "branch": "Cairo"})
    check("PUT updates name and branch",
          r.status_code == 200 and r.json()["name"] == "Edited Name"
          and r.json()["branch"] == "Cairo", r.status_code)
    check("list now exposes isActive / emailVerified",
          {"isActive", "emailVerified"} <= set(
              c.get("/api/users", headers=A).json()[0]))
    check("non-admin cannot change a role",
          c.put(f"/api/users/{uid}", headers=G,
                json={"role": "admin"}).status_code in (403, 404))
    check("cannot change your own role",
          c.put(f"/api/users/{admin['user']['id']}", headers=A,
                json={"role": "branch"}).status_code == 422)
    check("cannot suspend yourself",
          c.put(f"/api/users/{admin['user']['id']}", headers=A,
                json={"isActive": False}).status_code == 422)
    check("suspending a user works",
          c.put(f"/api/users/{uid}", headers=A,
                json={"isActive": False}).status_code == 200)
    check("last admin cannot be demoted",
          c.put(f"/api/users/{admin['user']['id']}", headers=A,
                json={"role": "supplier"}).status_code == 422)
    check("unknown user 404",
          c.put("/api/users/999999", headers=A, json={"name": "X"}).status_code == 404)

    # ------------------------------------------------------------- suppliers
    section("Suppliers CRUD (was read-only)")
    r = c.post("/api/suppliers", headers=A, json={
        "name": "Test Pharma Co", "contactEmail": "a@b.com", "phone": "0100",
        "leadTimeDays": 4, "reliabilityScore": 0.8, "minOrderValue": 500})
    check("create supplier 201", r.status_code == 201, r.status_code)
    sid = r.json()["id"]
    check("duplicate name rejected",
          c.post("/api/suppliers", headers=A,
                 json={"name": "Test Pharma Co"}).status_code == 409)
    check("non-admin cannot create",
          c.post("/api/suppliers", headers=G,
                 json={"name": "Nope Co"}).status_code == 403)
    check("update supplier",
          c.put(f"/api/suppliers/{sid}", headers=A, json={
              "name": "Renamed Pharma", "leadTimeDays": 9}
          ).json()["leadTimeDays"] == 9)
    check("delete unused supplier",
          c.delete(f"/api/suppliers/{sid}", headers=A).status_code == 204)
    in_use = c.get("/api/suppliers", headers=A).json()[0]
    check("supplier in use cannot be deleted",
          c.delete(f"/api/suppliers/{in_use['id']}", headers=A).status_code == 409)

    section("Categories CRUD (was read-only)")
    r = c.post("/api/categories", headers=A,
               json={"name": "Test Category", "description": "d"})
    check("create category 201", r.status_code == 201, r.status_code)
    cid = r.json()["id"]
    check("duplicate rejected",
          c.post("/api/categories", headers=A,
                 json={"name": "Test Category"}).status_code == 409)
    check("update category",
          c.put(f"/api/categories/{cid}", headers=A,
                json={"name": "Renamed Category"}).status_code == 200)
    check("delete category", c.delete(f"/api/categories/{cid}",
                                      headers=A).status_code == 204)
    used = c.get("/api/categories", headers=A).json()[0]
    check("category in use cannot be deleted",
          c.delete(f"/api/categories/{used['id']}", headers=A).status_code == 409)

    section("Branches CRUD (was read-only)")
    r = c.post("/api/branches", headers=A,
               json={"name": "Test Branch", "address": "somewhere"})
    check("create branch 201", r.status_code == 201, r.status_code)
    bid = r.json()["id"]
    check("duplicate rejected",
          c.post("/api/branches", headers=A,
                 json={"name": "Test Branch"}).status_code == 409)
    check("delete empty branch",
          c.delete(f"/api/branches/{bid}", headers=A).status_code == 204)
    giza_b = [b for b in c.get("/api/branches", headers=A).json()
              if b["name"] == "Giza"][0]
    check("branch with products cannot be deleted",
          c.delete(f"/api/branches/{giza_b['id']}", headers=A).status_code == 409)

    # ------------------------------------------------------------- customers
    section("Customers (screen had no backend at all)")
    check("list starts empty",
          c.get("/api/customers?branch=Giza", headers=A).json() == [])
    r = c.post("/api/customers", headers=G, json={
        "name": "Ahmed Client", "phone": "01099887711",
        "email": "AHMED@X.COM", "address": "Giza"})
    check("create customer 201", r.status_code == 201, r.status_code)
    cu = r.json()
    check("email is normalised", cu["email"] == "ahmed@x.com")
    check("branch derived from the signed-in user", cu["branch"] == "Giza")
    check("duplicate phone in the same branch rejected",
          c.post("/api/customers", headers=G, json={
              "name": "Other", "phone": "01099887711"}).status_code == 409)
    check("too-short name rejected",
          c.post("/api/customers", headers=G, json={"name": "X"}).status_code == 422)
    check("search by name",
          len(c.get("/api/customers?branch=Giza&q=Ahmed", headers=G).json()) == 1)
    check("search miss returns nothing",
          c.get("/api/customers?branch=Giza&q=zzzz", headers=G).json() == [])
    check("get one", c.get(f"/api/customers/{cu['id']}", headers=G).status_code == 200)
    check("update customer",
          c.put(f"/api/customers/{cu['id']}", headers=G, json={
              "name": "Ahmed Renamed", "phone": "01099887711"}
          ).json()["name"] == "Ahmed Renamed")
    check("a branch user cannot read another branch",
          c.get("/api/customers?branch=Cairo", headers=G).status_code == 403)
    check("delete customer",
          c.delete(f"/api/customers/{cu['id']}", headers=G).status_code == 204)

    # ---------------------------------------------------------- notifications
    section("Send a notification (permission existed, endpoint did not)")
    r = c.post("/api/notifications", headers=G, json={
        "title": "Stock check", "body": "Please review", "kind": "warning"})
    check("create notification 201", r.status_code == 201, r.status_code)
    check("it appears in the branch list",
          any(n["title"] == "Stock check"
              for n in c.get("/api/notifications?branch=Giza", headers=G).json()))
    check("empty title rejected",
          c.post("/api/notifications", headers=G, json={"title": " "}).status_code == 422)
    check("unknown target user 404",
          c.post("/api/notifications", headers=A, json={
              "title": "x", "userId": 999999}).status_code == 404)

    # ------------------------------------------------------------ chat history
    section("Chat history ordering (assistant repeated its own answer)")
    q1 = c.post("/api/ai/chat", headers=G,
                json={"message": "Which products are low on stock?",
                      "branch": "Giza"}).json()
    conv = q1["conversationId"]
    q2 = c.post("/api/ai/chat", headers=G,
                json={"message": "Which products are near expiry?",
                      "conversationId": conv, "branch": "Giza"}).json()
    q3 = c.post("/api/ai/chat", headers=G,
                json={"message": "Summarize inventory health",
                      "conversationId": conv, "branch": "Giza"}).json()

    hist = c.get(f"/api/ai/history?conversationId={conv}", headers=G).json()
    roles = [m["role"] for m in hist["messages"]]
    check("history holds 3 full turns", len(roles) == 6, roles)
    check("roles strictly alternate user/assistant",
          roles == ["user", "assistant"] * 3, roles)
    contents = [m["content"] for m in hist["messages"] if m["role"] == "user"]
    check("no duplicated user message", len(set(contents)) == 3, contents)
    check("each question got its own answer",
          len({q1["reply"], q2["reply"], q3["reply"]}) == 3)
    check("the reply does not restate the previous one",
          q2["reply"] not in q3["reply"] and q1["reply"] not in q2["reply"])

    # asking the same question twice must not append to the earlier answer
    same_a = c.post("/api/ai/chat", headers=G, json={
        "message": "Which products are low on stock?",
        "conversationId": conv, "branch": "Giza"}).json()["reply"]
    check("a repeated question is answered fresh, not extended",
          len(same_a) < len(q1["reply"]) * 2.5,
          f"{len(q1['reply'])} -> {len(same_a)} chars")

    check("clearing history works",
          c.delete(f"/api/ai/history?conversationId={conv}",
                   headers=G).status_code == 204)
    check("history is empty after clearing",
          c.get(f"/api/ai/history?conversationId={conv}",
                headers=G).json()["messages"] == [])

shutil.rmtree(_tmp, ignore_errors=True)
passed = sum(1 for ok, _ in res if ok)
print(f"\n{'=' * 56}\n{passed}/{len(res)} passed")
for ok, n in res:
    if not ok:
        print(f"  FAILED: {n}")
sys.exit(0 if passed == len(res) else 1)
