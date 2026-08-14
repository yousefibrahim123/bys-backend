"""
End-to-end test of registration, email verification, and invites — no server
needed.

    python test_email_flow.py

Runs against a temporary copy of the database so your pharma.db is untouched.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_tmp = Path(tempfile.mkdtemp())
src = HERE / "pharma.db"
dst = _tmp / "pharma.db"
if src.exists():
    shutil.copy(src, dst)
os.environ["DATABASE_URL"] = f"sqlite:///{dst}"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

OK, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results: list[tuple[bool, str]] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    results.append((cond, name))
    print(f"  [{OK if cond else FAIL}] {name}{('  → ' + extra) if extra else ''}")


with TestClient(app) as c:
    print("\n--- 1. Self-registration ---")
    r = c.post("/api/auth/register", json={
        "email": "flowtest@gmail.com", "password": "Test1234",
        "name": "Flow Test", "phone": "01155443322",
        "role": "branch", "branch": "Cairo"})
    check("register returns 201", r.status_code == 201, str(r.status_code))
    body = r.json() if r.status_code == 201 else {}
    check("no accessToken in response", "accessToken" not in body)
    check("verification link present", bool(body.get("verificationUrl")))

    print("\n--- 2. Login before verification ---")
    r = c.post("/api/auth/login", json={
        "email": "flowtest@gmail.com", "password": "Test1234"})
    check("rejected with 403", r.status_code == 403, r.json().get("detail", ""))

    print("\n--- 3. Verification ---")
    token = body["verificationUrl"].split("token=")[1]
    r = c.get(f"/api/auth/verify-email/{token}")
    check("token check 200", r.status_code == 200)
    r = c.post("/api/auth/verify-email", json={"token": token})
    check("verification returns a token", r.status_code == 200 and "accessToken" in r.json())
    r = c.post("/api/auth/verify-email", json={"token": token})
    check("re-using the link doesn't break", r.status_code == 200)
    r = c.get("/api/auth/verify-email/wrong-token-xyz")
    check("bad token returns 404", r.status_code == 404)

    print("\n--- 4. Login after verification ---")
    r = c.post("/api/auth/login", json={
        "email": "flowtest@gmail.com", "password": "Test1234"})
    check("login succeeded", r.status_code == 200, str(r.status_code))

    print("\n--- 5. Input validation ---")
    check("malformed email rejected", c.post("/api/auth/register", json={
        "email": "not-an-email", "password": "Test1234",
        "name": "X"}).status_code == 422)
    check("short password rejected", c.post("/api/auth/register", json={
        "email": "short@gmail.com", "password": "ab1", "name": "X"}).status_code == 422)
    check("password without a digit rejected", c.post("/api/auth/register", json={
        "email": "nodigit@gmail.com", "password": "abcdefgh",
        "name": "X"}).status_code == 422)
    check("duplicate email rejected", c.post("/api/auth/register", json={
        "email": "flowtest@gmail.com", "password": "Test1234",
        "name": "X"}).status_code == 409)

    print("\n--- 6. Pre-existing users (migration) ---")
    r = c.post("/api/auth/login", json={"email": "admin@gmail.com", "password": "123123"})
    check("pre-existing admin can still log in", r.status_code == 200, str(r.status_code))
    tok = r.json()["accessToken"] if r.status_code == 200 else ""
    H = {"Authorization": f"Bearer {tok}"}

    print("\n--- 7. Employee invite ---")
    r = c.post("/api/users/invite", headers=H, json={
        "email": "invitee@gmail.com", "name": "Invitee", "phone": "01166554433",
        "role": "branch", "branch": "Giza", "jobTitle": "Pharmacist"})
    check("invite returns 201", r.status_code == 201, str(r.status_code))
    inv = r.json() if r.status_code == 201 else {}
    check("activation link present", bool(inv.get("activationUrl")))
    check("invite without a token rejected",
          c.post("/api/users/invite", json={"email": "a@b.com", "name": "A"}
                 ).status_code == 401)

    print("\n--- 8. Invitee activation ---")
    itok = inv["activationUrl"].split("token=")[1]
    check("invite check 200", c.get(f"/api/users/invite/{itok}").status_code == 200)
    r = c.post("/api/users/activate", json={"token": itok, "password": "Invited123"})
    check("activation returns a token", r.status_code == 200 and "accessToken" in r.json())
    r = c.post("/api/auth/login", json={
        "email": "invitee@gmail.com", "password": "Invited123"})
    check("invitee can log in after activation", r.status_code == 200, str(r.status_code))
    check("link cannot be reused",
          c.post("/api/users/activate",
                 json={"token": itok, "password": "Other123"}).status_code == 409)

    print("\n--- 9. Email status and models ---")
    r = c.get("/api/auth/email-status")
    check("email-status works", r.status_code == 200)
    print(f"       SMTP configured = {r.json().get('configured')}  "
          f"missing = {r.json().get('missing')}")
    r = c.get("/api/ml/status").json()
    check("forecast model available", r["forecast"]["available"])
    check("expiry model available", r["expiry"]["available"])
    check("matcher model available", r["matcher_real"]["available"])
    check("forecast metrics not empty", bool(r["forecast"]["metrics"]),
          str(list(r["forecast"]["metrics"])[:4]))
    check("expiry metrics not empty", bool(r["expiry"]["metrics"]),
          str(list(r["expiry"]["metrics"])[:4]))

shutil.rmtree(_tmp, ignore_errors=True)
passed = sum(1 for ok, _ in results if ok)
print(f"\n{'='*50}\n{passed}/{len(results)} passed")
for ok, name in results:
    if not ok:
        print(f"  FAILED: {name}")
sys.exit(0 if passed == len(results) else 1)
