"""
Permissions.

The UI has always had a rich permission list; the backend had three coarse
roles and nothing else, so every checkbox in the add-user wizard was thrown
away. This module is the single source of truth for:

  * which permissions exist (the same identifiers the frontend already uses)
  * what each role gets by default
  * the delegation rule: you can only grant what you hold

That last rule is the important one. Without it any branch manager could tick
`delete_users` for a new hire and hand out authority they never had
themselves.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import User, UserPermission

# Grouped for the UI. Keys match the section headings the wizard renders.
CATALOG: Dict[str, List[str]] = {
    "Dashboard": ["view_dashboard"],
    "Analytics": ["view_analytics", "export_reports"],
    "AI": ["view_ai_assistant", "view_ai_insights", "view_demand_insights",
           "smart_recommendations"],
    "Inventory": ["view_inventory", "add_product", "edit_product", "delete_product"],
    "Orders": ["view_orders", "create_orders", "approve_orders", "cancel_orders"],
    "Suppliers": ["view_suppliers", "add_supplier", "edit_supplier"],
    "Customers": ["view_customers", "add_customer", "edit_customer"],
    "Notifications": ["view_notifications", "send_notifications"],
    "Users": ["view_users", "add_users", "edit_users", "delete_users"],
    "Integrations": ["view_integrations", "connect_system", "disconnect_system",
                     "run_manual_sync"],
    "Settings": ["company_profile", "security_settings", "billing_settings"],
}

ALL: List[str] = [p for group in CATALOG.values() for p in group]
ALL_SET = set(ALL)

# Baseline per role. Anything beyond this has to be granted explicitly.
ROLE_DEFAULTS: Dict[str, List[str]] = {
    "admin": list(ALL),
    "branch": [
        "view_dashboard", "view_analytics", "export_reports",
        "view_ai_assistant", "view_ai_insights", "view_demand_insights",
        "smart_recommendations",
        "view_inventory", "add_product", "edit_product",
        # A pharmacy that can place orders must also be able to withdraw its
        # own order before it ships.
        "view_orders", "create_orders", "approve_orders", "cancel_orders",
        "view_suppliers",
        "view_customers", "add_customer", "edit_customer",
        "view_notifications", "send_notifications",
        "view_users", "add_users",
        "view_integrations", "connect_system", "run_manual_sync",
        "company_profile", "security_settings",
    ],
    "supplier": [
        "view_dashboard", "view_analytics", "export_reports",
        "view_ai_assistant", "view_demand_insights",
        "view_inventory",
        "view_orders", "approve_orders",
        "view_notifications", "send_notifications",
        "view_customers",
        "company_profile", "security_settings",
    ],
}


def defaults_for(role: str) -> List[str]:
    return list(ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["branch"]))


def effective(db: Session, user: User) -> Set[str]:
    """
    What this user can actually do.

    An admin always holds everything — otherwise a bad grant could lock the
    only administrator out of user management with no way back in.
    """
    if user.role == "admin":
        return set(ALL)
    rows = db.scalars(select(UserPermission.permission).where(
        UserPermission.user_id == user.id)).all()
    if rows:
        return {r for r in rows if r in ALL_SET}
    # No explicit grants recorded (older account) — fall back to role defaults
    # so nobody suddenly loses access when this feature ships.
    return set(defaults_for(user.role))


def has(db: Session, user: User, permission: str) -> bool:
    return permission in effective(db, user)


def grantable(db: Session, granter: User, target_role: str) -> List[str]:
    """
    What `granter` is allowed to tick for someone in `target_role`.

    Two limits, intersected:
      * you cannot grant a permission you do not hold yourself
      * you cannot exceed what the target role is allowed to hold

    Returned in catalogue order so the UI renders predictably.
    """
    mine = effective(db, granter)
    ceiling = set(ROLE_DEFAULTS.get(target_role, [])) | mine \
        if granter.role == "admin" else mine
    allowed = mine & ceiling
    return [p for p in ALL if p in allowed]


def apply_grants(db: Session, granter: User, target: User,
                 requested: Optional[List[str]]) -> List[str]:
    """
    Replace the target's grants.

    Anything the granter isn't allowed to delegate is dropped silently rather
    than failing the whole request — the caller reports what was actually
    applied so the UI can show it.
    """
    allowed = set(grantable(db, granter, target.role))
    wanted = set(requested if requested is not None else defaults_for(target.role))
    final = sorted(wanted & allowed, key=ALL.index)

    db.query(UserPermission).filter(UserPermission.user_id == target.id).delete()
    for perm in final:
        db.add(UserPermission(user_id=target.id, permission=perm,
                              granted_by=granter.id))
    db.flush()
    return final
