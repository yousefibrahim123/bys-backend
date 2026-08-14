"""
Reference data (suppliers, categories, branches), customers, and outgoing
notifications.

These were the biggest holes in the API. Suppliers, categories and branches
were read-only, so the `add_supplier` / `edit_supplier` permissions in the UI
had nothing to call. Customers had no backend at all — the Customers screen
rendered a hardcoded list. And `send_notifications` existed as a permission
with no endpoint behind it.

Writes are restricted to admins (plus branch managers for customers and
notifications, which are branch-scoped).
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (Branch, Category, Customer, Notification, Product, Sale,
                      Store, Supplier, User)
from ..schemas import (BranchIn, CategoryIn, CustomerIn, CustomerOut,
                       NotificationIn, SupplierIn)
from ..security import get_current_user, require_roles, resolve_branch

suppliers = APIRouter(prefix="/api/suppliers", tags=["suppliers"])
categories = APIRouter(prefix="/api/categories", tags=["categories"])
branches = APIRouter(prefix="/api/branches", tags=["branches"])
customers = APIRouter(prefix="/api/customers", tags=["customers"])
notifications = APIRouter(prefix="/api/notifications", tags=["notifications"])

admin_only = require_roles("admin")
staff = require_roles("admin", "branch", "supplier")


def _supplier_out(s: Supplier) -> dict:
    return {"id": s.id, "name": s.name, "contactEmail": s.contact_email,
            "phone": s.phone, "leadTimeDays": s.lead_time_days,
            "reliabilityScore": s.reliability_score,
            "minOrderValue": s.min_order_value}


# ------------------------------------------------------------------ suppliers

@suppliers.post("", status_code=status.HTTP_201_CREATED)
def create_supplier(payload: SupplierIn, db: Session = Depends(get_db),
                    _: User = Depends(admin_only)) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Name is required")
    if db.scalar(select(Supplier).where(Supplier.name == name)):
        raise HTTPException(status.HTTP_409_CONFLICT, "That supplier already exists")
    s = Supplier(name=name, contact_email=payload.contact_email.strip(),
                 phone=payload.phone.strip(),
                 lead_time_days=max(0, payload.lead_time_days),
                 reliability_score=min(max(payload.reliability_score, 0.0), 1.0),
                 min_order_value=max(0.0, payload.min_order_value))
    db.add(s)
    db.commit()
    db.refresh(s)
    return _supplier_out(s)


@suppliers.put("/{supplier_id}")
def update_supplier(supplier_id: int, payload: SupplierIn,
                    db: Session = Depends(get_db),
                    _: User = Depends(admin_only)) -> dict:
    s = db.get(Supplier, supplier_id)
    if not s:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Supplier not found")
    name = payload.name.strip()
    if name and name != s.name:
        if db.scalar(select(Supplier).where(Supplier.name == name)):
            raise HTTPException(status.HTTP_409_CONFLICT, "That name is already used")
        # Products reference the supplier by name, so a rename has to cascade
        # or the products silently detach from their supplier.
        db.query(Product).filter(Product.supplier == s.name).update(
            {Product.supplier: name})
        s.name = name
    s.contact_email = payload.contact_email.strip()
    s.phone = payload.phone.strip()
    s.lead_time_days = max(0, payload.lead_time_days)
    s.reliability_score = min(max(payload.reliability_score, 0.0), 1.0)
    s.min_order_value = max(0.0, payload.min_order_value)
    db.commit()
    db.refresh(s)
    return _supplier_out(s)


@suppliers.delete("/{supplier_id}", status_code=status.HTTP_204_NO_CONTENT,
                  response_model=None)
def delete_supplier(supplier_id: int, db: Session = Depends(get_db),
                    _: User = Depends(admin_only)) -> None:
    s = db.get(Supplier, supplier_id)
    if not s:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Supplier not found")
    in_use = db.scalar(select(func.count()).select_from(Product).where(
        Product.supplier == s.name)) or 0
    if in_use:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{in_use} product(s) still use this supplier — reassign them first")
    db.delete(s)
    db.commit()


# ----------------------------------------------------------------- categories

@categories.post("", status_code=status.HTTP_201_CREATED)
def create_category(payload: CategoryIn, db: Session = Depends(get_db),
                    _: User = Depends(admin_only)) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Name is required")
    if db.scalar(select(Category).where(Category.name == name)):
        raise HTTPException(status.HTTP_409_CONFLICT, "That category already exists")
    c = Category(name=name, description=payload.description.strip())
    db.add(c)
    db.commit()
    db.refresh(c)
    return {"id": c.id, "name": c.name, "description": c.description}


@categories.put("/{category_id}")
def update_category(category_id: int, payload: CategoryIn,
                    db: Session = Depends(get_db),
                    _: User = Depends(admin_only)) -> dict:
    c = db.get(Category, category_id)
    if not c:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Category not found")
    name = payload.name.strip()
    if name and name != c.name:
        if db.scalar(select(Category).where(Category.name == name)):
            raise HTTPException(status.HTTP_409_CONFLICT, "That name is already used")
        db.query(Product).filter(Product.category == c.name).update(
            {Product.category: name})
        c.name = name
    c.description = payload.description.strip()
    db.commit()
    db.refresh(c)
    return {"id": c.id, "name": c.name, "description": c.description}


@categories.delete("/{category_id}", status_code=status.HTTP_204_NO_CONTENT,
                   response_model=None)
def delete_category(category_id: int, db: Session = Depends(get_db),
                    _: User = Depends(admin_only)) -> None:
    c = db.get(Category, category_id)
    if not c:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Category not found")
    in_use = db.scalar(select(func.count()).select_from(Product).where(
        Product.category == c.name)) or 0
    if in_use:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{in_use} product(s) still use this category — reassign them first")
    db.delete(c)
    db.commit()


# ------------------------------------------------------------------- branches

@branches.post("", status_code=status.HTTP_201_CREATED)
def create_branch(payload: BranchIn, db: Session = Depends(get_db),
                  _: User = Depends(admin_only)) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Name is required")
    if db.scalar(select(Branch).where(Branch.name == name)):
        raise HTTPException(status.HTTP_409_CONFLICT, "That branch already exists")
    store = db.scalar(select(Store).order_by(Store.id))
    if not store:
        store = Store(name="BYS")
        db.add(store)
        db.flush()
    b = Branch(store_id=store.id, name=name, address=payload.address.strip(),
               phone=payload.phone.strip())
    db.add(b)
    db.commit()
    db.refresh(b)
    return {"id": b.id, "name": b.name, "address": b.address, "phone": b.phone}


@branches.put("/{branch_id}")
def update_branch(branch_id: int, payload: BranchIn, db: Session = Depends(get_db),
                  _: User = Depends(admin_only)) -> dict:
    b = db.get(Branch, branch_id)
    if not b:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Branch not found")
    name = payload.name.strip()
    if name and name != b.name:
        if db.scalar(select(Branch).where(Branch.name == name)):
            raise HTTPException(status.HTTP_409_CONFLICT, "That name is already used")
        # Products, users, sales and orders all key off the branch *name*.
        # Renaming without cascading would orphan every one of them.
        old = b.name
        for model, col in ((Product, Product.branch), (User, User.branch),
                           (Sale, Sale.branch), (Customer, Customer.branch),
                           (Notification, Notification.branch)):
            db.query(model).filter(col == old).update({col: name})
        b.name = name
    b.address = payload.address.strip()
    b.phone = payload.phone.strip()
    db.commit()
    db.refresh(b)
    return {"id": b.id, "name": b.name, "address": b.address, "phone": b.phone}


@branches.delete("/{branch_id}", status_code=status.HTTP_204_NO_CONTENT,
                 response_model=None)
def delete_branch(branch_id: int, db: Session = Depends(get_db),
                  _: User = Depends(admin_only)) -> None:
    b = db.get(Branch, branch_id)
    if not b:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Branch not found")
    products = db.scalar(select(func.count()).select_from(Product).where(
        Product.branch == b.name)) or 0
    users = db.scalar(select(func.count()).select_from(User).where(
        User.branch == b.name)) or 0
    if products or users:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Branch still has {products} product(s) and {users} user(s)")
    db.delete(b)
    db.commit()


# ------------------------------------------------------------------ customers

def _customer_out(c: Customer) -> CustomerOut:
    return CustomerOut.model_validate(c)


@customers.get("", response_model=List[CustomerOut])
def list_customers(branch: Optional[str] = None, q: Optional[str] = None,
                   active_only: bool = Query(False),
                   db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)) -> List[CustomerOut]:
    b = resolve_branch(user, branch)
    stmt = select(Customer).where(Customer.branch == b)
    if active_only:
        stmt = stmt.where(Customer.is_active.is_(True))
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(Customer.name.ilike(like), Customer.phone.ilike(like),
                              Customer.email.ilike(like)))
    return [_customer_out(c) for c in db.scalars(stmt.order_by(Customer.name))]


@customers.post("", response_model=CustomerOut, status_code=status.HTTP_201_CREATED)
def create_customer(payload: CustomerIn, db: Session = Depends(get_db),
                    user: User = Depends(staff)) -> CustomerOut:
    name = payload.name.strip()
    if len(name) < 2:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Name must be at least 2 characters")
    b = resolve_branch(user, payload.branch)
    phone = payload.phone.strip()
    if phone and db.scalar(select(Customer).where(Customer.phone == phone,
                                                  Customer.branch == b)):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "A customer with that number already exists in this branch")
    c = Customer(name=name, phone=phone, email=payload.email.strip().lower(),
                 branch=b, address=payload.address.strip(),
                 notes=payload.notes.strip(), is_active=payload.is_active)
    db.add(c)
    db.commit()
    db.refresh(c)
    return _customer_out(c)


@customers.get("/{customer_id}", response_model=CustomerOut)
def get_customer(customer_id: int, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> CustomerOut:
    c = db.get(Customer, customer_id)
    if not c:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
    resolve_branch(user, c.branch)
    return _customer_out(c)


@customers.get("/{customer_id}/history")
def customer_history(customer_id: int, limit: int = Query(100, ge=1, le=500),
                     db: Session = Depends(get_db),
                     user: User = Depends(get_current_user)) -> dict:
    """
    The customer's real purchase history: every sale recorded under their
    name in their branch, newest first, with totals.
    """
    c = db.get(Customer, customer_id)
    if not c:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
    resolve_branch(user, c.branch)

    from ..models import Product as ProductModel
    rows = db.execute(
        select(Sale, ProductModel.name)
        .join(ProductModel, ProductModel.id == Sale.product_id)
        .where(Sale.branch == c.branch, Sale.customer_name == c.name)
        .order_by(Sale.sold_at.desc()).limit(limit)).all()

    purchases = [{
        "id": s.id, "productId": s.product_id, "productName": name,
        "quantity": s.quantity, "unitPrice": s.unit_price, "total": s.total,
        "soldAt": s.sold_at.replace(tzinfo=timezone.utc).isoformat(),
    } for s, name in rows]
    return {
        "customer": _customer_out(c),
        "purchases": purchases,
        "totalSpent": round(sum(p["total"] for p in purchases), 2),
        "visits": len({p["soldAt"][:10] for p in purchases}),
    }


@customers.put("/{customer_id}", response_model=CustomerOut)
def update_customer(customer_id: int, payload: CustomerIn,
                    db: Session = Depends(get_db),
                    user: User = Depends(staff)) -> CustomerOut:
    c = db.get(Customer, customer_id)
    if not c:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
    resolve_branch(user, c.branch)
    name = payload.name.strip()
    if len(name) < 2:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Name must be at least 2 characters")
    phone = payload.phone.strip()
    if phone and db.scalar(select(Customer).where(Customer.phone == phone,
                                                  Customer.branch == c.branch,
                                                  Customer.id != c.id)):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "A customer with that number already exists in this branch")
    c.name, c.phone = name, phone
    c.email = payload.email.strip().lower()
    c.address = payload.address.strip()
    c.notes = payload.notes.strip()
    c.is_active = payload.is_active
    db.commit()
    db.refresh(c)
    return _customer_out(c)


@customers.delete("/{customer_id}", status_code=status.HTTP_204_NO_CONTENT,
                  response_model=None)
def delete_customer(customer_id: int, db: Session = Depends(get_db),
                    user: User = Depends(staff)) -> None:
    c = db.get(Customer, customer_id)
    if not c:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
    resolve_branch(user, c.branch)
    db.delete(c)
    db.commit()


# -------------------------------------------------------------- notifications

@notifications.post("", status_code=status.HTTP_201_CREATED)
def create_notification(payload: NotificationIn, db: Session = Depends(get_db),
                        user: User = Depends(staff)) -> dict:
    """
    Send a notification to a branch or a single user. The UI has a
    `send_notifications` permission that had no endpoint to call.
    """
    title = payload.title.strip()
    if not title:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Title is required")
    if payload.user_id is not None and not db.get(User, payload.user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Target user not found")
    b = resolve_branch(user, payload.branch)
    n = Notification(user_id=payload.user_id, branch=b, title=title,
                     body=payload.body.strip(), kind=payload.kind,
                     created_at=datetime.now(timezone.utc))
    db.add(n)
    db.commit()
    db.refresh(n)
    return {"id": n.id, "title": n.title, "body": n.body, "kind": n.kind,
            "branch": n.branch, "userId": n.user_id, "isRead": n.is_read,
            "createdAt": n.created_at.isoformat()}
