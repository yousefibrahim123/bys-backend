"""Products — search, filtering, pagination, and full CRUD."""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import csv
import io
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Branch, Product, User
from ..schemas import (PageMeta, ProductCreate, ProductOut, ProductPage,
                       ProductUpdate)
from ..security import get_current_user, resolve_branch
from ..seed import compute_inventory_status

router = APIRouter(prefix="/api/products", tags=["products"])


def _touch(p: Product) -> None:
    p.inventory_status = compute_inventory_status(p)
    p.last_updated_date = datetime.now(timezone.utc).isoformat()


@router.get("", response_model=ProductPage)
def list_products(
    branch: Optional[str] = None,
    search: Optional[str] = None,
    category: Optional[str] = None,
    supplier: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    inventory_status: Optional[str] = None,
    risk_level: Optional[str] = None,
    sort_by: str = Query("name", pattern="^(name|quantity|price|expiry|sales)$"),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProductPage:
    b = resolve_branch(user, branch)
    stmt = select(Product).where(Product.branch == b)

    if search:
        like = f"%{search.strip()}%"
        # Supplier was missing from this list, so typing a distributor's name
        # into the product search returned nothing even when their products
        # were sitting right there.
        stmt = stmt.where(or_(Product.name.ilike(like), Product.sku.ilike(like),
                              Product.barcode.ilike(like), Product.category.ilike(like),
                              Product.supplier.ilike(like)))
    if category:
        stmt = stmt.where(Product.category == category)
    if supplier:
        stmt = stmt.where(Product.supplier == supplier)
    if status_filter:
        stmt = stmt.where(Product.status == status_filter)
    if inventory_status:
        stmt = stmt.where(Product.inventory_status == inventory_status)
    if risk_level:
        stmt = stmt.where(Product.risk_level == risk_level)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0

    col = {"name": Product.name, "quantity": Product.quantity,
           "price": Product.selling_price, "expiry": Product.expiry_date,
           "sales": Product.monthly_sales}[sort_by]
    stmt = stmt.order_by(col.desc() if order == "desc" else col.asc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)

    items = [ProductOut.from_orm_product(p) for p in db.scalars(stmt)]
    return ProductPage(items=items, meta=PageMeta(
        page=page, page_size=page_size, total=total,
        total_pages=max(1, -(-total // page_size))))


@router.get("/all", response_model=List[ProductOut])
def all_products(branch: Optional[str] = None, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> List[ProductOut]:
    """All products in a branch, unpaginated — used by screens computing totals."""
    b = resolve_branch(user, branch)
    return [ProductOut.from_orm_product(p)
            for p in db.scalars(select(Product).where(Product.branch == b)
                                .order_by(Product.name))]


@router.get("/global", response_model=List[ProductOut])
def global_products(db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)) -> List[ProductOut]:
    """All branches — admins and suppliers only."""
    if user.role not in ("admin", "supplier"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admins and suppliers only")
    return [ProductOut.from_orm_product(p)
            for p in db.scalars(select(Product).order_by(Product.branch, Product.name))]


@router.get("/categories", response_model=List[str])
def categories(branch: Optional[str] = None, db: Session = Depends(get_db),
               user: User = Depends(get_current_user)) -> List[str]:
    b = resolve_branch(user, branch)
    rows = db.scalars(select(Product.category).where(
        Product.branch == b, Product.category != "").distinct())
    return sorted(rows)


@router.get("/suppliers", response_model=List[str])
def suppliers(branch: Optional[str] = None, db: Session = Depends(get_db),
              user: User = Depends(get_current_user)) -> List[str]:
    b = resolve_branch(user, branch)
    rows = db.scalars(select(Product.supplier).where(
        Product.branch == b, Product.supplier != "").distinct())
    return sorted(rows)


@router.get("/barcode/{barcode}", response_model=ProductOut)
def by_barcode(barcode: str, branch: Optional[str] = None, db: Session = Depends(get_db),
               user: User = Depends(get_current_user)) -> ProductOut:
    b = resolve_branch(user, branch)
    p = db.scalar(select(Product).where(Product.branch == b, Product.barcode == barcode))
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No product with that barcode")
    return ProductOut.from_orm_product(p)


@router.get("/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db),
                user: User = Depends(get_current_user)) -> ProductOut:
    p = db.get(Product, product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    resolve_branch(user, p.branch)
    return ProductOut.from_orm_product(p)


def _validate_product_payload(name: str, purchase: float, selling: float,
                              quantity: int, minimum: int, maximum: int) -> None:
    if len(name.strip()) < 2:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Product name must be at least 2 characters")
    if purchase < 0 or selling < 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Prices can't be negative")
    if quantity < 0 or minimum < 0 or maximum < 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Stock figures can't be negative")
    if maximum and minimum > maximum:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Minimum stock can't exceed maximum stock")


@router.post("", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)) -> ProductOut:
    b = resolve_branch(user, payload.branch)
    if not b or not db.scalar(select(Branch).where(Branch.name == b)):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Branch '{b or '(none)'}' does not exist — create the "
                            f"branch first or select an existing one")
    _validate_product_payload(payload.name, payload.purchase_price,
                              payload.selling_price, payload.quantity,
                              payload.minimum_stock, payload.maximum_stock)
    sku = payload.sku.strip() or f"SKU-{int(datetime.now().timestamp())}"
    if db.scalar(select(Product).where(Product.branch == b, Product.sku == sku)):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"SKU {sku} is already used in the {b} branch")
    data = payload.model_dump(exclude={"branch"})
    data["sku"] = sku
    data["name"] = payload.name.strip()
    p = Product(**data, branch=b)
    _touch(p)
    p.last_restock_date = datetime.now(timezone.utc).date().isoformat()
    db.add(p)
    db.commit()
    db.refresh(p)
    return ProductOut.from_orm_product(p)


@router.put("/{product_id}", response_model=ProductOut)
def update_product(product_id: int, payload: ProductUpdate,
                   db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)) -> ProductOut:
    p = db.get(Product, product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    resolve_branch(user, p.branch)
    _validate_product_payload(
        payload.name if payload.name is not None else p.name,
        payload.purchase_price if payload.purchase_price is not None else p.purchase_price,
        payload.selling_price if payload.selling_price is not None else p.selling_price,
        payload.quantity if payload.quantity is not None else p.quantity,
        payload.minimum_stock if payload.minimum_stock is not None else p.minimum_stock,
        payload.maximum_stock if payload.maximum_stock is not None else p.maximum_stock)
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(p, k, v)
    if payload.name is not None:
        p.name = payload.name.strip()
    _touch(p)
    db.commit()
    db.refresh(p)
    return ProductOut.from_orm_product(p)


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_product(product_id: int, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)) -> None:
    p = db.get(Product, product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    resolve_branch(user, p.branch)
    db.delete(p)
    db.commit()


@router.post("/import-csv")
async def import_csv(file: UploadFile = File(...), branch: Optional[str] = None,
                     db: Session = Depends(get_db),
                     user: User = Depends(get_current_user)) -> dict:
    """
    CSV import. Required column: name. Everything else is optional.
    An existing product (same sku) is updated rather than duplicated.
    """
    b = resolve_branch(user, branch)
    raw = (await file.read()).decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames or "name" not in [f.strip().lower() for f in reader.fieldnames]:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "The file must contain a 'name' column")

    created = updated = skipped = 0
    errors: List[str] = []
    for i, row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        name = row.get("name", "")
        if not name:
            skipped += 1
            continue
        try:
            sku = row.get("sku") or f"IMP-{abs(hash(name)) % 10**8}"
            p = db.scalar(select(Product).where(Product.branch == b, Product.sku == sku))
            if p is None:
                p = Product(branch=b, sku=sku, name=name)
                db.add(p)
                created += 1
            else:
                updated += 1
            p.name = name
            p.barcode = row.get("barcode", p.barcode)
            p.category = row.get("category", p.category)
            p.supplier = row.get("supplier", p.supplier)
            for field, key in (("purchase_price", "purchaseprice"),
                               ("selling_price", "sellingprice"),
                               ("daily_sales", "dailysales")):
                if row.get(key):
                    setattr(p, field, float(row[key]))
            for field, key in (("quantity", "quantity"), ("minimum_stock", "minimumstock"),
                               ("maximum_stock", "maximumstock")):
                if row.get(key):
                    setattr(p, field, int(float(row[key])))
            p.expiry_date = row.get("expirydate", p.expiry_date)
            p.batch_number = row.get("batchnumber", p.batch_number)
            _touch(p)
        except (ValueError, TypeError) as e:
            skipped += 1
            errors.append(f"row {i}: {e}")
    db.commit()
    return {"branch": b, "created": created, "updated": updated, "skipped": skipped,
            "errors": errors[:20]}
