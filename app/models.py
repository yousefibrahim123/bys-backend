"""
Database models.

Names and fields mirror the frontend Product interface
(frontend/src/data/products.ts) so nothing needs manual mapping, and the
SmartSupply.API entities in backend-dotnet so the two stay interchangeable.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

from datetime import datetime, timezone

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String,
                        UniqueConstraint,
                        Text, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Store(Base):
    __tablename__ = "stores"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    branches: Mapped[List["Branch"]] = relationship(back_populates="store")


class Branch(Base):
    __tablename__ = "branches"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"))
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    address: Mapped[str] = mapped_column(String(300), default="")
    phone: Mapped[str] = mapped_column(String(40), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    store: Mapped["Store"] = relationship(back_populates="branches")
    products: Mapped[List["Product"]] = relationship(back_populates="branch_ref")


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    phone: Mapped[str] = mapped_column(String(40), default="", index=True)
    name: Mapped[str] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(30), default="branch")  # branch|admin|supplier
    # pharmacy|distributor. This used to live only in the browser's
    # localStorage, so the same account showed a pharmacy sidebar on one
    # machine and a distributor sidebar on another — and clearing site data
    # turned a pharmacy into a supplier. It belongs on the account.
    organization_type: Mapped[str] = mapped_column(String(20), default="pharmacy")
    branch: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Self-registration starts False until the user clicks the link in their
    # inbox. Invited accounts are verified by the invite link itself, so this
    # flips to True at activation time.
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    # Explicit server-side marker for the predefined development/test accounts.
    # Only accounts with this flag skip email verification and OTP — the
    # bypass is never decided by email address or by the frontend.
    is_test_account: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class EmailVerification(Base):
    """Single-use email verification token — expires after 24 hours."""
    __tablename__ = "email_verifications"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    token: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class ExpiryAction(Base):
    """
    A near-expiry action that actually happened.

    The screens offered discount / transfer / return and only rendered a
    confirmation. Each row here corresponds to real state that changed.
    """
    __tablename__ = "expiry_actions"
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    branch: Mapped[str] = mapped_column(String(120), index=True)
    # discount | transfer | return | writeoff
    action: Mapped[str] = mapped_column(String(20), index=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    value_egp: Mapped[float] = mapped_column(Float, default=0.0)
    reference_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    performed_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"),
                                                     nullable=True)
    reverted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class UserPermission(Base):
    """
    A permission granted to one user.

    The add-user wizard offered 40+ checkboxes that were never sent anywhere:
    the backend only knew three coarse roles, so every selection was discarded
    the moment the modal closed. Grants now live here and are enforced.
    """
    __tablename__ = "user_permissions"
    __table_args__ = (UniqueConstraint("user_id", "permission",
                                       name="uq_user_permission"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    permission: Mapped[str] = mapped_column(String(60), index=True)
    granted_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"),
                                                   nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PasswordReset(Base):
    """Single-use password-reset token — expires after one hour."""
    __tablename__ = "password_resets"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Customer(Base):
    """
    Pharmacy customers. The Customers screen existed with no backend at all —
    it rendered a hardcoded list.
    """
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    phone: Mapped[str] = mapped_column(String(40), default="", index=True)
    email: Mapped[str] = mapped_column(String(200), default="")
    branch: Mapped[str] = mapped_column(String(120), index=True)
    address: Mapped[str] = mapped_column(String(300), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    description: Mapped[str] = mapped_column(String(300), default="")


class Supplier(Base):
    __tablename__ = "suppliers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    contact_email: Mapped[str] = mapped_column(String(200), default="")
    phone: Mapped[str] = mapped_column(String(40), default="")
    lead_time_days: Mapped[int] = mapped_column(Integer, default=3)
    reliability_score: Mapped[float] = mapped_column(Float, default=0.9)
    min_order_value: Mapped[float] = mapped_column(Float, default=0.0)


class Product(Base):
    """A product within a branch. Fields match the frontend Product interface."""
    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("branch", "sku", name="uq_branch_sku"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(String(80), index=True)
    barcode: Mapped[str] = mapped_column(String(80), default="", index=True)
    # Set when a discount is applied, so the original price can be restored.
    original_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    discount_percent: Mapped[float] = mapped_column(Float, default=0.0)
    discount_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    name: Mapped[str] = mapped_column(String(300), index=True)
    category: Mapped[str] = mapped_column(String(120), default="", index=True)
    supplier: Mapped[str] = mapped_column(String(200), default="", index=True)
    branch: Mapped[str] = mapped_column(String(120), index=True)
    branch_id: Mapped[Optional[int]] = mapped_column(ForeignKey("branches.id"), nullable=True)

    purchase_price: Mapped[float] = mapped_column(Float, default=0.0)
    selling_price: Mapped[float] = mapped_column(Float, default=0.0)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    minimum_stock: Mapped[int] = mapped_column(Integer, default=0)
    maximum_stock: Mapped[int] = mapped_column(Integer, default=0)

    expiry_date: Mapped[str] = mapped_column(String(30), default="")
    batch_number: Mapped[str] = mapped_column(String(80), default="")

    daily_sales: Mapped[float] = mapped_column(Float, default=0.0)
    weekly_sales: Mapped[float] = mapped_column(Float, default=0.0)
    monthly_sales: Mapped[float] = mapped_column(Float, default=0.0)

    predicted_demand: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    recommended_action: Mapped[str] = mapped_column(Text, default="")
    expected_stockout_date: Mapped[str] = mapped_column(String(30), default="")

    risk_level: Mapped[str] = mapped_column(String(20), default="low")
    status: Mapped[str] = mapped_column(String(20), default="active")
    inventory_status: Mapped[str] = mapped_column(String(20), default="normal")

    last_restock_date: Mapped[str] = mapped_column(String(30), default="")
    last_updated_date: Mapped[str] = mapped_column(String(40), default="")

    # Link to the real Egyptian catalogue (25,065 items) once the matcher hits
    catalog_product_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Link to a training time series — without it the forecaster has no history
    # to work from. Assigned at seed time by nearest sales rate, and dropped
    # once there's enough real sales history.
    ml_pharmacy_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ml_product_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    branch_ref: Mapped["Branch"] = relationship(back_populates="products")


class Sale(Base):
    __tablename__ = "sales"
    id: Mapped[int] = mapped_column(primary_key=True)
    branch: Mapped[str] = mapped_column(String(120), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price: Mapped[float] = mapped_column(Float, default=0.0)
    total: Mapped[float] = mapped_column(Float, default=0.0)
    sold_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    customer_name: Mapped[str] = mapped_column(String(200), default="")


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    order_number: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    branch: Mapped[str] = mapped_column(String(120), index=True)
    supplier: Mapped[str] = mapped_column(String(200), index=True)
    # pending | approved | shipped | delivered | cancelled
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    total_amount: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expected_delivery: Mapped[str] = mapped_column(String(30), default="")
    # "purchase" or "return". A return carries negative quantities and total.
    kind: Mapped[str] = mapped_column(String(20), default="purchase", index=True)

    items: Mapped[List["PurchaseOrderItem"]] = relationship(
        back_populates="order", cascade="all, delete-orphan")


class PurchaseOrderItem(Base):
    __tablename__ = "purchase_order_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("purchase_orders.id"))
    product_id: Mapped[Optional[int]] = mapped_column(ForeignKey("products.id"), nullable=True)
    # Set when the line was ordered from a supplier's catalogue listing.
    listing_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("supplier_listings.id"), nullable=True)
    product_name: Mapped[str] = mapped_column(String(300), default="")
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    unit_price: Mapped[float] = mapped_column(Float, default=0.0)
    line_total: Mapped[float] = mapped_column(Float, default=0.0)

    order: Mapped["PurchaseOrder"] = relationship(back_populates="items")


class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    branch: Mapped[str] = mapped_column(String(120), default="", index=True)
    title: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(40), default="info")  # info|warning|critical|success
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class UserInvite(Base):
    """Employee invite — single-use token, expires after a week."""
    __tablename__ = "user_invites"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    invited_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    job_title: Mapped[str] = mapped_column(String(120), default="")
    department: Mapped[str] = mapped_column(String(120), default="")
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class TransferRequest(Base):
    """
    A pharmacy-to-pharmacy stock transfer request.

    Stock moves ONLY after the destination branch accepts. Until then the
    request is pending and both sides can see it; the destination gets a
    notification and can accept or reject.
    """
    __tablename__ = "transfer_requests"
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    sku: Mapped[str] = mapped_column(String(80), default="")
    product_name: Mapped[str] = mapped_column(String(300), default="")
    from_branch: Mapped[str] = mapped_column(String(120), index=True)
    to_branch: Mapped[str] = mapped_column(String(120), index=True)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    # pending | accepted | rejected | cancelled
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    requested_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"),
                                                     nullable=True)
    decided_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"),
                                                   nullable=True)
    decision_note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class AppSetting(Base):
    """
    Per-user application settings, persisted server-side.

    The Settings screen used to write everything to localStorage, so "saved"
    settings evaporated on another machine or after clearing site data.
    Values are JSON strings keyed by section ("organization", "notifications").
    """
    __tablename__ = "app_settings"
    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_user_setting"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    key: Mapped[str] = mapped_column(String(60), index=True)
    value: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow,
                                                 onupdate=utcnow)


class SupplierListing(Base):
    """
    A medicine a supplier stocks and offers to pharmacies.

    Suppliers manage their own catalogue (name, price, available stock) and
    can attach an offer (a discount valid until a date). Pharmacies search
    these listings by drug name when placing orders. An order placed against
    a supplier's own listing is auto-approved — the listing IS the supplier's
    standing acceptance.
    """
    __tablename__ = "supplier_listings"
    id: Mapped[int] = mapped_column(primary_key=True)
    supplier_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    supplier_name: Mapped[str] = mapped_column(String(200), index=True)
    product_name: Mapped[str] = mapped_column(String(300), index=True)
    category: Mapped[str] = mapped_column(String(120), default="", index=True)
    unit_price: Mapped[float] = mapped_column(Float, default=0.0)
    available_qty: Mapped[int] = mapped_column(Integer, default=0)
    min_order_qty: Mapped[int] = mapped_column(Integer, default=1)
    # Offer: percentage off unit_price, active until offer_until (or forever
    # while offer_until is NULL and percent > 0).
    offer_percent: Mapped[float] = mapped_column(Float, default=0.0)
    offer_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow,
                                                 onupdate=utcnow)


class OrderMessage(Base):
    """Chat between the ordering pharmacy and the supplier, per purchase order."""
    __tablename__ = "order_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("purchase_orders.id"), index=True)
    sender_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    sender_name: Mapped[str] = mapped_column(String(200), default="")
    # pharmacy | supplier | admin — which side of the conversation sent it
    sender_side: Mapped[str] = mapped_column(String(20), default="pharmacy")
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class PosConnection(Base):
    """
    A registered POS / external-system connection for a branch.

    The old Integrations screen faked everything client-side. Connections now
    live here, and "Test Connection" performs a real TCP reachability check —
    the recorded status is whatever actually happened.
    """
    __tablename__ = "pos_connections"
    id: Mapped[int] = mapped_column(primary_key=True)
    branch: Mapped[str] = mapped_column(String(120), index=True)
    name: Mapped[str] = mapped_column(String(200))
    system_type: Mapped[str] = mapped_column(String(80), default="")
    host: Mapped[str] = mapped_column(String(200))
    port: Mapped[int] = mapped_column(Integer, default=1433)
    database_name: Mapped[str] = mapped_column(String(120), default="")
    # unchecked | connected | failed — set only by real checks
    status: Mapped[str] = mapped_column(String(20), default="unchecked")
    last_checked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SyncLog(Base):
    """Real activity log for POS connections — one row per actual check."""
    __tablename__ = "sync_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    connection_id: Mapped[int] = mapped_column(ForeignKey("pos_connections.id"), index=True)
    branch: Mapped[str] = mapped_column(String(120), index=True)
    action: Mapped[str] = mapped_column(String(30), default="test")
    ok: Mapped[bool] = mapped_column(Boolean, default=False)
    detail: Mapped[str] = mapped_column(Text, default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class ChatMessage(Base):
    """AI assistant chat log — keeps memory across sessions."""
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    conversation_id: Mapped[str] = mapped_column(String(80), index=True)
    role: Mapped[str] = mapped_column(String(20))  # user|assistant
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[str] = mapped_column(Text, default="")  # JSON: charts, tools used
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
