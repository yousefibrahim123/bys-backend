"""
API schemas.

Every response is emitted in camelCase to match the frontend exactly
(`sellingPrice`, not `selling_price`), while input accepts both.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


def camel(s: str) -> str:
    head, *rest = s.split("_")
    return head + "".join(w.capitalize() for w in rest)


class Schema(BaseModel):
    model_config = ConfigDict(alias_generator=camel, populate_by_name=True,
                              from_attributes=True)

    @field_serializer("*", when_used="json-unless-none")
    def _utc_datetimes(self, v: Any, _info: Any) -> Any:
        """
        Timestamps are stored as naive UTC. Serialised without a timezone
        marker the browser parses them as *local* time, so a just-created
        notification showed as "3 hours ago" in Egypt. Emit an explicit
        +00:00 offset instead.
        """
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


# ------------------------------------------------------------------- auth

class LoginRequest(BaseModel):
    # The frontend sends email or mobile in the same field
    email: str
    password: str


class RegisterRequest(Schema):
    email: str
    password: str
    name: str
    phone: str = ""
    # Self-registration can never produce an administrator. Admin accounts are
    # created only by an existing admin through the invite flow.
    role: Literal["branch", "supplier"] = "branch"
    organization_type: Literal["pharmacy", "distributor"] = "pharmacy"
    branch: Optional[str] = None


class UserOut(Schema):
    id: int
    email: str
    phone: str = ""
    name: str
    role: str
    organization_type: str = "pharmacy"
    branch: Optional[str] = None
    # Effective permissions (role defaults + explicit grants). The UI gates
    # navigation and sections on this so what you see is what you may do.
    permissions: List[str] = []


class TokenResponse(Schema):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class RefreshRequest(Schema):
    refresh_token: str


class RegisterResponse(Schema):
    """
    Registration returns no token — the email must be verified first.
    `verification_url` is only returned when SMTP isn't configured, so
    development isn't blocked.
    """
    user: Optional[UserOut] = None
    email_sent: bool
    verification_url: Optional[str] = None
    expires_in_hours: int
    note: str


class VerifyEmailRequest(Schema):
    token: str


class VerifyTokenInfo(Schema):
    email: str
    name: str
    already_verified: bool = False
    expires_at: datetime


class EmailOnlyRequest(Schema):
    email: str


class ChangePasswordRequest(Schema):
    current_password: str
    new_password: str


class ResetPasswordRequest(Schema):
    token: str
    new_password: str


class ProfileUpdate(Schema):
    name: Optional[str] = None
    phone: Optional[str] = None


class UserUpdate(Schema):
    """Admin edit of another user. Every field optional — only what's sent changes."""
    name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[Literal["branch", "admin", "supplier"]] = None
    branch: Optional[str] = None
    is_active: Optional[bool] = None


class UserDetail(UserOut):
    is_active: bool = True
    email_verified: bool = False
    created_at: Optional[datetime] = None


class SimpleMessage(Schema):
    message: str


class PermissionsUpdate(Schema):
    permissions: List[str] = []


# ------------------------------------------------------------ expiry actions

class DiscountRequest(Schema):
    product_id: int
    percent: float
    days: Optional[int] = 30
    allow_below_cost: bool = False


class TransferRequest(Schema):
    product_id: int
    quantity: int
    to_branch: str


class ReturnRequest(Schema):
    product_id: int
    quantity: int
    reason: str = ""


class WriteOffRequest(Schema):
    product_id: int
    quantity: int
    reason: str = ""


class ExpiryActionOut(Schema):
    id: int
    product_id: int
    product_name: str = ""
    action: str
    detail: str = ""
    branch: str
    value_egp: float = 0.0
    reference_id: Optional[int] = None
    reverted: bool = False
    created_at: datetime


# ---------------------------------------------------------------- customers

class CustomerIn(Schema):
    name: str
    phone: str = ""
    email: str = ""
    branch: Optional[str] = None
    address: str = ""
    notes: str = ""
    is_active: bool = True


class CustomerOut(Schema):
    id: int
    name: str
    phone: str = ""
    email: str = ""
    branch: str
    address: str = ""
    notes: str = ""
    is_active: bool = True
    created_at: datetime


# ------------------------------------------------------- reference data CRUD

class SupplierIn(Schema):
    name: str
    contact_email: str = ""
    phone: str = ""
    lead_time_days: int = 3
    reliability_score: float = 0.9
    min_order_value: float = 0.0


class CategoryIn(Schema):
    name: str
    description: str = ""


class BranchIn(Schema):
    name: str
    address: str = ""
    phone: str = ""


class NotificationIn(Schema):
    title: str
    body: str = ""
    kind: Literal["info", "warning", "critical", "success"] = "info"
    branch: Optional[str] = None
    user_id: Optional[int] = None


class InviteCreate(Schema):
    email: str
    name: str
    phone: str = ""
    role: Literal["branch", "admin", "supplier"] = "branch"
    branch: Optional[str] = None
    job_title: str = ""
    department: str = ""
    # Ticked in the add-user wizard. None means "use the role defaults".
    permissions: Optional[List[str]] = None

class InviteOut(Schema):
    user: UserOut
    token: str
    activation_url: str
    expires_at: datetime
    email_sent: bool
    note: str


class InviteVerify(Schema):
    email: str
    name: str
    role: str
    branch: Optional[str] = None
    job_title: str = ""
    expires_at: datetime


class ActivateRequest(Schema):
    token: str
    password: str


# --------------------------------------------------------------- products

class AIPrediction(Schema):
    predicted_demand: float = 0
    confidence_score: float = 0
    recommended_action: str = ""
    expected_stockout_date: str = ""


class ProductOut(Schema):
    id: int
    sku: str
    barcode: str = ""
    name: str
    category: str = ""
    supplier: str = ""
    branch: str
    purchase_price: float = 0
    selling_price: float = 0
    quantity: int = 0
    minimum_stock: int = 0
    maximum_stock: int = 0
    expiry_date: str = ""
    batch_number: str = ""
    daily_sales: float = 0
    weekly_sales: float = 0
    monthly_sales: float = 0
    ai_prediction_data: AIPrediction
    risk_level: str = "low"
    status: str = "active"
    last_restock_date: str = ""
    last_updated_date: str = ""
    catalog_product_id: Optional[int] = None
    # Back-compat fields the frontend still reads
    qty: int = 0
    max_qty: int = 0
    inventory_status: str = "normal"
    expiry: str = ""
    price: float = 0

    @classmethod
    def from_orm_product(cls, p) -> "ProductOut":
        return cls(
            id=p.id, sku=p.sku, barcode=p.barcode, name=p.name, category=p.category,
            supplier=p.supplier, branch=p.branch, purchase_price=p.purchase_price,
            selling_price=p.selling_price, quantity=p.quantity,
            minimum_stock=p.minimum_stock, maximum_stock=p.maximum_stock,
            expiry_date=p.expiry_date, batch_number=p.batch_number,
            daily_sales=p.daily_sales, weekly_sales=p.weekly_sales,
            monthly_sales=p.monthly_sales,
            ai_prediction_data=AIPrediction(
                predicted_demand=p.predicted_demand, confidence_score=p.confidence_score,
                recommended_action=p.recommended_action,
                expected_stockout_date=p.expected_stockout_date),
            risk_level=p.risk_level, status=p.status,
            last_restock_date=p.last_restock_date, last_updated_date=p.last_updated_date,
            catalog_product_id=p.catalog_product_id,
            qty=p.quantity, max_qty=p.maximum_stock,
            inventory_status=p.inventory_status, expiry=p.expiry_date,
            price=p.selling_price,
        )


class ProductCreate(Schema):
    sku: str = ""
    barcode: str = ""
    name: str
    category: str = ""
    supplier: str = ""
    branch: Optional[str] = None
    purchase_price: float = 0
    selling_price: float = 0
    quantity: int = 0
    minimum_stock: int = 0
    maximum_stock: int = 0
    expiry_date: str = ""
    batch_number: str = ""
    daily_sales: float = 0
    status: str = "active"


class ProductUpdate(Schema):
    sku: Optional[str] = None
    barcode: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    supplier: Optional[str] = None
    purchase_price: Optional[float] = None
    selling_price: Optional[float] = None
    quantity: Optional[int] = None
    minimum_stock: Optional[int] = None
    maximum_stock: Optional[int] = None
    expiry_date: Optional[str] = None
    batch_number: Optional[str] = None
    daily_sales: Optional[float] = None
    weekly_sales: Optional[float] = None
    monthly_sales: Optional[float] = None
    risk_level: Optional[str] = None
    status: Optional[str] = None


class PageMeta(Schema):
    page: int
    page_size: int
    total: int
    total_pages: int


class ProductPage(Schema):
    items: List[ProductOut]
    meta: PageMeta


# -------------------------------------------------------------- inventory

class InventorySummary(Schema):
    branch: str
    total_products: int
    total_units: int
    inventory_value: float
    retail_value: float
    potential_profit: float
    low_stock_count: int
    out_of_stock_count: int
    near_expiry_count: int
    expired_count: int
    categories: int
    suppliers: int


# ------------------------------------------------------------------ sales

class SaleCreate(Schema):
    product_id: int
    quantity: int = 1
    unit_price: Optional[float] = None
    customer_name: str = ""


class SaleOut(Schema):
    id: int
    branch: str
    product_id: int
    product_name: str = ""
    quantity: int
    unit_price: float
    total: float
    sold_at: datetime
    customer_name: str = ""


class SalesSummary(Schema):
    branch: str
    days: int
    total_revenue: float
    total_units: int
    transactions: int
    average_basket: float
    daily: List[Dict[str, Any]]
    top_products: List[Dict[str, Any]]


# -------------------------------------------------------- purchase orders

class POItemIn(Schema):
    product_id: Optional[int] = None
    # Set when ordering from a supplier's catalogue listing — the server
    # takes the (offer) price from the listing, never from the client.
    listing_id: Optional[int] = None
    product_name: str = ""
    quantity: int
    unit_price: float = 0


class PurchaseOrderCreate(Schema):
    supplier: str
    items: List[POItemIn]
    notes: str = ""
    expected_delivery: str = ""


class POItemOut(Schema):
    id: int
    product_id: Optional[int] = None
    product_name: str
    quantity: int
    unit_price: float
    line_total: float


class PurchaseOrderOut(Schema):
    id: int
    order_number: str
    branch: str
    supplier: str
    status: str
    total_amount: float
    notes: str = ""
    expected_delivery: str = ""
    created_at: datetime
    items: List[POItemOut] = Field(default_factory=list)


class POStatusUpdate(Schema):
    status: Literal["pending", "approved", "rejected", "shipped", "delivered",
                    "cancelled"]


# -------------------------------------------------------------- suppliers

class SupplierOut(Schema):
    id: int
    name: str
    contact_email: str = ""
    phone: str = ""
    lead_time_days: int = 3
    reliability_score: float = 0.9
    min_order_value: float = 0


# ---------------------------------------------------------- notifications

class NotificationOut(Schema):
    id: int
    title: str
    body: str = ""
    kind: str = "info"
    is_read: bool = False
    branch: str = ""
    created_at: datetime


# ---------------------------------------------------------------- ML models

class ForecastRequest(Schema):
    product_id: int
    horizon_days: int = 28
    current_stock: Optional[int] = None

    @field_validator("horizon_days")
    @classmethod
    def _h(cls, v: int) -> int:
        if not 1 <= v <= 28:
            raise ValueError("horizon_days must be between 1 and 28")
        return v


class ForecastDay(Schema):
    date: str
    p50: float
    p90: float


class ForecastResponse(Schema):
    product_id: int
    product_name: str
    branch: str
    horizon_days: int
    daily: List[ForecastDay]
    total_p50: float
    total_p90: float
    safety_stock: float
    reorder_point: float
    current_stock: int
    recommended_order_qty: int
    days_of_cover: float
    stockout_date: Optional[str] = None
    model: str
    explanation: str


class ExpiryRequest(Schema):
    quantity: int
    days_to_expiry: int
    daily_sales_rate: float
    unit_cost: float


class ExpiryResponse(Schema):
    risk_probability: float
    risk_level: str
    expected_leftover_units: float
    expected_loss_egp: float
    action: str
    explanation: str
    model: str


class MatchRequest(Schema):
    query: str
    limit: int = 5
    catalog: Literal["real", "synthetic"] = "real"


class MatchCandidate(Schema):
    product_id: int
    name_ar: str
    name_en: str
    confidence: float
    price_egp: Optional[float] = None
    manufacturer: Optional[str] = None


class MatchResponse(Schema):
    query: str
    decision: str
    confidence: float
    best: Optional[MatchCandidate]
    alternatives: List[MatchCandidate]
    catalog: str
    catalog_size: int


class ReorderLine(Schema):
    product_id: int
    name: str
    supplier: str
    current_stock: int
    minimum_stock: int
    forecast_28d: float
    recommended_qty: int
    estimated_cost: float
    urgency: str
    reason: str


class ReorderResponse(Schema):
    branch: str
    generated_at: datetime
    lines: List[ReorderLine]
    total_estimated_cost: float
    model: str


# ----------------------------------------------------------- AI assistant

class ChatRequest(Schema):
    message: str
    conversation_id: Optional[str] = None
    branch: Optional[str] = None


class ChatChart(Schema):
    label: str
    data: List[Dict[str, Any]]


class ChatResponse(Schema):
    conversation_id: str
    reply: str
    chart: Optional[ChatChart] = None
    tools_used: List[str] = Field(default_factory=list)
    provider: str
    grounded: bool = True


class AIStatus(Schema):
    provider: str
    model: str
    llm_available: bool
    reason: str
    configured_providers: List[str]
    free_options: List[Dict[str, str]]


# ----------------------------------------------------------- stock transfers

class TransferCreate(Schema):
    product_id: int
    to_branch: str
    quantity: int
    note: str = ""


class TransferDecision(Schema):
    note: str = ""


class TransferOut(Schema):
    id: int
    product_id: int
    sku: str
    product_name: str
    from_branch: str
    to_branch: str
    quantity: int
    status: str
    note: str
    decision_note: str = ""
    requested_by: Optional[int] = None
    decided_by: Optional[int] = None
    created_at: datetime
    decided_at: Optional[datetime] = None


# ---------------------------------------------------------------- settings

class SettingsUpdate(Schema):
    """Partial update — only the keys sent are replaced."""
    settings: Dict[str, Any]


class POItemQtyUpdate(Schema):
    """Edit line quantities on a pending order. {item_id: new_quantity}"""
    quantities: Dict[int, int]


# -------------------------------------------------- supplier catalogue

class ListingIn(Schema):
    product_name: str
    category: str = ""
    unit_price: float = 0
    available_qty: int = 0
    min_order_qty: int = 1
    offer_percent: float = 0
    # ISO date/datetime; empty or None clears the deadline
    offer_until: Optional[str] = None
    notes: str = ""
    is_active: bool = True


class ListingOut(Schema):
    id: int
    supplier_user_id: int
    supplier_name: str
    supplier_phone: str = ""
    supplier_email: str = ""
    product_name: str
    category: str = ""
    unit_price: float
    available_qty: int
    min_order_qty: int
    offer_percent: float
    offer_until: Optional[datetime] = None
    offer_active: bool = False
    # unit_price after the active offer (equals unit_price when no offer)
    effective_price: float
    notes: str = ""
    is_active: bool
    created_at: datetime
    updated_at: datetime


# -------------------------------------------------------- order chat

class OrderMessageIn(Schema):
    body: str


class OrderMessageOut(Schema):
    id: int
    order_id: int
    sender_id: Optional[int] = None
    sender_name: str
    sender_side: str
    body: str
    created_at: datetime


# -------------------------------------------------- POS integrations

class PosConnectionIn(Schema):
    name: str
    system_type: str = ""
    host: str
    port: int = 1433
    database_name: str = ""
    branch: Optional[str] = None


class PosConnectionOut(Schema):
    id: int
    branch: str
    name: str
    system_type: str
    host: str
    port: int
    database_name: str
    status: str
    last_checked_at: Optional[datetime] = None
    last_error: str = ""
    created_at: datetime


class SyncLogOut(Schema):
    id: int
    connection_id: int
    branch: str
    action: str
    ok: bool
    detail: str
    duration_ms: int
    created_at: datetime
