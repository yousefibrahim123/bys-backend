"""
API entry point.

    uvicorn app.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .database import Base, engine, ensure_columns
from .middleware import QueryParamCaseMiddleware
from .ml.registry import registry
from .routers import (ai, auth, expiry, health, integrations, ml, operations,
                      products, purchase_orders, reference,
                      settings as settings_router, supplier_catalog, transfers,
                      users)
from .seed import seed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    added = ensure_columns()          # add new columns to an older database
    if added:
        log.info("migrated columns: %s", ", ".join(added))
    result = seed()
    log.info("seed: %s", result)
    try:
        registry.warmup()          # preload models so the first request isn't slow
        log.info("ML models warmed up")
    except Exception as e:         # pragma: no cover
        log.warning("warmup skipped: %s", e)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Smart pharmacy supply-chain platform.\n\n"
        "Three built-in models: demand forecasting (LightGBM quantile), "
        "expiry risk (LightGBM binary), and drug-name matching against the "
        "Egyptian catalogue (25,065 items). Plus an AI assistant grounded in "
        "live data."),
    lifespan=lifespan,
)

# Registered before CORS so CORS ends up outermost and handles preflight
app.add_middleware(QueryParamCaseMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=(["*"] if settings.cors_origins.strip() == "*"
                   else [o.strip() for o in settings.cors_origins.split(",") if o.strip()]),
    allow_credentials=False,   # browsers reject credentials when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """A readable error instead of the deeply nested default shape."""
    issues = [{"field": ".".join(str(x) for x in e["loc"][1:]) or "body",
               "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        content={"detail": "Invalid request body", "issues": issues})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):  # pragma: no cover
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": "Unexpected server error",
                                 "type": type(exc).__name__})


for r in (health.router, auth.router, products.router, operations.inventory,
          operations.sales, operations.misc, operations.analytics,
          purchase_orders.router, users.router, ml.router, ai.router,
          # Writes for reference data, plus customers and outgoing
          # notifications. operations.misc already serves the GET lists, and
          # FastAPI merges routers sharing a prefix, so these add the missing
          # verbs rather than replacing anything.
          reference.suppliers, reference.categories, reference.branches,
          reference.customers, reference.notifications, expiry.router,
          transfers.router, settings_router.router, supplier_catalog.router,
          integrations.router):
    app.include_router(r)


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "name": settings.app_name, "version": settings.version,
        "docs": "/docs", "health": "/health",
        "endpoints": {
            "auth": "/api/auth", "products": "/api/products",
            "inventory": "/api/inventory", "sales": "/api/sales",
            "purchaseOrders": "/api/purchase-orders", "ml": "/api/ml",
            "aiAssistant": "/api/ai/chat",
            "users": "/api/users", "customers": "/api/customers",
            "suppliers": "/api/suppliers", "notifications": "/api/notifications",
        },
    }
