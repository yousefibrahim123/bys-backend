"""Database layer — SQLite by default, no setup required."""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_columns() -> List[str]:
    """
    `Base.metadata.create_all` only creates missing tables — it never adds a
    column to a table that already exists. A database created before
    `email_verified` was introduced would keep going without it, and every
    query against users would fail with OperationalError.

    This adds only the missing columns and is safe to call on every startup.
    Pre-existing users get verified=1 so working accounts aren't suddenly
    locked out.
    """
    from sqlalchemy import inspect, text

    added: List[str] = []
    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return added

    wanted: Optional[Dict[str, List[Tuple[str, str, str]]]] = {
        "users": [("email_verified", "BOOLEAN DEFAULT 1", "1"),
                  ("organization_type", "VARCHAR(20)",
                   "CASE WHEN role = 'supplier' THEN 'distributor' "
                   "ELSE 'pharmacy' END"),
                  # Backfill: the predefined demo logins are the only accounts
                  # that were ever test accounts. Everyone else gets 0.
                  ("is_test_account", "BOOLEAN DEFAULT 0",
                   "CASE WHEN email IN ('admin@gmail.com', 'supplier@gmail.com', "
                   "'giza@gmail.com', 'cairo@gmail.com', 'alex@gmail.com', "
                   "'nascity@gmail.com') THEN 1 ELSE 0 END")],
        "products": [("original_price", "FLOAT", None),
                     ("discount_percent", "FLOAT DEFAULT 0", "0"),
                     ("discount_until", "DATETIME", None)],
        "purchase_orders": [("kind", "VARCHAR(20) DEFAULT 'purchase'",
                             "'purchase'")],
        "purchase_order_items": [("listing_id", "INTEGER", None)],
    }
    tables = set(insp.get_table_names())
    for table, columns in wanted.items():
        if table not in tables:
            continue
        existing = {c["name"] for c in insp.get_columns(table)}
        for name, ddl, backfill in columns:
            if name in existing:
                continue
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                if backfill is not None:
                    conn.execute(text(
                        f"UPDATE {table} SET {name} = {backfill} "
                        f"WHERE {name} IS NULL"))
            added.append(f"{table}.{name}")
    return added
