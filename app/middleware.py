"""
Normalise query parameter names.

All responses are camelCase, so the frontend naturally sends
`?sortBy=quantity` and `?conversationId=abc`. Python names parameters in
snake_case. This middleware adds a snake_case alias for any camelCase key,
so both spellings work.

Without it the frontend sends camelCase and the server answers 422 for a
"missing" required parameter — the worst kind of bug, because it only shows
up once the UI is actually wired to the API.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal

import re
from urllib.parse import parse_qsl, urlencode

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def camel_to_snake(k: str) -> str:
    return _CAMEL.sub("_", k).lower()


class QueryParamCaseMiddleware:
    """ASGI middleware — runs before routing, so it reaches the params before
    validation does."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("query_string"):
            qs = scope["query_string"].decode("latin-1")
            pairs = parse_qsl(qs, keep_blank_values=True)
            seen = {k for k, _ in pairs}
            extra = [(s, v) for k, v in pairs
                     if (s := camel_to_snake(k)) != k and s not in seen]
            if extra:
                scope = dict(scope)
                scope["query_string"] = urlencode(pairs + extra).encode("latin-1")
        await self.app(scope, receive, send)
