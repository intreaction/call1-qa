"""Opaque page tokens for ``Page[T]`` list routes (keyset pagination).

A token is base64url JSON of the last item's sort key, e.g. ``encode(created_at_text, id)``.
``decode(token, arity)`` returns that list, or None for the first page, and answers 422
``validation_failed`` for a token that is not one of ours. Tokens fit ``PageToken`` (<= 256
characters of ``[A-Za-z0-9_-]``); keep sort keys short.
"""

from __future__ import annotations

import base64
import json
from typing import Any, List, Optional

from call1.contracts.errors import ErrorCode

from .errors import StoreError

MAX_TOKEN = 256


def encode(*values: Any) -> str:
    raw = json.dumps(list(values), separators=(",", ":"), ensure_ascii=True).encode()
    token = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    if len(token) > MAX_TOKEN:
        raise ValueError("page token sort key too long; use shorter keys")
    return token


def decode(token: Optional[str], arity: int) -> Optional[List[Any]]:
    if token is None:
        return None
    try:
        values = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
        if not isinstance(values, list) or len(values) != arity:
            raise ValueError
    except (ValueError, TypeError):
        raise StoreError(ErrorCode.VALIDATION_FAILED, "page_token is not valid for this list", details={"field": "page_token"}) from None
    return values
