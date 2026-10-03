"""The contract error envelope for every non-2xx Store response.

Handlers raise ``StoreError(ErrorCode.X, "safe message", details={...})``; the HTTP status comes
from ``ERROR_HTTP_STATUS``. Framework errors are mapped too: request validation is 422
``validation_failed`` (with the first failing field, never the input value), unknown paths 404
``not_found``, and a locked or busy database 503 ``store_unavailable`` (retryable). Messages and
details are safe text: never transcript content, prompts, provider bodies or secrets.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any, Dict, Mapping, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from call1.contracts.common import JsonScalar
from call1.contracts.errors import ERROR_HTTP_STATUS, ErrorCode, ErrorResponse

log = logging.getLogger("call1.store")

_RETRYABLE = frozenset({ErrorCode.RATE_LIMITED, ErrorCode.STORE_UNAVAILABLE})


class StoreError(Exception):
    """An API error with a contract code. ``status`` defaults to ``ERROR_HTTP_STATUS[code]``."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: Optional[Mapping[str, JsonScalar]] = None,
        retryable: Optional[bool] = None,
        status: Optional[int] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.code = ErrorCode(code)
        self.message = message
        self.details: Dict[str, JsonScalar] = dict(details or {})
        self.retryable = (self.code in _RETRYABLE) if retryable is None else retryable
        self.status = status or ERROR_HTTP_STATUS[self.code]
        self.headers = dict(headers or {})


def not_found(what: str, **details: JsonScalar) -> StoreError:
    return StoreError(ErrorCode.NOT_FOUND, f"{what} not found", details=details)


def request_id_of(request: Request) -> Optional[str]:
    return getattr(request.state, "request_id", None)


def error_response(request: Request, code: ErrorCode, message: str, *, details: Optional[Mapping[str, Any]] = None,
                   retryable: bool = False, status: Optional[int] = None, headers: Optional[Mapping[str, str]] = None) -> JSONResponse:
    body = ErrorResponse(code=code, message=message[:2000], details=dict(details or {}), retryable=retryable, request_id=request_id_of(request))
    return JSONResponse(body.model_dump(mode="json"), status_code=status or ERROR_HTTP_STATUS[code], headers=dict(headers or {}))


async def _store_error(request: Request, exc: StoreError) -> JSONResponse:
    return error_response(request, exc.code, exc.message, details=exc.details, retryable=exc.retryable, status=exc.status, headers=exc.headers)


_HTTP_CODES = {
    400: ErrorCode.VALIDATION_FAILED,
    401: ErrorCode.UNAUTHENTICATED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    422: ErrorCode.VALIDATION_FAILED,
    429: ErrorCode.RATE_LIMITED,
    501: ErrorCode.NOT_IMPLEMENTED,
    503: ErrorCode.STORE_UNAVAILABLE,
}


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _HTTP_CODES.get(exc.status_code, ErrorCode.VALIDATION_FAILED if exc.status_code < 500 else ErrorCode.STORE_UNAVAILABLE)
    message = {404: "No such route", 405: "Method not allowed on this route"}.get(exc.status_code, "Request failed")
    # 405 keeps its HTTP status; the envelope code says the method/route pair does not exist.
    return error_response(request, code, message, status=exc.status_code, retryable=code in _RETRYABLE, headers=getattr(exc, "headers", None))


_TERM_LISTS = ("customer_terms", "disabled_pack_terms")
_TERM_RULE = re.compile(r"not a vocabulary term \((\w+)\)")
_TERM_TYPE_REASONS = {"string_too_long": "too_long", "string_too_short": "empty", "string_type": "character"}


def vocabulary_term_details(first: Mapping[str, Any]) -> Optional[Dict[str, JsonScalar]]:
    """``saveAsrVocabulary`` (contract 1.3.0, docs/DualAsr.md section 4): a term the contract model
    refused is reported like Store's own refusals, ``details.field``, ``details.reason`` (the
    ``vocabulary.vocabulary_term_problem`` code, ``duplicate_term`` or ``too_many_terms``) and
    ``details.index``, never the term. None for any other error."""
    loc = [part for part in first.get("loc", ()) if part != "body"]
    msg = str(first.get("msg", ""))
    for i, part in enumerate(loc):
        if part not in _TERM_LISTS:
            continue
        field = ".".join(str(p) for p in loc[:i + 1])
        rest = loc[i + 1:]
        if rest and isinstance(rest[0], int):
            found = _TERM_RULE.search(msg)
            reason = found.group(1) if found else _TERM_TYPE_REASONS.get(str(first.get("type")), "invalid_term")
            return {"field": field, "reason": reason, "index": rest[0]}
        if first.get("type") == "too_long":
            return {"field": field, "reason": "too_many_terms"}
        return None
    for name in _TERM_LISTS:
        if f"{name} lists one term twice" in msg:
            return {"field": ".".join([*(str(p) for p in loc), name]), "reason": "duplicate_term"}
    return None


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    first = errors[0] if errors else {}
    field = ".".join(str(part) for part in first.get("loc", ()))
    details: Dict[str, JsonScalar] = {"field": field[:200], "reason": str(first.get("msg", "invalid"))[:300], "error_count": len(errors)}
    if request.url.path.endswith("/vocabulary"):
        term = vocabulary_term_details(first)
        if term is not None:
            details = {**term, "error_count": len(errors)}
    return error_response(request, ErrorCode.VALIDATION_FAILED, "Request validation failed", details=details)


async def _sqlite_error(request: Request, exc: sqlite3.OperationalError) -> JSONResponse:
    text = str(exc).lower()
    if "locked" in text or "busy" in text:
        return error_response(request, ErrorCode.STORE_UNAVAILABLE, "Store is busy; retry shortly", retryable=True)
    raise exc  # a genuine SQL error is a bug: let it surface as a 500 (and fail the test that hit it)


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    # Starlette re-raises after sending this, so the server log (and TestClient) still see the bug.
    log.exception("unhandled error in %s %s", request.method, request.url.path)
    return error_response(request, ErrorCode.STORE_UNAVAILABLE, "Internal error; quote the request_id when reporting it")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StoreError, _store_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(sqlite3.OperationalError, _sqlite_error)
    app.add_exception_handler(Exception, _unhandled)
