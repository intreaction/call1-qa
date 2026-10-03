"""Small helpers for the auth end-to-end tests (tests/e2e/test_auth_*.py).

They drive the real ceremonies over HTTP with the harness's ``StoreSession`` and software
authenticator; nothing here is minted and nothing imports Store code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .softauthn import SoftAuthenticator
from .stack import API, Stack, StoreSession, api_path


def code_of(response: httpx.Response) -> Optional[str]:
    """The contract error envelope's ``code`` (``{code, message, details}``), or None."""
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("code") if isinstance(body, dict) else None


def details_of(response: httpx.Response) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return (body.get("details") or {}) if isinstance(body, dict) else {}


def describe(response: httpx.Response) -> str:
    return f"{response.request.method} {response.request.url.path} -> {response.status_code} {response.text[:600]}"


def begin_sign_in(stack: Stack, email: str, *, client: Optional[httpx.Client] = None,
                  forwarded_for: Optional[str] = None) -> httpx.Response:
    """``POST /auth/sign-in/begin`` from an anonymous client (optionally a chosen X-Forwarded-For)."""
    headers = {"Origin": stack.store_url}
    if forwarded_for:
        headers["X-Forwarded-For"] = forwarded_for
    http = client or stack.http
    return http.post(api_path("/auth/sign-in/begin"), json={"email": email}, headers=headers)


def finish_sign_in(stack: Stack, client: httpx.Client, ceremony_id: str, credential: Dict[str, Any]) -> httpx.Response:
    return client.post(api_path("/auth/sign-in/finish"), json={"ceremony_id": ceremony_id, "credential": credential},
                       headers={"Origin": stack.store_url})


def sign_in_with(stack: Stack, email: str, authenticator: SoftAuthenticator, *, credential=None,
                 forwarded_for: Optional[str] = "10.250.0.1", mutate=None) -> httpx.Response:
    """A whole sign-in as ``email`` signed by ``authenticator`` (optionally a specific credential it
    holds, even one the account does not own), in a fresh cookie jar. ``mutate(credential_json)``
    may edit the assertion before it is sent. Returns the finish response."""
    client = httpx.Client(base_url=stack.store_url, timeout=30.0)
    begin = begin_sign_in(stack, email, client=client, forwarded_for=forwarded_for)
    assert begin.status_code == 200, describe(begin)
    options = begin.json()
    assertion = authenticator.assert_(options["options"], origin=stack.store_url, credential=credential)
    if mutate is not None:
        mutate(assertion)
    response = finish_sign_in(stack, client, options["ceremony_id"], assertion)
    response.extensions["e2e_client"] = client  # keep the jar reachable for callers that want it
    return response


def only_credential(authenticator: SoftAuthenticator):
    creds = list(authenticator.credentials.values())
    assert len(creds) == 1, f"expected one credential in the authenticator, found {len(creds)}"
    return creds[0]


def add_authenticator(person: StoreSession, *, new: Optional[SoftAuthenticator] = None, nickname: Optional[str] = "backup key",
                      reauth_kw: Optional[Dict[str, Any]] = None, check: bool = True) -> httpx.Response:
    """The step-up ceremony (``/auth/authenticators/begin`` + ``/finish``): a fresh user-verified
    assertion from ``person``'s authenticator, then a registration on ``new`` (a separate backup
    key by default). Returns the finish response."""
    stack = person.stack
    new = new or SoftAuthenticator()
    begin = person.post("/auth/authenticators/begin", json={"nickname": nickname})
    if begin.status_code != 200:
        assert not check, describe(begin)
        return begin
    body = begin.json()
    reauth = person.authenticator.assert_(body["reauthentication"], origin=stack.store_url, **(reauth_kw or {}))
    credential = new.register(body["options"], origin=stack.store_url)
    finish = person.post("/auth/authenticators/finish",
                         json={"ceremony_id": body["ceremony_id"], "credential": credential, "reauthentication": reauth, "nickname": nickname})
    if check:
        assert finish.status_code == 200, describe(finish)
    finish.extensions["e2e_authenticator"] = new
    return finish


def clone_cookie_client(stack: Stack, person: StoreSession) -> httpx.Client:
    """A second 'tab' of the same browser: a new client holding only the session cookie (no CSRF token in memory)."""
    client = httpx.Client(base_url=stack.store_url, timeout=30.0)
    for name, value in person.http.cookies.items():
        client.cookies.set(name, value)
    return client


def audit_events(stack: Stack, admin: StoreSession, **query) -> List[Dict[str, Any]]:
    response = admin.get("/admin/audit", params={"limit": 200, **query})
    assert response.status_code == 200, describe(response)
    return response.json()["items"]


__all__ = ["API", "code_of", "details_of", "describe", "begin_sign_in", "finish_sign_in", "sign_in_with", "only_credential",
           "add_authenticator", "clone_cookie_client", "audit_events"]
