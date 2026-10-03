"""Ceremony helpers for auth tests: bootstrap the first admin, invite and enroll reviewers, sign in.

Each ``Person`` has its own ``TestClient`` (its own cookie jar, like a browser) and its own
software authenticator. Requests go through the real HTTP routes; nothing is minted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fastapi.testclient import TestClient

from call1.contracts.auth import SetupCodeIssueRequest
from call1.store.auth import cli as auth_cli

from .auth_softauthn import ORIGIN, SoftAuthenticator

API = "/store/v1"


def new_client(app) -> TestClient:
    return TestClient(app, base_url=ORIGIN)


@dataclass
class Person:
    client: TestClient
    authn: SoftAuthenticator
    email: str
    session: dict = field(default_factory=dict)

    @property
    def account_id(self) -> str:
        return self.session["account_id"]

    @property
    def csrf(self) -> dict:
        return {"X-Call1-CSRF": self.session["csrf_token"]}

    def get(self, path: str, **kw):
        return self.client.get(API + path, **kw)

    def write(self, method: str, path: str, **kw):
        headers = {**kw.pop("headers", {}), **self.csrf}
        return self.client.request(method, API + path, headers=headers, **kw)

    def sign_in(self, **assert_kw):
        response = sign_in(self.client, self.authn, self.email, **assert_kw)
        if response.status_code == 200:
            self.session = response.json()["session"]
        return response


def setup_code(store, email: str, display_name: str = "First Admin", purpose: str = "first_admin",
               target_account_id: Optional[str] = None) -> str:
    code, _ = auth_cli.issue_setup_code(store, SetupCodeIssueRequest(purpose=purpose, email=email, display_name=display_name,
                                                                     target_account_id=target_account_id, os_user="call1store"))
    return code


def enroll(client: TestClient, authn: SoftAuthenticator, *, setup_code: Optional[str] = None, invitation_token: Optional[str] = None,
           nickname: Optional[str] = None, **register_kw):
    body = {"setup_code": setup_code} if setup_code else {"invitation_token": invitation_token}
    begin = client.post(f"{API}/auth/enroll/begin", json=body)
    if begin.status_code != 200:
        return begin
    options = begin.json()
    credential = authn.register(options["options"], **register_kw)
    finish = {"ceremony_id": options["ceremony_id"], "credential": credential}
    if nickname:
        finish["nickname"] = nickname
    return client.post(f"{API}/auth/enroll/finish", json=finish)


def sign_in(client: TestClient, authn: SoftAuthenticator, email: str, **assert_kw):
    begin = client.post(f"{API}/auth/sign-in/begin", json={"email": email})
    if begin.status_code != 200:
        return begin
    options = begin.json()
    credential = authn.assert_(options["options"], **assert_kw)
    return client.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": options["ceremony_id"], "credential": credential})


def bootstrap_admin(app, email: str = "admin@example.com", display_name: str = "First Admin") -> Person:
    store = app.state.store
    person = Person(new_client(app), SoftAuthenticator(), email)
    response = enroll(person.client, person.authn, setup_code=setup_code(store, email, display_name), nickname="Admin key")
    assert response.status_code == 200, response.text
    person.session = response.json()["signed_in"]["session"]
    return person


def invite(admin: Person, email: str, role: str = "reviewer", display_name: Optional[str] = None, **extra) -> str:
    response = admin.write("POST", "/admin/invitations", json={"email": email, "display_name": display_name or email.split("@")[0],
                                                              "role": role, **extra})
    assert response.status_code == 200, response.text
    return response.json()["invitation_url"].split("#", 1)[1]


def invite_and_enroll(app, admin: Person, email: str, role: str = "reviewer") -> Person:
    token = invite(admin, email, role)
    person = Person(new_client(app), SoftAuthenticator(), email)
    response = enroll(person.client, person.authn, invitation_token=token)
    assert response.status_code == 200, response.text
    person.session = response.json()["signed_in"]["session"]
    return person
