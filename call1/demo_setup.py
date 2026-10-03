"""DEMO MODE setup over the Store API: the retail policy, one signal alert rule and its queue rule.

``python -m call1.launch --demo`` runs :func:`apply_demo_setup` once Store is up and before the
sample calls are ingested, so every seeded call is scored with the policy and can match the alert.
``scripts/apply_demo_policy.py`` runs the same steps against an already-running demo (and can ask
for a QA reanalysis of every call so the existing scorecards pick the policy up).

What it does, each step idempotent (a second run changes nothing):

1. **Policy** (BUG D1). ``call1_standard_v2`` ships with SEC-01 "Caller ID & Verification" and
   COMP-01 "Mandatory Regulatory Disclosures" set to ``requires_policy`` with no
   ``policy_context``, so both are always FLAGGED and no call passes. This publishes the next
   version of the rubric with ``contextual_rubrics.RETAIL_DEMO_POLICY`` in those checks (draft
   save, then publish, as the demo admin). An open draft is replaced: the demo has no other author.
2. **Alert rule**: ``stock-check`` fires on the retail taxonomy's intent subcategory
   ``check_stock_availability`` ("Check stock / availability"), so Metrics > Alerts, the calls
   list's "Any alert" filter and the change feed have something to show.
3. **Queue rule**: ``signal-stock-check``, a SIGNAL-stream review-queue rule that targets that
   alert, so matching calls reach the review queue.

HTTP only (``httpx``, or any client with the same ``get/put/post`` shape such as FastAPI's
``TestClient``); it signs in through demo mode's ``/demo/sign-in`` (localhost only) and then uses
the ordinary ``/store/v1`` routes, which enforce roles and CSRF as usual. Nothing here runs outside
demo mode: with demo mode off, sign-in answers 404 and the setup stops.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

from call1.pipeline.contextual_rubrics import RETAIL_DEMO_POLICY, with_policy

V = "/store/v1"
DEMO_RUBRIC_ID = "call1_standard_v2"
DEMO_POLICY_NOTES = "Demo: retail verification and disclosure policy for SEC-01 and COMP-01 (call1.demo_setup)."

DEMO_ALERT_RULE: Dict[str, Any] = {
    "rule_id": "stock-check",
    "name": "Stock availability ask",
    "condition": {"category_id": "intent", "subcategory_id": "check_stock_availability"},
    "enabled": True,
}

DEMO_QUEUE_RULE: Dict[str, Any] = {
    "id": "signal-stock-check",
    "name": "Signal: stock availability ask",
    "stream": "SIGNAL",
    "enabled": True,
    "rank": 50,
    "distribution_strategy": "UNASSIGNED_CLAIM",
    "target_signal_alerts": [DEMO_ALERT_RULE["rule_id"]],
    "description": "Demo: calls where the caller asks whether an item is in stock (Contact Signals alert stock-check).",
}


class DemoSetupError(RuntimeError):
    pass


def _base(client) -> str:
    return str(client.base_url).rstrip("/")


def _check(response, what: str, ok=(200, 201, 204)):
    if response.status_code not in ok:
        raise DemoSetupError(f"{what}: HTTP {response.status_code} {response.text[:300]}")
    return response


def sign_in(client, persona: str = "admin") -> Dict[str, str]:
    """Demo sign-in; returns the headers every later request sends (cookie, CSRF, Origin)."""
    response = client.post("/demo/sign-in", json={"persona": persona}, headers={"Origin": _base(client)})
    if response.status_code == 404:
        raise DemoSetupError("Store is not in demo mode (/demo/sign-in answered 404); start it with python -m call1.launch --demo")
    _check(response, "demo sign-in")
    body = response.json()
    token = body["session"]["csrf_token"]
    headers = {"X-Call1-CSRF": token, "Origin": _base(client)}
    cookies = "; ".join(f"{name}={value}" for name, value in response.cookies.items())
    if cookies:
        headers["Cookie"] = cookies
    return headers


def apply_policy(client, headers: Dict[str, str], rubric_id: str = DEMO_RUBRIC_ID,
                 policy: Optional[Dict[str, str]] = None) -> Optional[int]:
    """Publish the next rubric version with the policy text; None when the current one has it."""
    current = _check(client.get(f"{V}/rubrics/{rubric_id}", headers=headers), f"read rubric {rubric_id}").json()
    definition, changed = with_policy(current["definition"], policy or RETAIL_DEMO_POLICY)
    if not changed:
        return None
    draft = client.get(f"{V}/rubrics/{rubric_id}/draft", headers=headers)
    expected = draft.json()["draft_revision"] if draft.status_code == 200 else 0
    saved = _check(client.put(f"{V}/rubrics/{rubric_id}/draft", json={"definition": definition, "expected_draft_revision": expected},
                              headers=headers), "save rubric draft").json()
    published = _check(client.post(f"{V}/rubrics/{rubric_id}/publish",
                                   json={"expected_current_version": current["ref"]["version"],
                                         "expected_draft_revision": saved["draft_revision"], "notes": DEMO_POLICY_NOTES},
                                   headers=headers), "publish rubric").json()
    return int(published["ref"]["version"])


def _items(client, path: str, headers: Dict[str, str], what: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    token = None
    while True:
        params: Dict[str, Any] = {"limit": 200}
        if token:
            params["page_token"] = token
        page = _check(client.get(path, params=params, headers=headers), what).json()
        items.extend(page.get("items") or [])
        token = page.get("next_page_token")
        if not token:
            return items


def apply_alert_rule(client, headers: Dict[str, str], rule: Dict[str, Any] = DEMO_ALERT_RULE) -> Optional[int]:
    """Create the alert rule; None when a rule with that ID already exists (on-stage edits are kept)."""
    existing = {r["rule_id"] for r in _items(client, f"{V}/signals/alert-rules", headers, "list alert rules")}
    if rule["rule_id"] in existing:
        return None
    saved = _check(client.put(f"{V}/signals/alert-rules/{rule['rule_id']}", json={"rule": rule, "expected_record_version": 0},
                              headers=headers), "save alert rule").json()
    return int(saved["record_version"])


def apply_queue_rule(client, headers: Dict[str, str], rule: Dict[str, Any] = DEMO_QUEUE_RULE) -> Optional[int]:
    """Create the SIGNAL queue rule; None when a rule with that ID already exists."""
    existing = {r["id"] for r in _items(client, f"{V}/review-queue/rules", headers, "list queue rules")}
    if rule["id"] in existing:
        return None
    saved = _check(client.put(f"{V}/review-queue/rules/{rule['id']}", json={"rule": rule, "expected_rule_version": 0},
                              headers=headers), "save queue rule").json()
    return int(saved["rule_version"])


def request_reanalysis(client, headers: Dict[str, str], kind: str = "qa", say: Callable[[str], None] = print) -> int:
    """Ask for a ``kind`` reanalysis of every call (Process scores QA with the current rubric
    version); returns how many were accepted. A call Store refuses (say, one still processing) is
    reported and skipped."""
    count = 0
    for call in _items(client, f"{V}/calls", headers, "list calls"):
        response = client.post(f"{V}/calls/{call['call_id']}/reanalysis-requests", json={"kind": kind, "note": "Demo policy applied"},
                               headers={**headers, "Idempotency-Key": f"demo-policy-{uuid.uuid4().hex}"})
        if response.status_code in (200, 201):
            count += 1
        else:
            say(f"  {call['call_id']}: reanalysis not requested (HTTP {response.status_code} {response.text[:160]})")
    return count


def apply_demo_setup(client, *, reanalyze: bool = False, say: Callable[[str], None] = print) -> Dict[str, Any]:
    """Sign in as the demo admin and apply the three steps; returns what changed."""
    headers = sign_in(client)
    summary: Dict[str, Any] = {}
    version = apply_policy(client, headers)
    summary["rubric_version"] = version
    say(f"Rubric {DEMO_RUBRIC_ID}: published version {version} with the retail verification and disclosure policy."
        if version else f"Rubric {DEMO_RUBRIC_ID}: already carries the retail policy.")
    try:
        created = apply_alert_rule(client, headers)
        summary["alert_rule"] = created
        say(f"Alert rule {DEMO_ALERT_RULE['rule_id']} ({DEMO_ALERT_RULE['name']}): "
            + ("created." if created else "already exists."))
        queued = apply_queue_rule(client, headers)
        summary["queue_rule"] = queued
        say(f"Queue rule {DEMO_QUEUE_RULE['id']}: " + ("created." if queued else "already exists."))
    except DemoSetupError as exc:  # e.g. a taxonomy without the retail intent: the policy still stands
        summary["alert_error"] = str(exc)
        say(f"Alert rule not applied: {exc}")
    if reanalyze:
        summary["reanalysis_requests"] = request_reanalysis(client, headers, say=say)
        say(f"Requested a QA reanalysis of {summary['reanalysis_requests']} calls; Process rescores them with the current rubric.")
    return summary
