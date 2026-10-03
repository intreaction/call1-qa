"""The Contact Signals v2 taxonomy, its settings, alert rules and text redaction (contract 1.3.0;
docs/ContactSignalsV2.md sections 7.2, 9.1, 9.2, 9.4 and 9.6). It mirrors ``rubric_store.py``.

* **One document, versioned whole.** ``results_signal_taxonomy`` is a singleton record (current
  version, ``record_version``, settings); ``results_signal_taxonomy_versions`` holds immutable
  versions. 042_signals.sql seeds version 1 with the built-ins only. A save whose
  ``taxonomy_digest`` equals the current version's returns the record unchanged; any other save
  publishes N+1. A stale ``expected_record_version`` is 409 ``signal_taxonomy_conflict``.
* **Save-time rules.** The contract models enforce the built-in rules, reserved IDs, forbidden PII
  classes and the outer ceilings. Store adds the section 9.6 caps from its *effective*
  ``ContractParameters`` (``signals.signal_taxonomy_cap_violations``) and runs the Masker's
  rule-based detectors (``call1.redaction.find_pii``: SSN, card, phone, account number, PIN and
  digit runs) over every text path (``signals.signal_taxonomy_text_paths``). A refusal is
  ``validation_failed`` with ``details.field`` naming the path, never the value.
* **Redaction** stores ``signals.redact_signal_taxonomy_text`` of a non-current version and keeps
  its digest; a redacted version cannot be minted into a snapshot.
* Audit details and change events carry IDs, versions and digests, never taxonomy text.

Other areas reach this module only through ``results/api.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from call1.contracts.common import ContractParameters
from call1.contracts.errors import ErrorCode
from call1.contracts.events import (
    SIGNAL_SETTINGS_STATUS,
    SIGNAL_TAXONOMY_ID,
    Actor,
    AuditAction,
    ChangeKind,
    signal_taxonomy_saved_status,
)
from call1.contracts.signals import (
    SignalAlertRule,
    SignalAlertRuleRecord,
    SignalAlertRuleSave,
    SignalSettings,
    SignalSettingsSave,
    SignalTaxonomy,
    SignalTaxonomyRecord,
    SignalTaxonomyRedaction,
    SignalTaxonomySave,
    SignalTaxonomyVersion,
    redact_signal_taxonomy_text,
    signal_alert_condition_problem,
    signal_alert_node_active,
    signal_taxonomy_cap_violations,
    signal_taxonomy_text_paths,
    taxonomy_digest,
)
from call1.redaction import find_pii

from .. import audit, db, feed
from ..db import StoreConnection
from ..errors import StoreError, not_found

# --- reads -----------------------------------------------------------------------------------


def _version(row) -> SignalTaxonomyVersion:
    return SignalTaxonomyVersion(
        version=int(row["version"]),
        digest=row["digest"],
        taxonomy=SignalTaxonomy.model_validate(db.loads(row["taxonomy_json"])),
        published_at=db.parse_ts(row["published_at"]),
        published_by_account_id=row["published_by_account_id"],
        notes=row["notes"],
        text_redacted=bool(row["text_redacted"]),
        redacted_at=db.parse_ts(row["redacted_at"]),
        redacted_by_account_id=row["redacted_by_account_id"],
    )


def record_row(conn: StoreConnection):
    row = conn.execute("SELECT * FROM results_signal_taxonomy WHERE id = 1").fetchone()
    if row is None:  # 042_signals.sql always seeds it
        raise RuntimeError("results_signal_taxonomy is not seeded; run the Store migrations")
    return row


def get_version(conn: StoreConnection, version: int) -> Optional[SignalTaxonomyVersion]:
    row = conn.execute("SELECT * FROM results_signal_taxonomy_versions WHERE version = ?", (version,)).fetchone()
    return None if row is None else _version(row)


def list_versions(conn: StoreConnection) -> List[SignalTaxonomyVersion]:
    return [_version(r) for r in conn.execute("SELECT * FROM results_signal_taxonomy_versions ORDER BY version DESC").fetchall()]


def current_version(conn: StoreConnection) -> SignalTaxonomyVersion:
    found = get_version(conn, int(record_row(conn)["current_version"]))
    assert found is not None, "the current signal taxonomy version exists"
    return found


def settings(conn: StoreConnection) -> SignalSettings:
    return SignalSettings.model_validate(db.loads(record_row(conn)["settings_json"]))


def record(conn: StoreConnection) -> SignalTaxonomyRecord:
    row = record_row(conn)
    return SignalTaxonomyRecord(
        current=current_version(conn),
        settings=SignalSettings.model_validate(db.loads(row["settings_json"])),
        record_version=int(row["record_version"]),
        updated_at=db.parse_ts(row["updated_at"]),
        updated_by_account_id=row["updated_by_account_id"],
    )


# --- save-time rules --------------------------------------------------------------------------


def _refuse(message: str, field: str, **details) -> StoreError:
    return StoreError(ErrorCode.VALIDATION_FAILED, message, details={"field": field, **details})


def check_text(path: str, text: str) -> None:
    """Section 9.4: refuse definition text that one of the Masker's rule-based detectors matches.
    The error names the path and the detector's label, never the value."""
    found = find_pii(text or "")
    if found:
        raise _refuse("Definition text looks like caller details (a number the PII rules mask). Describe the behavior instead.",
                      path, reason="sensitive_text", detector=found[0][0])


def check_taxonomy(taxonomy: SignalTaxonomy, parameters: ContractParameters) -> None:
    """The section 9.6 caps from ``parameters`` and the section 9.4 text detectors (save and preview)."""
    violations = signal_taxonomy_cap_violations(taxonomy, parameters)
    if violations:
        v = violations[0]
        raise _refuse(f"{v.cap} is {v.limit} (this has {v.actual})", v.field, reason="cap", cap=v.cap, limit=v.limit, actual=v.actual)
    for path, text in signal_taxonomy_text_paths(taxonomy):
        check_text(path, text)


def check_no_delete(old: SignalTaxonomy, new: SignalTaxonomy) -> None:
    """Section 7.2 "No delete": a category or subcategory once published stays in the document
    (``active: false`` retires it), so versions, results, feedback and alert rules keep resolving it."""
    for c in old.categories:
        found = new.category(c.category_id)
        if found is None:
            raise _refuse("A published category cannot be removed; set active to false to retire it", "categories", reason="node_removed",
                          node_id=c.category_id)
        kept = {s.subcategory_id for s in found.subcategories}
        for s in c.subcategories:
            if s.subcategory_id not in kept:
                index = new.categories.index(found)
                raise _refuse("A published subcategory cannot be removed; set active to false to retire it",
                              f"categories[{index}].subcategories", reason="node_removed", node_id=s.subcategory_id)


def changed_paths(old: SignalTaxonomy, new: SignalTaxonomy) -> List[str]:
    """Paths (in the new taxonomy) of categories and subcategories that are new or differ."""
    before = {c.category_id: c for c in old.categories}
    out: List[str] = []
    for i, c in enumerate(new.categories):
        o = before.get(c.category_id)
        if o is None:
            out.append(f"categories[{i}]")
            continue
        subs = {s.subcategory_id: s for s in o.subcategories}
        own = c.model_dump(exclude={"subcategories"}) != o.model_dump(exclude={"subcategories"})
        if own:
            out.append(f"categories[{i}]")
        for j, s in enumerate(c.subcategories):
            if subs.get(s.subcategory_id) != s:
                out.append(f"categories[{i}].subcategories[{j}]")
    return out


def _conflict(conn: StoreConnection, message: str) -> StoreError:
    row = record_row(conn)
    return StoreError(ErrorCode.SIGNAL_TAXONOMY_CONFLICT, message,
                      details={"current_version": int(row["current_version"]), "record_version": int(row["record_version"])})


# --- writes -----------------------------------------------------------------------------------


def save_taxonomy(conn: StoreConnection, body: SignalTaxonomySave, *, actor: Actor, account_id: Optional[str],
                  parameters: ContractParameters) -> Tuple[SignalTaxonomyRecord, Optional[int]]:
    """``saveSignalTaxonomy`` in the caller's transaction. Returns the record and the version it
    published (None for a no-op replay). Audits and appends the change event."""
    row = record_row(conn)
    if body.expected_record_version != int(row["record_version"]):
        raise _conflict(conn, "The signal taxonomy changed since you read it; re-read and re-apply")
    check_taxonomy(body.taxonomy, parameters)
    current = current_version(conn)
    digest = taxonomy_digest(body.taxonomy)
    if digest == current.digest:
        return record(conn), None
    check_no_delete(current.taxonomy, body.taxonomy)
    version = current.version + 1
    now = db.ts(conn.now())
    conn.execute(
        "INSERT INTO results_signal_taxonomy_versions (version, digest, taxonomy_json, published_at, published_by_account_id, notes) VALUES (?, ?, ?, ?, ?, ?)",
        (version, digest, db.dumps(body.taxonomy), now, account_id, body.notes),
    )
    conn.execute("UPDATE results_signal_taxonomy SET current_version = ?, record_version = record_version + 1, updated_at = ?, "
                 "updated_by_account_id = ? WHERE id = 1", (version, now, account_id))
    paths = changed_paths(current.taxonomy, body.taxonomy)
    audit.append(conn, actor=actor, action=AuditAction.SIGNAL_TAXONOMY_SAVED, target_kind="signal_taxonomy", target_id=SIGNAL_TAXONOMY_ID,
                 details={"version": version, "digest": digest, "changed_paths": ",".join(paths)[:1000], "changed_count": len(paths)})
    feed.append(conn, ChangeKind.SIGNAL_TAXONOMY, SIGNAL_TAXONOMY_ID, version, signal_taxonomy_saved_status(version))
    return record(conn), version


def save_settings(conn: StoreConnection, body: SignalSettingsSave, *, actor: Actor, account_id: Optional[str]) -> SignalTaxonomyRecord:
    row = record_row(conn)
    if body.expected_record_version != int(row["record_version"]):
        raise _conflict(conn, "The signal settings changed since you read them; re-read and re-apply")
    old = SignalSettings.model_validate(db.loads(row["settings_json"]))
    if old == body.settings:
        return record(conn)
    conn.execute("UPDATE results_signal_taxonomy SET settings_json = ?, record_version = record_version + 1, updated_at = ?, "
                 "updated_by_account_id = ? WHERE id = 1", (db.dumps(body.settings), db.ts(conn.now()), account_id))
    new_record_version = int(row["record_version"]) + 1
    audit.append(conn, actor=actor, action=AuditAction.SIGNAL_SETTINGS_CHANGED, target_kind="signal_taxonomy", target_id=SIGNAL_TAXONOMY_ID,
                 details={"old_pipeline": old.pipeline, "new_pipeline": body.settings.pipeline, "v1_fallback": body.settings.v1_fallback,
                          "old_detection": old.detection, "new_detection": body.settings.detection, "record_version": new_record_version})
    feed.append(conn, ChangeKind.SIGNAL_TAXONOMY, SIGNAL_TAXONOMY_ID, new_record_version, SIGNAL_SETTINGS_STATUS)
    return record(conn)


def set_pipeline(conn: StoreConnection, pipeline: str, *, actor: Actor) -> SignalTaxonomyRecord:
    """Host-side pipeline switch (demo seeding, ``CALL1_SIGNALS_PIPELINE`` for e2e): an audited
    settings save at the current record version. No-op when already set."""
    current = record(conn)
    wanted = current.settings.model_copy(update={"pipeline": pipeline})
    return save_settings(conn, SignalSettingsSave(settings=SignalSettings.model_validate(wanted.model_dump()),
                                                  expected_record_version=current.record_version), actor=actor, account_id=None)


def set_detection(conn: StoreConnection, detection: str, *, actor: Actor) -> SignalTaxonomyRecord:
    """Host-side detection switch (1.4.0, docs/SignalsEmbeddings.md): ``model`` (today's Gemma stages)
    or ``rules`` (categories whose recipe engine is rules are decided by the rules engine). An audited
    settings save at the current record version; a no-op when already set."""
    current = record(conn)
    wanted = current.settings.model_copy(update={"detection": detection})
    return save_settings(conn, SignalSettingsSave(settings=SignalSettings.model_validate(wanted.model_dump()),
                                                  expected_record_version=current.record_version), actor=actor, account_id=None)


def with_recipes(current: SignalTaxonomy, source: SignalTaxonomy) -> SignalTaxonomy:
    """``current`` with the rules-engine configuration of ``source`` (a seed or pack): each category's
    ``recipe`` by category ID, and ``rules`` (the bank and kNN settings). Nothing else changes, so an
    admin's on-stage edits to names, subcategories and fields are kept."""
    recipes = {c.category_id: c.recipe for c in source.categories}
    data = current.model_dump(mode="json")
    for c in data["categories"]:
        if c["category_id"] in recipes:
            recipe = recipes[c["category_id"]]
            c["recipe"] = recipe.model_dump(mode="json") if recipe is not None else None
    data["rules"] = source.rules.model_dump(mode="json") if source.rules is not None else None
    return SignalTaxonomy.model_validate(data)


def apply_recipes(conn: StoreConnection, source: SignalTaxonomy, *, actor: Actor, parameters: ContractParameters,
                  detection: Optional[str] = None) -> Tuple[SignalTaxonomyRecord, Optional[int]]:
    """Install a seed's recipes into the current taxonomy as an audited admin save (host side,
    ``python -m call1.store apply-signals-recipes``), then optionally set ``detection``. Publishes
    version N+1 only when a recipe changed; otherwise a no-op."""
    with db.transaction(conn):
        current = record(conn)
        merged = with_recipes(current.current.taxonomy, source)
        body = SignalTaxonomySave(taxonomy=merged, expected_record_version=current.record_version,
                                  notes="Rules-engine recipes installed from a seed (docs/SignalsEmbeddings.md).")
        result = save_taxonomy(conn, body, actor=actor, account_id=None, parameters=parameters)
        if detection is not None:
            return set_detection(conn, detection, actor=actor), result[1]
        return result


def redact(conn: StoreConnection, version: int, body: SignalTaxonomyRedaction, *, actor: Actor, account_id: str) -> SignalTaxonomyVersion:
    """``redactSignalTaxonomyText``: tombstone a non-current version's custom text (idempotent)."""
    found = get_version(conn, version)
    if found is None:
        raise not_found("Signal taxonomy version", version=version)
    if body.digest != found.digest:
        raise StoreError(ErrorCode.CONFLICT, "The digest is not this version's", details={"reason": "digest_mismatch", "version": version})
    if found.text_redacted:
        return found
    if version == int(record_row(conn)["current_version"]):
        raise StoreError(ErrorCode.CONFLICT, "The current taxonomy version cannot be redacted; publish a new version first",
                         details={"reason": "redact_current", "version": version})
    now = db.ts(conn.now())
    conn.execute("UPDATE results_signal_taxonomy_versions SET taxonomy_json = ?, text_redacted = 1, redacted_at = ?, redacted_by_account_id = ? "
                 "WHERE version = ?", (db.dumps(redact_signal_taxonomy_text(found.taxonomy)), now, account_id, version))
    audit.append(conn, actor=actor, action=AuditAction.SIGNAL_TAXONOMY_REDACTED, target_kind="signal_taxonomy_version", target_id=str(version),
                 details={"version": version, "digest": found.digest})
    return get_version(conn, version)


def apply_seed(conn: StoreConnection, body: SignalTaxonomySave, *, actor: Actor, parameters: ContractParameters,
               pipeline: Optional[str] = None) -> Tuple[SignalTaxonomyRecord, Optional[int]]:
    """Apply a seed taxonomy (``call1/store/seeds/*.json``) as an admin save, host side
    (``python -m call1.store apply-signals-seed``; ``--demo``). A seed already current is a no-op;
    otherwise ``expected_record_version`` must match (the seed is meant for a fresh install)."""
    with db.transaction(conn):
        if taxonomy_digest(body.taxonomy) == current_version(conn).digest:
            result: Tuple[SignalTaxonomyRecord, Optional[int]] = (record(conn), None)
        else:
            result = save_taxonomy(conn, body, actor=actor, account_id=None, parameters=parameters)
        if pipeline is not None:
            return set_pipeline(conn, pipeline, actor=actor), result[1]
        return result


# --- alert rules ------------------------------------------------------------------------------


def _rule_record(row, taxonomy: SignalTaxonomy) -> SignalAlertRuleRecord:
    rule = SignalAlertRule.model_validate(db.loads(row["rule_json"]))
    return SignalAlertRuleRecord(**rule.model_dump(), record_version=int(row["record_version"]),
                                 node_active=signal_alert_node_active(rule.condition, taxonomy),
                                 created_at=db.parse_ts(row["created_at"]), updated_at=db.parse_ts(row["updated_at"]),
                                 updated_by_account_id=row["updated_by_account_id"])


def list_alert_rules(conn: StoreConnection, *, enabled_only: bool = False) -> List[SignalAlertRuleRecord]:
    taxonomy = current_version(conn).taxonomy
    sql = "SELECT * FROM results_signal_alert_rules" + (" WHERE enabled = 1" if enabled_only else "") + " ORDER BY rule_id"
    return [_rule_record(r, taxonomy) for r in conn.execute(sql).fetchall()]


def get_alert_rule(conn: StoreConnection, rule_id: str) -> Optional[SignalAlertRuleRecord]:
    row = conn.execute("SELECT * FROM results_signal_alert_rules WHERE rule_id = ?", (rule_id,)).fetchone()
    return None if row is None else _rule_record(row, current_version(conn).taxonomy)


def active_alert_rules(conn: StoreConnection) -> List[SignalAlertRuleRecord]:
    """Enabled rules whose node is active: the only rules that can match."""
    return [r for r in list_alert_rules(conn, enabled_only=True) if r.node_active]


def save_alert_rule(conn: StoreConnection, rule_id: str, body: SignalAlertRuleSave, *, actor: Actor, account_id: str,
                    parameters: ContractParameters) -> SignalAlertRuleRecord:
    rule = body.rule
    if rule.rule_id != rule_id:
        raise _refuse("rule.rule_id must equal the rule in the path", "rule.rule_id")
    existing = conn.execute("SELECT * FROM results_signal_alert_rules WHERE rule_id = ?", (rule_id,)).fetchone()
    current = int(existing["record_version"]) if existing else 0
    if body.expected_record_version != current:
        raise StoreError(ErrorCode.CONFLICT, "The alert rule changed since you read it", details={"current_version": current, "reason": "record_version"})
    problem = signal_alert_condition_problem(rule.condition, current_version(conn).taxonomy)
    if problem is not None:
        raise _refuse("The condition does not fit the current taxonomy", "rule.condition", reason=problem)
    check_text("rule.name", rule.name)
    if existing is None:
        count = conn.execute("SELECT COUNT(*) AS n FROM results_signal_alert_rules").fetchone()["n"]
        if count >= parameters.max_signal_alert_rules:
            raise _refuse(f"At most {parameters.max_signal_alert_rules} signal alert rules", "rule", reason="cap", cap="max_signal_alert_rules",
                          limit=parameters.max_signal_alert_rules, actual=int(count) + 1)
    now = db.ts(conn.now())
    conn.execute(
        "INSERT INTO results_signal_alert_rules (rule_id, rule_json, enabled, record_version, created_at, updated_at, updated_by_account_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(rule_id) DO UPDATE SET rule_json = excluded.rule_json, enabled = excluded.enabled, "
        "record_version = excluded.record_version, updated_at = excluded.updated_at, updated_by_account_id = excluded.updated_by_account_id",
        (rule_id, db.dumps(rule), 1 if rule.enabled else 0, current + 1, now, now, account_id),
    )
    audit.append(conn, actor=actor, action=AuditAction.SIGNAL_ALERT_RULE_SAVED, target_kind="signal_alert_rule", target_id=rule_id,
                 details={"rule_id": rule_id, "record_version": current + 1, "enabled": rule.enabled})
    was_enabled = bool(existing["enabled"]) if existing else None
    status = "saved" if was_enabled is None or was_enabled == rule.enabled else ("enabled" if rule.enabled else "disabled")
    feed.append(conn, ChangeKind.SIGNAL_ALERT_RULE, rule_id, current + 1, status)
    return get_alert_rule(conn, rule_id)


def rule_names(conn: StoreConnection) -> Dict[str, str]:
    return {r["rule_id"]: SignalAlertRule.model_validate(db.loads(r["rule_json"])).name
            for r in conn.execute("SELECT rule_id, rule_json FROM results_signal_alert_rules").fetchall()}
