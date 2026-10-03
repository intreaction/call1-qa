"""The ASR vocabulary for dual transcription (contract 1.3.0, team decision 33; docs/DualAsr.md
section 4): ``getAsrVocabulary``, ``saveAsrVocabulary`` and the local pack install.

* **One document.** ``results_asr_vocabulary`` (044_asr_vocabulary.sql) is a singleton row: the
  admin-edited settings (``enabled``, ``customer_terms``, ``disabled_pack_terms``), the installed
  industry pack and ``record_version`` (0 until the first save or install). The effective terms,
  their digest and ``active`` are derived with the contract functions
  (``vocabulary.effective_vocabulary``, ``vocabulary_digest``, ``vocabulary_active``) on every read.
* **Save-time rules** (``check_settings``). The contract models already refuse a term that fails
  ``vocabulary_term_problem`` (so no digit) and a term listed twice. Store adds the cap from its
  *effective* ``ContractParameters`` (``max_vocabulary_terms``, reason ``too_many_terms``), that every
  ``disabled_pack_terms`` entry is a term of the installed pack (``unknown_pack_term``), and the
  Masker's rule-based detectors (``call1.redaction.find_pii``, as the signal taxonomy save runs them)
  over every term (``pii_detected``). Each refusal is ``validation_failed`` with ``details.field``,
  ``details.reason`` and, for one term, ``details.index``: never the term.
* **Pack install** (``install_pack``; ``python -m call1.store apply-vocabulary-seed``, ``--demo``)
  checks ``max_vocabulary_pack_terms`` and the detectors, replaces the pack, drops disabled terms the
  new pack no longer has and bumps the version. Installing the pack already installed is a no-op.
* Audit details (``asr_vocabulary_saved``) and change events (``asr_vocabulary``, ``saved`` or
  ``pack_installed``) carry versions, digests, the pack ID and counts, never terms.

Edits never re-run calls: Process freezes the effective terms into new ``asr`` jobs
(``JobParameters.asr_vocabulary``), and a full reanalysis uses the current vocabulary.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from call1.contracts.common import ContractParameters, JsonScalar
from call1.contracts.errors import ErrorCode
from call1.contracts.events import ASR_VOCABULARY_ID, Actor, AuditAction, ChangeKind
from call1.contracts.vocabulary import (
    AsrVocabularyPack,
    AsrVocabularyRecord,
    AsrVocabularySave,
    AsrVocabularySettings,
    VocabularyTermSource,
    effective_vocabulary,
    vocabulary_active,
    vocabulary_digest,
    vocabulary_term_key,
    vocabulary_term_problem,
)
from call1.redaction import find_pii

from .. import audit, db, feed
from ..db import StoreConnection
from ..errors import StoreError

SAVED_STATUS = "saved"
PACK_INSTALLED_STATUS = "pack_installed"


# --- reads -----------------------------------------------------------------------------------


def _row(conn: StoreConnection):
    row = conn.execute("SELECT * FROM results_asr_vocabulary WHERE id = 1").fetchone()
    if row is None:  # 044_asr_vocabulary.sql always seeds it
        raise RuntimeError("results_asr_vocabulary is not seeded; run the Store migrations")
    return row


def _settings(row) -> AsrVocabularySettings:
    return AsrVocabularySettings(enabled=bool(row["enabled"]), customer_terms=db.loads(row["customer_terms_json"]),
                                 disabled_pack_terms=db.loads(row["disabled_pack_terms_json"]))


def _pack(row) -> Optional[AsrVocabularyPack]:
    return AsrVocabularyPack.model_validate(db.loads(row["pack_json"])) if row["pack_json"] else None


def _record(row) -> AsrVocabularyRecord:
    settings, pack = _settings(row), _pack(row)
    effective = effective_vocabulary(pack, settings)
    return AsrVocabularyRecord(
        record_version=int(row["record_version"]),
        settings=settings,
        pack=pack,
        effective_terms=effective,
        effective_digest=vocabulary_digest(effective) if effective else None,
        active=vocabulary_active(settings, effective),
        updated_at=db.parse_ts(row["updated_at"]),
        updated_by_account_id=row["updated_by_account_id"],
    )


def record(conn: StoreConnection) -> AsrVocabularyRecord:
    """The current ``AsrVocabularyRecord`` (``getAsrVocabulary``)."""
    return _record(_row(conn))


# --- save-time rules --------------------------------------------------------------------------


def _refuse(message: str, field: str, reason: str, **details: JsonScalar) -> StoreError:
    return StoreError(ErrorCode.VALIDATION_FAILED, message, details={"field": field, "reason": reason, **details})


def check_terms(field: str, terms: List[str]) -> None:
    """Every term passes the contract's term rule (the models already enforce it; checked again so a
    stored or seeded document cannot bypass it) and no rule-based PII detector matches it. The
    refusal names the field, the reason and the index, never the term."""
    for index, term in enumerate(terms):
        problem = vocabulary_term_problem(term)
        if problem is not None:
            raise _refuse("Business terms only: letters, spaces and ' ’ & . - (no digits), at most 6 words and 60 characters",
                          field, problem, index=index)
        found = find_pii(term)
        if found:
            raise _refuse("A vocabulary term looks like caller details (a number the PII rules mask). Business terms only.",
                          field, "pii_detected", index=index, detector=found[0][0])


def check_settings(settings: AsrVocabularySettings, pack: Optional[AsrVocabularyPack], parameters: ContractParameters) -> None:
    """The save-time rules of ``saveAsrVocabulary`` (``validation_failed``, never the term)."""
    cap = parameters.max_vocabulary_terms
    if len(settings.customer_terms) > cap:
        raise _refuse(f"At most {cap} of your own vocabulary terms (this has {len(settings.customer_terms)})",
                      "settings.customer_terms", "too_many_terms", cap="max_vocabulary_terms", limit=cap, actual=len(settings.customer_terms))
    check_terms("settings.customer_terms", settings.customer_terms)
    pack_keys = {vocabulary_term_key(t) for t in pack.terms} if pack is not None else set()
    for index, term in enumerate(settings.disabled_pack_terms):
        if vocabulary_term_key(term) not in pack_keys:
            raise _refuse("Only terms of the installed industry pack can be switched off", "settings.disabled_pack_terms",
                          "unknown_pack_term", index=index)
    check_terms("settings.disabled_pack_terms", settings.disabled_pack_terms)


def check_pack(pack: AsrVocabularyPack, parameters: ContractParameters) -> None:
    """The install-time rules of a pack: ``max_vocabulary_pack_terms`` and the detectors."""
    cap = parameters.max_vocabulary_pack_terms
    if len(pack.terms) > cap:
        raise _refuse(f"An industry vocabulary pack holds at most {cap} terms (this has {len(pack.terms)})", "terms", "too_many_terms",
                      cap="max_vocabulary_pack_terms", limit=cap, actual=len(pack.terms))
    check_terms("terms", pack.terms)


def _conflict(current: int) -> StoreError:
    return StoreError(ErrorCode.CONFLICT, "The vocabulary changed since you read it; reload and re-apply your edits",
                      details={"current_version": current, "reason": "record_version"})


# --- writes -----------------------------------------------------------------------------------


def audit_details(before: AsrVocabularyRecord, after: AsrVocabularyRecord) -> Dict[str, JsonScalar]:
    """Versions, digests, the pack and counts per source: never a term."""
    by_source = {s: sum(1 for t in after.effective_terms if t.source is s) for s in VocabularyTermSource}
    return {
        "record_version": after.record_version,
        "old_digest": before.effective_digest,
        "new_digest": after.effective_digest,
        "enabled": after.settings.enabled,
        "active": after.active,
        "pack_id": after.pack.pack_id if after.pack else None,
        "pack_version": after.pack.version if after.pack else None,
        "pack_terms": len(after.pack.terms) if after.pack else 0,
        "customer_terms": len(after.settings.customer_terms),
        "disabled_pack_terms": len(after.settings.disabled_pack_terms),
        "effective_pack_terms": by_source[VocabularyTermSource.INDUSTRY_PACK],
        "effective_customer_terms": by_source[VocabularyTermSource.CUSTOMER],
    }


def _write(conn: StoreConnection, settings: AsrVocabularySettings, pack: Optional[AsrVocabularyPack], account_id: Optional[str]) -> None:
    conn.execute(
        "UPDATE results_asr_vocabulary SET record_version = record_version + 1, enabled = ?, customer_terms_json = ?, "
        "disabled_pack_terms_json = ?, pack_json = ?, updated_at = ?, updated_by_account_id = ? WHERE id = 1",
        (1 if settings.enabled else 0, db.dumps(list(settings.customer_terms)), db.dumps(list(settings.disabled_pack_terms)),
         db.dumps(pack) if pack is not None else None, db.ts(conn.now()), account_id),
    )


def save(conn: StoreConnection, body: AsrVocabularySave, *, actor: Actor, account_id: Optional[str],
         parameters: ContractParameters) -> AsrVocabularyRecord:
    """``saveAsrVocabulary`` in the caller's transaction: replace the settings whole. A stale
    ``expected_record_version`` is 409 ``conflict`` with ``details.current_version``; a save that
    changes nothing returns the record unchanged (no audit, no event)."""
    row = _row(conn)
    current = int(row["record_version"])
    if body.expected_record_version != current:
        raise _conflict(current)
    pack = _pack(row)
    check_settings(body.settings, pack, parameters)
    before = _record(row)
    if before.settings == body.settings:
        return before
    _write(conn, body.settings, pack, account_id)
    after = record(conn)
    audit.append(conn, actor=actor, action=AuditAction.ASR_VOCABULARY_SAVED, target_kind="asr_vocabulary", target_id=ASR_VOCABULARY_ID,
                 details={**audit_details(before, after), "change": SAVED_STATUS})
    feed.append(conn, ChangeKind.ASR_VOCABULARY, ASR_VOCABULARY_ID, after.record_version, SAVED_STATUS)
    return after


def install_pack(conn: StoreConnection, pack: AsrVocabularyPack, *, actor: Actor, parameters: ContractParameters) -> Tuple[AsrVocabularyRecord, bool]:
    """Install (or replace) the industry pack, host side (``python -m call1.store
    apply-vocabulary-seed``; ``--demo``). Returns the record and whether anything changed: the pack
    already installed is a no-op. Disabled terms the new pack no longer has are dropped. Audited
    with the installer as actor; change event ``pack_installed``."""
    check_pack(pack, parameters)
    with db.transaction(conn):
        row = _row(conn)
        before = _record(row)
        if before.pack == pack:
            return before, False
        keys = {vocabulary_term_key(t) for t in pack.terms}
        kept = [t for t in before.settings.disabled_pack_terms if vocabulary_term_key(t) in keys]
        settings = before.settings.model_copy(update={"disabled_pack_terms": kept})
        _write(conn, settings, pack, None)
        after = record(conn)
        audit.append(conn, actor=actor, action=AuditAction.ASR_VOCABULARY_SAVED, target_kind="asr_vocabulary", target_id=ASR_VOCABULARY_ID,
                     details={**audit_details(before, after), "change": PACK_INSTALLED_STATUS})
        feed.append(conn, ChangeKind.ASR_VOCABULARY, ASR_VOCABULARY_ID, after.record_version, PACK_INSTALLED_STATUS)
        return after, True


__all__ = ["PACK_INSTALLED_STATUS", "SAVED_STATUS", "audit_details", "check_pack", "check_settings", "check_terms", "install_pack", "record", "save"]
