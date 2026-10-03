"""ASR vocabulary routes of the results area (contract 1.3.0, team decision 33; docs/DualAsr.md
section 4): ``getAsrVocabulary`` (admin with ``manage_vocabulary``, or Process ``jobs:write``) and
``saveAsrVocabulary`` (admin). The route guard applies the contract's principals; the rules live in
``vocabulary.py``."""

from __future__ import annotations

from fastapi import Depends

from call1.contracts.vocabulary import AsrVocabularyRecord, AsrVocabularySave

from .. import audit
from ..context import Store
from ..db import StoreConnection, read_snapshot, transaction
from ..deps import current_principal, get_conn, get_store
from ..principals import Principal, require_session
from . import vocabulary
from .routes import router


@router.operation("getAsrVocabulary")
def get_asr_vocabulary(conn: StoreConnection = Depends(get_conn)) -> AsrVocabularyRecord:
    with read_snapshot(conn):
        return vocabulary.record(conn)


@router.operation("saveAsrVocabulary")
def save_asr_vocabulary(body: AsrVocabularySave, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                        principal: Principal = Depends(current_principal)) -> AsrVocabularyRecord:
    session = require_session(principal)
    with transaction(conn):
        return vocabulary.save(conn, body, actor=audit.actor_for(principal), account_id=session.account_id, parameters=store.config.parameters)
