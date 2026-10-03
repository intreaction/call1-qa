"""On-device training route of the results area (contract 1.3.0, team decision 28;
docs/OnDeviceTraining.md section 2): ``listTrainingLabels``.

Process only (``training:read``); the principal guard refuses sessions and keys without the scope.
A read with ``limit > 0`` appends ``training_labels_read`` (actor: the Process installation;
details: the cursor, the item count and counts per kind, never content). A count-only read
(``limit 0``), which the Process console polls, is not audited.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query

from call1.contracts.events import AuditAction
from call1.contracts.training import TrainingLabelPage, TrainingLabelQuery

from .. import audit
from ..db import StoreConnection, read_snapshot, transaction
from ..deps import current_principal, get_conn
from ..principals import Principal, require_service_key
from . import training_labels
from .routes import router


@router.operation("listTrainingLabels")
def list_training_labels(query: Annotated[TrainingLabelQuery, Query()], conn: StoreConnection = Depends(get_conn),
                         principal: Principal = Depends(current_principal)) -> TrainingLabelPage:
    key = require_service_key(principal)
    with read_snapshot(conn):
        page = training_labels.list_labels(conn, query)
    if query.limit > 0:
        with transaction(conn):
            audit.append(conn, actor=audit.actor_for(key), action=AuditAction.TRAINING_LABELS_READ, target_kind="installation",
                         target_id=key.installation_id, details=training_labels.audit_details(query, page))
    return page
