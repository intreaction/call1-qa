"""Functions the auth area offers the other Store areas.

Other areas import only this module (and never the auth tables). Each function takes the caller's
connection and reads inside whatever transaction the caller holds.
"""

from __future__ import annotations

from typing import List, Optional

from call1.contracts.auth import AccountStatus, ProcessInstallation, ReviewerAccount
from call1.contracts.common import ReviewerRole

from ..db import StoreConnection
from . import records


def get_account(conn: StoreConnection, account_id: str) -> Optional[ReviewerAccount]:
    """Review history, assignment and reviewer profiles show and validate accounts through this."""
    return records.get_account(conn, account_id)


def list_accounts(conn: StoreConnection, *, role: Optional[ReviewerRole] = None, status: Optional[AccountStatus] = None) -> List[ReviewerAccount]:
    """Every account matching the filters, oldest first."""
    return records.list_accounts(conn, role=role, status=status)


def get_installation(conn: StoreConnection, installation_id: str) -> Optional[ProcessInstallation]:
    row = records.installation_row(conn, installation_id)
    return None if row is None else records.installation_from_row(row)
