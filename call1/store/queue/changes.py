"""Change events the queue area appends, collected per transaction.

A queue write touches several jobs (a completion releases dependents, a cancel cascades), so the
handler records what changed on a ``ChangeLog`` and calls ``flush()`` once, inside the same
transaction, before it builds its receipt. ``flush`` appends one ``job`` event per changed job,
one ``reanalysis_request`` event per changed request, then one ``job_group`` event per touched
conversation (status ``settled`` or ``active``), and returns the last cursor for the receipt.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from call1.contracts.events import ChangeKind

from .. import feed
from .records import ConversationJobs, call_id_of


class ChangeLog:
    def __init__(self, conn) -> None:
        self.conn = conn
        self._jobs: Dict[str, Tuple[str, str]] = {}
        self._requests: Dict[str, Tuple[str, str, str]] = {}
        self._conversations: List[str] = []
        self._catalog: List[str] = []
        self.cursor: Optional[str] = None

    def _touch(self, conversation_id: str) -> None:
        if conversation_id not in self._conversations:
            self._conversations.append(conversation_id)

    def job(self, job_id: str, status: str, conversation_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._jobs[job_id] = (str(getattr(status, "value", status)), conversation_id)
        self._touch(conversation_id)

    def request(self, request_id: str, status: str, conversation_id: str, call_id: str) -> None:
        self._requests.pop(request_id, None)
        self._requests[request_id] = (str(getattr(status, "value", status)), conversation_id, call_id)

    def conversation(self, conversation_id: str) -> None:
        self._touch(conversation_id)

    def catalog(self, installation_id: str) -> None:
        if installation_id not in self._catalog:
            self._catalog.append(installation_id)

    def flush(self) -> str:
        """Append every collected event in the open transaction; return the last cursor."""
        calls: Dict[str, Optional[str]] = {}

        def call_of(conversation_id: str) -> Optional[str]:
            if conversation_id not in calls:
                calls[conversation_id] = call_id_of(self.conn, conversation_id)
            return calls[conversation_id]

        cursor = None
        for job_id, (status, conversation_id) in self._jobs.items():
            cursor = feed.append(self.conn, ChangeKind.JOB, job_id, None, status, conversation_id=conversation_id, call_id=call_of(conversation_id))
        for request_id, (status, conversation_id, call_id) in self._requests.items():
            cursor = feed.append(self.conn, ChangeKind.REANALYSIS_REQUEST, request_id, None, status, conversation_id=conversation_id, call_id=call_id)
        for conversation_id in self._conversations:
            settled = ConversationJobs(self.conn, conversation_id).settled()
            cursor = feed.append(self.conn, ChangeKind.JOB_GROUP, conversation_id, None, "settled" if settled else "active",
                                 conversation_id=conversation_id, call_id=call_of(conversation_id))
        for installation_id in self._catalog:
            cursor = feed.append(self.conn, ChangeKind.CATALOG, installation_id, None, "published")
        self._jobs.clear()
        self._requests.clear()
        self._conversations.clear()
        self._catalog.clear()
        self.cursor = cursor or feed.latest_cursor(self.conn)
        return self.cursor
