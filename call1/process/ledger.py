"""The conversations this Process installation created graphs for, so the console can list them.

Store is the only owner of call and queue data; this is an index of IDs (conversation, call,
graphs) plus the operator's label for the recording, kept as append-only JSON lines in Process's
data directory. Progress and results are always read from Store.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._offset = 0
        with self._lock:
            self._refresh()

    def _refresh(self) -> None:
        """Fold lines appended since the last read (``python -m call1.process ingest`` appends to
        the same file while ``serve`` runs)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self._offset:  # replaced or truncated: start over
            self._entries, self._offset = {}, 0
        if size == self._offset:
            return
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        complete = chunk[: chunk.rfind(b"\n") + 1]
        self._offset += len(complete)
        for line in complete.decode("utf-8", errors="replace").splitlines():
            try:
                self._fold(json.loads(line))
            except (ValueError, TypeError, AttributeError):
                continue

    def _fold(self, event: Dict[str, Any]) -> None:
        cid = event.get("conversation_id")
        if not cid:
            return
        entry = self._entries.setdefault(cid, {"conversation_id": cid, "call_id": None, "label": None, "graphs": [], "first_seen_at": event.get("at")})
        if event.get("call_id"):
            entry["call_id"] = event["call_id"]
        if event.get("label") and not entry.get("label"):
            entry["label"] = event["label"]
        graph = event.get("graph_id")
        if graph and graph not in [g["graph_id"] for g in entry["graphs"]]:
            entry["graphs"].append({"graph_id": graph, "reason": event.get("reason"), "at": event.get("at")})
        entry["last_seen_at"] = event.get("at")

    def record(self, *, conversation_id: str, call_id: Optional[str] = None, graph_id: Optional[str] = None, reason: Optional[str] = None,
               label: Optional[str] = None) -> None:
        event = {"conversation_id": conversation_id, "call_id": call_id, "graph_id": graph_id, "reason": reason, "label": label,
                 "at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._refresh()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            self._refresh()

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._refresh()
            entry = self._entries.get(conversation_id)
            return json.loads(json.dumps(entry)) if entry else None

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            self._refresh()
            items = sorted(self._entries.values(), key=lambda e: e.get("last_seen_at") or "", reverse=True)
            return json.loads(json.dumps(items[:limit]))

    def latest_at(self) -> Optional[datetime]:
        """When the newest ledger event (an ingest or a graph) was recorded, or None."""
        with self._lock:
            self._refresh()
            stamps = [e.get("last_seen_at") for e in self._entries.values() if e.get("last_seen_at")]
        if not stamps:
            return None
        try:
            return max(datetime.fromisoformat(s) for s in stamps)
        except ValueError:
            return None

    def count(self) -> int:
        with self._lock:
            self._refresh()
            return len(self._entries)
