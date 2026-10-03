"""Store's clock. Every timestamp Store writes comes from ``Store.clock`` (also reachable as
``conn.now()`` on a Store connection), so tests can move time for leases, sessions and grants."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Protocol

UTC = timezone.utc


class Clock(Protocol):
    def now(self) -> datetime:
        """The current time, timezone-aware UTC."""


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class ManualClock:
    """A clock that moves only when told to. Thread-safe."""

    def __init__(self, start: datetime | None = None) -> None:
        start = start or datetime(2026, 9, 25, 12, 0, 0, tzinfo=UTC)
        if start.tzinfo is None:
            raise ValueError("ManualClock needs an aware datetime")
        self._now = start.astimezone(UTC)
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float = 0, **kwargs: float) -> datetime:
        with self._lock:
            self._now = self._now + timedelta(seconds=seconds, **kwargs)
            return self._now

    def set(self, value: datetime) -> None:
        if value.tzinfo is None:
            raise ValueError("ManualClock needs an aware datetime")
        with self._lock:
            self._now = value.astimezone(UTC)
