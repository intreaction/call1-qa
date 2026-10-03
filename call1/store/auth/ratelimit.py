"""In-process rate limits for the anonymous auth steps (sign-in begin, enrollment begin).

Sliding windows on Store's clock, kept in memory: Store runs as one process, and a restart clearing
them is acceptable. Keys are the client address and, for sign-in, a hash of the email, so one
address cannot probe many accounts quickly and many addresses cannot hammer one account. An email
is limited identically whether or not the account exists, so the limit reveals nothing either.
Exceeding a limit is 429 ``rate_limited`` with ``Retry-After``.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Optional

from call1.contracts.errors import ErrorCode

from ..errors import StoreError


@dataclass(frozen=True)
class Limit:
    name: str
    max_hits: int
    window_seconds: int


SIGN_IN_PER_CLIENT = Limit("sign_in_client", 30, 60)
SIGN_IN_PER_EMAIL = Limit("sign_in_email", 10, 300)
ENROLL_PER_CLIENT = Limit("enroll_client", 10, 60)

_MAX_KEYS = 20_000


class RateLimiter:
    def __init__(self, clock) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: "OrderedDict[str, Deque[float]]" = OrderedDict()

    def check(self, limit: Limit, key: str) -> None:
        """Count one hit for ``key`` under ``limit``; raise 429 when the window is full."""
        now = self._clock.now().timestamp()
        bucket_key = f"{limit.name}|{key}"
        with self._lock:
            hits = self._hits.get(bucket_key)
            if hits is None:
                hits = deque()
                self._hits[bucket_key] = hits
                while len(self._hits) > _MAX_KEYS:
                    self._hits.popitem(last=False)
            else:
                self._hits.move_to_end(bucket_key)
            horizon = now - limit.window_seconds
            while hits and hits[0] <= horizon:
                hits.popleft()
            if len(hits) >= limit.max_hits:
                retry_after = max(1, math.ceil(hits[0] + limit.window_seconds - now))
                raise StoreError(ErrorCode.RATE_LIMITED, "Too many attempts; wait and try again",
                                 details={"retry_after_seconds": retry_after}, headers={"Retry-After": str(retry_after)})
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def email_bucket(email_key: str) -> str:
    return hashlib.sha256(email_key.encode("utf-8")).hexdigest()[:32]


def limiter_for(store) -> RateLimiter:
    """The Store's limiter (one per running Store)."""
    limiter: Optional[RateLimiter] = getattr(store, "_auth_rate_limiter", None)
    if limiter is None:
        limiter = RateLimiter(store.clock)
        setattr(store, "_auth_rate_limiter", limiter)
    return limiter
