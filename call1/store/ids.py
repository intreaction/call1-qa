"""Store-minted identifiers.

IDs are ``<prefix>_<12 hex ms timestamp><12 hex random>``: opaque to clients (contract
``ResourceId``), roughly time-ordered so ``id asc`` is a sensible tiebreak after ``created_at``.
Conventional prefixes: conv, call, art, upl, job, grf, att, rcpt, use, rq, acct, cred, inv, sess,
key, inst, evt, rvw, rule, rub.
"""

from __future__ import annotations

import re
import secrets
import time

_PREFIX = re.compile(r"^[a-z][a-z0-9]{0,15}$")


def new_id(prefix: str) -> str:
    if not _PREFIX.match(prefix):
        raise ValueError(f"id prefix {prefix!r} must be short lower-case letters and digits")
    return f"{prefix}_{int(time.time() * 1000):012x}{secrets.token_hex(6)}"
