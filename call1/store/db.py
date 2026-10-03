"""Store's SQLite database: connections, transactions, numbered migrations and core metadata.

Only Store opens this database (docs/SplitBuild.md, architecture rule 1).

Connections
    ``Database.connect()`` returns a ``StoreConnection``: autocommit mode (``isolation_level=None``)
    so transactions are explicit, ``sqlite3.Row`` rows, WAL journal, ``foreign_keys=ON``, a busy
    timeout, and ``conn.now()`` from Store's clock. A request gets its own connection
    (``context.get_conn``); connections are never shared between concurrent requests.

Transactions
    ``with transaction(conn):`` starts ``BEGIN IMMEDIATE`` (the write lock is taken up front, so
    concurrent writers queue on the busy timeout instead of failing on a lock upgrade) and commits,
    or rolls back on any exception. Nested use becomes a SAVEPOINT, so a hook called inside another
    area's transaction can use ``transaction(conn)`` too. SQLite serializes writers, so change-feed
    cursors assigned inside a transaction are in commit order. ``with read_snapshot(conn):`` gives
    a consistent multi-statement read; writing inside one is an error.

Migrations
    ``call1/store/migrations/NNN_name.sql`` files are applied in ascending order, each in its own
    transaction, and recorded in ``schema_migrations`` with a SHA-256 of the file. Each area owns a
    number range (``MIGRATION_OWNERS``). A migration may depend only on core tables and its own
    area's earlier migrations: no foreign keys into another area's tables (reference by ID). A file
    that changed after it was applied is an error; add a new file instead (in development, delete
    the data directory). A file with only comments (an unfilled placeholder) is skipped and not
    recorded. Migration files carry no BEGIN/COMMIT and no PRAGMA statements.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .clock import Clock, SystemClock

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_FILE = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")

MIGRATION_OWNERS: Dict[str, range] = {
    "core": range(10, 20),
    "queue": range(20, 30),
    "auth": range(30, 40),
    "results": range(40, 50),
}
"""Which area owns which migration numbers. 010 core, 020 queue, 030 auth, 040 results."""


class MigrationError(RuntimeError):
    pass


# --- timestamps and JSON --------------------------------------------------------------------

_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def ts(value: datetime) -> str:
    """Storage form of a timestamp: fixed-width UTC text, so text order is time order."""
    if value.tzinfo is None:
        raise ValueError("Store stores aware timestamps only")
    return value.astimezone(timezone.utc).strftime(_TS_FORMAT)


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.strptime(value, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def dumps(value: Any) -> str:
    """Compact, key-sorted JSON for TEXT columns. Pydantic models are dumped in JSON mode first."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def loads(value: Optional[str]) -> Any:
    return None if value is None else json.loads(value)


# --- connections ----------------------------------------------------------------------------


class StoreConnection(sqlite3.Connection):
    """A sqlite3 connection that knows Store's clock (``conn.now()``)."""

    clock: Clock
    txn_mode: Optional[str]

    def now(self) -> datetime:
        return self.clock.now()


def connect(path: Path, clock: Optional[Clock] = None, *, busy_timeout_ms: int = 10_000) -> StoreConnection:
    conn = sqlite3.connect(
        str(path),
        timeout=busy_timeout_ms / 1000,
        isolation_level=None,
        check_same_thread=False,
        factory=StoreConnection,
    )
    conn.clock = clock or SystemClock()
    conn.txn_mode = None
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


_savepoints = itertools.count(1)


@contextmanager
def transaction(conn: StoreConnection, *, immediate: bool = True) -> Iterator[StoreConnection]:
    """A write transaction (``BEGIN IMMEDIATE``), or a SAVEPOINT when one is already open."""
    if conn.in_transaction:
        if getattr(conn, "txn_mode", None) == "read":
            raise RuntimeError("cannot write inside read_snapshot(); open a transaction() instead")
        name = f"sp_{next(_savepoints)}"
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield conn
        except BaseException:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            conn.execute(f"RELEASE SAVEPOINT {name}")
            raise
        else:
            conn.execute(f"RELEASE SAVEPOINT {name}")
        return
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    conn.txn_mode = "write"
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
    finally:
        conn.txn_mode = None


@contextmanager
def read_snapshot(conn: StoreConnection) -> Iterator[StoreConnection]:
    """A consistent read across several statements. Inside an open transaction it is a no-op."""
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN")
    conn.txn_mode = "read"
    try:
        yield conn
    finally:
        if conn.in_transaction:
            conn.execute("COMMIT")
        conn.txn_mode = None


# --- migrations -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path
    checksum: str
    owner: str

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


def migration_owner(version: int) -> Optional[str]:
    for owner, numbers in MIGRATION_OWNERS.items():
        if version in numbers:
            return owner
    return None


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> List[Migration]:
    found: Dict[int, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _MIGRATION_FILE.match(path.name)
        if not match:
            continue  # includes iCloud conflict copies ("010_core 2.sql")
        version = int(match.group(1))
        owner = migration_owner(version)
        if owner is None:
            raise MigrationError(f"{path.name}: migration {version:03d} is outside every area's range ({MIGRATION_OWNERS})")
        if version in found:
            raise MigrationError(f"two migrations numbered {version:03d}: {found[version].path.name} and {path.name}")
        checksum = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        found[version] = Migration(version, match.group(2), path, checksum, owner)
    return [found[v] for v in sorted(found)]


def split_statements(sql: str) -> List[str]:
    """Split a migration file into complete SQL statements (triggers included)."""
    statements: List[str] = []
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            text = buffer.strip()
            if text:
                statements.append(text)
            buffer = ""
    rest = "\n".join(l for l in buffer.splitlines() if l.strip() and not l.strip().startswith("--"))
    if rest.strip():
        raise MigrationError(f"incomplete SQL statement at end of migration: {rest.strip()[:80]!r}")
    for statement in statements:
        head = re.sub(r"^(\s*--[^\n]*\n)*", "", statement + "\n").lstrip().upper()
        if head.startswith(("BEGIN", "COMMIT", "END", "ROLLBACK", "PRAGMA", "SAVEPOINT", "RELEASE", "VACUUM")):
            raise MigrationError(f"migrations may not contain {head.split()[0]} statements")
    return statements


_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    owner      TEXT NOT NULL,
    checksum   TEXT NOT NULL,
    applied_at TEXT NOT NULL
)
"""


def applied_migrations(conn: sqlite3.Connection) -> Dict[int, str]:
    conn.execute(_BOOTSTRAP)
    return {row["version"]: row["checksum"] for row in conn.execute("SELECT version, checksum FROM schema_migrations")}


def migrate(conn: StoreConnection, directory: Path = MIGRATIONS_DIR) -> List[int]:
    """Apply every pending migration in ascending order; return the versions applied."""
    migrations = discover_migrations(directory)
    applied_now: List[int] = []
    conn.execute(_BOOTSTRAP)
    for migration in migrations:
        statements = split_statements(migration.sql)
        with transaction(conn):
            applied = applied_migrations(conn)
            if migration.version in applied:
                if applied[migration.version] != migration.checksum:
                    raise MigrationError(
                        f"migration {migration.path.name} changed after it was applied to {conn_path(conn)}; "
                        "add a new numbered migration instead (in development, delete the Store data directory)"
                    )
                continue
            if not statements:
                continue  # a placeholder with comments only is not recorded, so it applies once filled
            for statement in statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, owner, checksum, applied_at) VALUES (?, ?, ?, ?, ?)",
                (migration.version, migration.name, migration.owner, migration.checksum, ts(conn.now())),
            )
            applied_now.append(migration.version)
    return applied_now


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
    return int(row["v"] or 0)


def conn_path(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    return row["file"] if row is not None else "?"


# --- core metadata --------------------------------------------------------------------------

META_DATASET_ID = "dataset_id"
META_FEED_EPOCH = "feed_epoch"
META_TRANSFER_SECRET = "transfer_secret"
META_CREATED_AT = "created_at"


def new_feed_epoch() -> str:
    return "ep" + secrets.token_hex(6)


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM store_meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def set_meta(conn: StoreConnection, key: str, value: str) -> None:
    with transaction(conn):
        conn.execute("INSERT INTO store_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def ensure_core_meta(conn: StoreConnection) -> None:
    """Seed the dataset ID, feed epoch and transfer-signing secret on a fresh database."""
    seeds = {
        META_DATASET_ID: "ds_" + secrets.token_hex(8),
        META_FEED_EPOCH: new_feed_epoch(),
        META_TRANSFER_SECRET: secrets.token_hex(32),
        META_CREATED_AT: ts(conn.now()),
    }
    with transaction(conn):
        for key, value in seeds.items():
            conn.execute("INSERT OR IGNORE INTO store_meta (key, value) VALUES (?, ?)", (key, value))


# --- the database -------------------------------------------------------------------------


class Database:
    """Store's database file. ``initialize()`` runs once at startup; ``connect()`` per request."""

    def __init__(self, path: Path, clock: Optional[Clock] = None, *, busy_timeout_ms: int = 10_000) -> None:
        self.path = Path(path)
        self.clock = clock or SystemClock()
        self.busy_timeout_ms = busy_timeout_ms

    def connect(self) -> StoreConnection:
        return connect(self.path, self.clock, busy_timeout_ms=self.busy_timeout_ms)

    @contextmanager
    def connection(self) -> Iterator[StoreConnection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self, migrations_dir: Path = MIGRATIONS_DIR) -> List[int]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RuntimeError(f"Store needs SQLite WAL mode; got {mode}")
            applied = migrate(conn, migrations_dir)
            ensure_core_meta(conn)
            return applied
