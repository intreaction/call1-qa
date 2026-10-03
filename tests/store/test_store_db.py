"""SQLite connections, transactions and the numbered-migration framework."""

from __future__ import annotations

import shutil
import threading

import pytest

from call1.store import db
from call1.store.db import MIGRATIONS_DIR, Database, MigrationError, discover_migrations, migration_owner, split_statements


def test_core_migration_is_applied_and_placeholders_wait(store, conn):
    applied = {row["version"]: row["owner"] for row in conn.execute("SELECT version, owner FROM schema_migrations")}
    assert applied[10] == "core"
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"store_meta", "change_events", "audit_events", "object_uploads", "schema_migrations"} <= tables
    for version in (20, 30, 40):  # comment-only placeholders are not recorded until filled
        if not split_statements((MIGRATIONS_DIR / f"{version:03d}_{migration_owner(version)}.sql").read_text()):
            assert version not in applied
    for key in (db.META_DATASET_ID, db.META_FEED_EPOCH, db.META_TRANSFER_SECRET):
        assert db.get_meta(conn, key)


def test_every_migration_file_belongs_to_an_area():
    migrations = discover_migrations()
    assert [m.version for m in migrations][:1] == [10]
    assert {m.owner for m in migrations} <= {"core", "queue", "auth", "results"}
    assert {(m.version, m.owner) for m in migrations} >= {(10, "core"), (20, "queue"), (30, "auth"), (40, "results")}


def test_a_filled_placeholder_applies_and_a_changed_migration_is_refused(tmp_path):
    migrations = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, migrations, ignore=shutil.ignore_patterns("* 2.*"))
    (migrations / "020_queue.sql").write_text("-- placeholder\n")
    (migrations / "021_signal_requests.sql").write_text("-- placeholder (alters 020's tables)\n")
    database = Database(tmp_path / "store.db")
    database.initialize(migrations)
    (migrations / "020_queue.sql").write_text("CREATE TABLE queue_probe (id TEXT PRIMARY KEY);\n")
    with database.connection() as conn:
        assert db.migrate(conn, migrations) == [20]
        assert db.migrate(conn, migrations) == []
    (migrations / "020_queue.sql").write_text("CREATE TABLE queue_probe (id TEXT PRIMARY KEY, x INTEGER);\n")
    with database.connection() as conn, pytest.raises(MigrationError, match="changed after it was applied"):
        db.migrate(conn, migrations)


def test_a_failing_migration_leaves_nothing_behind(tmp_path):
    migrations = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, migrations, ignore=shutil.ignore_patterns("* 2.*"))
    (migrations / "030_auth.sql").write_text("CREATE TABLE auth_probe (id TEXT);\nINSERT INTO no_such_table VALUES (1);\n")
    database = Database(tmp_path / "store.db")
    with pytest.raises(Exception):
        database.initialize(migrations)
    with database.connection() as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'auth_probe'").fetchone() is None
        assert conn.execute("SELECT 1 FROM schema_migrations WHERE version = 30").fetchone() is None


def test_migration_numbers_outside_every_range_are_refused(tmp_path):
    (tmp_path / "055_orphan.sql").write_text("CREATE TABLE x (id TEXT);\n")
    with pytest.raises(MigrationError, match="outside every area"):
        discover_migrations(tmp_path)


def test_split_statements():
    sql = """-- header
CREATE TABLE a (id INTEGER);
CREATE TRIGGER a_t AFTER INSERT ON a BEGIN
    UPDATE a SET id = id WHERE id = NEW.id;
END;
-- trailing comment
"""
    statements = split_statements(sql)
    assert len(statements) == 2 and statements[1].startswith("CREATE TRIGGER") and statements[1].endswith("END;")
    for bad in ("BEGIN;\n", "COMMIT;\n", "PRAGMA foreign_keys = OFF;\n"):
        with pytest.raises(MigrationError):
            split_statements(bad)
    with pytest.raises(MigrationError, match="incomplete"):
        split_statements("CREATE TABLE b (id INTEGER)\n")


def test_connections_are_wal_with_foreign_keys(conn, clock):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.now() == clock.now()
    assert conn.isolation_level is None and not conn.in_transaction


def test_transactions_roll_back_and_nest_as_savepoints(conn):
    with pytest.raises(RuntimeError):
        with db.transaction(conn):
            conn.execute("INSERT INTO store_meta (key, value) VALUES ('t1', 'x')")
            raise RuntimeError("boom")
    assert db.get_meta(conn, "t1") is None
    with db.transaction(conn):
        conn.execute("INSERT INTO store_meta (key, value) VALUES ('outer', 'x')")
        with pytest.raises(ValueError):
            with db.transaction(conn):
                conn.execute("INSERT INTO store_meta (key, value) VALUES ('inner', 'x')")
                raise ValueError("inner fails")
        with db.transaction(conn):
            conn.execute("INSERT INTO store_meta (key, value) VALUES ('inner2', 'x')")
    assert db.get_meta(conn, "outer") == "x" and db.get_meta(conn, "inner") is None and db.get_meta(conn, "inner2") == "x"
    assert not conn.in_transaction


def test_writing_inside_a_read_snapshot_is_refused(conn):
    with db.read_snapshot(conn):
        with pytest.raises(RuntimeError, match="read_snapshot"):
            with db.transaction(conn):
                pass
    assert not conn.in_transaction


def test_concurrent_writers_serialize_without_lost_updates(store):
    with store.connection() as conn:
        db.set_meta(conn, "counter", "0")
    errors = []

    def worker():
        try:
            with store.connection() as own:
                for _ in range(25):
                    with db.transaction(own):
                        value = int(db.get_meta(own, "counter"))
                        own.execute("UPDATE store_meta SET value = ? WHERE key = 'counter'", (str(value + 1),))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    with store.connection() as conn:
        assert db.get_meta(conn, "counter") == "150"


def test_timestamps_round_trip_as_fixed_width_utc(clock):
    now = clock.now()
    text = db.ts(now)
    assert text == "2026-09-25T12:00:00.000000Z" and db.parse_ts(text) == now
    with pytest.raises(ValueError):
        db.ts(now.replace(tzinfo=None))
