"""Scratch space with bounded retention, and the completion spool.

``scratch/`` holds recordings between upload and Store commit and each attempt's downloaded inputs.
An attempt's directory is removed when the attempt ends; ``sweep`` removes anything older than the
retention window and, oldest first, whatever exceeds the byte cap (leftovers from a crash).

``spool/`` holds the completion, failure or release request a worker is about to send, written
before the request and removed once Store answers. ``Worker.recover`` replays each one with the
same key and body, while Process runs (every few seconds, and as soon as Store is reachable again
after an outage) and at start-up after a crash: Store returns the original receipt when it had
committed, and otherwise applies it if the claim is still active. Those requests hold IDs,
checksums and measurements only, never content.

The one exception is a *deferred publication*: a job that finished while Store was unreachable,
so its outputs could not be uploaded. Its ``<job>.<attempt>.publish.json`` entry holds the claim
and the measurements, and its output files wait in ``spool/outputs/<job>.<attempt>/`` (mode 0600,
this appliance only) until ``recover`` uploads them and completes the job; they are removed then,
or when the claim turns out to be stale or cancelled.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


class Scratch:
    def __init__(self, root: Path, *, retention_seconds: float = 24 * 3600, max_bytes: int = 5 * 1024 ** 3) -> None:
        self.root = Path(root)
        self.retention_seconds = retention_seconds
        self.max_bytes = max_bytes

    def _dir(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        return path

    def attempt_dir(self, job_id: str, attempt_number: int) -> Path:
        return self._dir("jobs", f"{job_id}-{attempt_number}-{uuid.uuid4().hex[:8]}")

    def ingest_dir(self) -> Path:
        return self._dir("ingest", uuid.uuid4().hex)

    @staticmethod
    def remove(path: Optional[Path]) -> None:
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)

    def _entries(self) -> List[Tuple[float, int, Path]]:
        out = []
        for group in ("jobs", "ingest"):
            base = self.root / group
            if not base.is_dir():
                continue
            for entry in base.iterdir():
                try:
                    stat = entry.stat()
                    size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file()) if entry.is_dir() else stat.st_size
                except OSError:
                    continue
                out.append((stat.st_mtime, size, entry))
        return sorted(out)

    def sweep(self, *, keep: Optional[set] = None, now: Optional[float] = None) -> int:
        """Remove expired entries, then the oldest until under the byte cap. ``keep`` protects
        directories in use. Returns how many entries were removed."""
        now = time.time() if now is None else now
        keep = {Path(p) for p in (keep or set())}
        removed = 0
        entries = self._entries()
        total = sum(size for _, size, _ in entries)
        for mtime, size, path in entries:
            if path in keep:
                continue
            if now - mtime > self.retention_seconds or total > self.max_bytes:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                total -= size
                removed += 1
        return removed

    def usage_bytes(self) -> int:
        return sum(size for _, size, _ in self._entries())


class Spool:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, job_id: str, attempt_number: int, operation: str) -> Path:
        return self.root / f"{job_id}.{attempt_number}.{operation}.json"

    def write(self, job_id: str, attempt_number: int, operation: str, body: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(job_id, attempt_number, operation)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"job_id": job_id, "attempt_number": attempt_number, "operation": operation, "body": body, "meta": meta or {}}, handle)
        os.replace(tmp, path)
        return path

    def remove(self, job_id: str, attempt_number: int, operation: str) -> None:
        self._path(job_id, attempt_number, operation).unlink(missing_ok=True)

    def has(self, job_id: str, attempt_number: int, operation: str) -> bool:
        return self._path(job_id, attempt_number, operation).is_file()

    def pending(self) -> Iterator[Dict[str, Any]]:
        if not self.root.is_dir():
            return
        for path in sorted(self.root.glob("*.json")):
            try:
                yield json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                path.unlink(missing_ok=True)

    def count(self) -> int:
        return len(list(self.root.glob("*.json"))) if self.root.is_dir() else 0

    # --- deferred publications -----------------------------------------------------------------

    def outputs_dir(self, job_id: str, attempt_number: int, *, create: bool = True) -> Path:
        base = self.root / "outputs"
        path = base / f"{job_id}.{attempt_number}"
        if create:
            path.mkdir(parents=True, exist_ok=True)
            for folder in (self.root, base, path):
                try:
                    os.chmod(folder, 0o700)
                except OSError:
                    pass
        return path

    def discard_publication(self, job_id: str, attempt_number: int) -> None:
        """Remove a deferred publication: its entry, then its output files."""
        self.remove(job_id, attempt_number, "publish")
        shutil.rmtree(self.outputs_dir(job_id, attempt_number, create=False), ignore_errors=True)

    def sweep_outputs(self, *, min_age_seconds: float = 3600.0, now: Optional[float] = None) -> int:
        """Remove output folders whose publish entry is gone (a crash between the two writes).
        Young folders are kept: a worker may be writing one right now."""
        base = self.root / "outputs"
        if not base.is_dir():
            return 0
        now = time.time() if now is None else now
        removed = 0
        for folder in base.iterdir():
            job_id, _, attempt = folder.name.rpartition(".")
            if not job_id or not attempt.isdigit() or self.has(job_id, int(attempt), "publish"):
                continue
            try:
                if now - folder.stat().st_mtime < min_age_seconds:
                    continue
            except OSError:
                continue
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1
        return removed
