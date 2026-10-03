"""The split's architecture rules (docs/SplitBuild.md), enforced:

* Process never imports ``call1.store``, ``call1.db`` or ``call1.ingest``, directly or transitively,
  and never opens SQLite (checked in a fresh interpreter that imports every Process module, and by
  a static scan of the Process sources, real handlers included).
* The one-computer launcher imports neither app.
* Evaluate's source never calls the legacy ``/api/v1`` or Process.
* Evaluate talks only to ``/store/v1``, with exactly one documented exception: ``/demo/*``
  (``api/demo.ts``, localhost-only demo mode). ``fetch()`` may appear only in ``api/client.ts``
  (the typed Store client, which itself refuses any path outside ``/store/v1``) and ``api/demo.ts``,
  and every ``fetch()`` in ``api/demo.ts`` must target ``/demo/...``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROCESS = REPO / "call1" / "process"
EVALUATE = REPO / "frontend" / "src" / "apps" / "evaluate"
FORBIDDEN = ("call1.store", "call1.db", "call1.ingest")


def _process_modules():
    modules = []
    for path in sorted(PROCESS.rglob("*.py")):
        if " 2." in path.name or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(parts))
    return modules


def test_process_never_loads_store_db_or_legacy_ingest_in_a_fresh_interpreter():
    modules = _process_modules() + ["call1.launch"]
    script = (
        "import importlib, json, sys\n"
        f"mods = {modules!r}\n"
        "failed = {}\n"
        "for name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        failed[name] = type(exc).__name__ + ': ' + str(exc)[:200]\n"
        f"bad = sorted(m for m in sys.modules if m.split('.')[:2] in ({[f.split('.') for f in FORBIDDEN]!r}))\n"
        "print(json.dumps({'loaded': bad, 'failed': failed, 'sqlite3': 'sqlite3' in sys.modules}))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report["failed"] == {}, report["failed"]
    assert report["loaded"] == [], f"Process pulled in {report['loaded']}"
    assert report["sqlite3"] is False, "importing Process loaded sqlite3 (Process never opens SQLite)"


_IMPORT = re.compile(r"^\s*(?:from|import)\s+(call1\.(?:store|db|ingest))\b", re.MULTILINE)
_DYNAMIC = re.compile(r"""import_module\(\s*['"](call1\.(?:store|db|ingest))""")
_SQLITE = re.compile(r"^\s*(?:from|import)\s+sqlite3\b", re.MULTILINE)
_RELATIVE_UP = re.compile(r"^\s*from\s+\.\.\.+", re.MULTILINE)


def test_process_sources_name_no_forbidden_module():
    offenders = []
    for path in sorted(PROCESS.rglob("*.py")):
        if " 2." in path.name:
            continue
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(REPO))
        offenders += [f"{rel}: imports {m}" for m in _IMPORT.findall(text)]
        offenders += [f"{rel}: dynamically imports {m}" for m in _DYNAMIC.findall(text)]
        if _SQLITE.search(text):
            offenders.append(f"{rel}: opens SQLite (Store's database is Store's alone)")
        if _RELATIVE_UP.search(text):
            offenders.append(f"{rel}: relative import out of call1.process")
    assert offenders == []


def test_the_launcher_imports_neither_app():
    text = (REPO / "call1" / "launch.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:from|import)\s+call1\.(?:store|process|db|ingest)\b", text, re.MULTILINE)


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"(^|\s)//[^\n]*", r"\1", source)


def test_evaluate_never_calls_the_legacy_api_or_process():
    offenders = []
    for path in sorted(EVALUATE.rglob("*")):
        if path.suffix not in (".ts", ".tsx", ".js", ".jsx", ".mjs") or " 2." in path.name:
            continue
        code = _strip_comments(path.read_text(encoding="utf-8"))
        for needle in ("/api/v1", "/process/api", ":8020"):
            if needle in code:
                offenders.append(f"{path.relative_to(REPO)}: {needle}")
    assert offenders == []


def test_evaluate_talks_only_to_store_v1_and_the_documented_demo_exception():
    """Rule 3 (docs/SplitBuild.md): the typed Store client is the only way to reach Store, with
    exactly one documented, narrowly-scoped exception for demo mode's `/demo/*` (not part of the
    frozen contract, so it has no generated types: `call1/store/auth/demo.py`)."""
    demo_file = EVALUATE / "api" / "demo.ts"
    client_file = EVALUATE / "api" / "client.ts"
    bare_fetch = re.compile(r"(?<![A-Za-z0-9_.])fetch\(")  # not `refetch(`, `client.fetch(`, ...
    offenders = []
    for path in sorted(EVALUATE.rglob("*")):
        if path.suffix not in (".ts", ".tsx") or " 2." in path.name:
            continue
        code = _strip_comments(path.read_text(encoding="utf-8"))
        if bare_fetch.search(code) and path not in (client_file, demo_file):
            offenders.append(f"{path.relative_to(REPO)}: calls fetch() directly (must go through api/client.ts)")
    assert offenders == []
    assert demo_file.exists(), "expected frontend/src/apps/evaluate/api/demo.ts (the /demo/* exception)"
    demo_code = _strip_comments(demo_file.read_text(encoding="utf-8"))
    targets = re.findall(r"fetch\(\s*([A-Za-z0-9_.]+|`[^`]*`|'[^']*'|\"[^\"]*\")", demo_code)
    assert targets, "api/demo.ts should call fetch() against /demo/*"
    for target in targets:
        literal = target.strip("`'\"")
        # a bare identifier (e.g. a `const DEMO_..._PATH` reference) is trusted only if its name says so
        assert "/demo/" in literal or "DEMO" in literal.upper(), f"api/demo.ts fetch target is not /demo/*: {target}"
