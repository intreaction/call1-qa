"""Architecture rule 1 (docs/SplitBuild.md), Store's side: Store loads no model runtime at import
and never imports Process or the legacy app. The one model Store runs (decision 18, contract 1.2.0)
is the search embedder in ``call1.embedding``, which imports torch and transformers only when the
first search loads it, so importing and starting Store still loads neither. Checked in a fresh interpreter so other tests' imports cannot
hide a regression. Process's side (never importing call1.store, call1.db or call1.ingest) belongs
in tests/test_split_boundaries.py with the Process package."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FORBIDDEN_PREFIXES = (
    "torch", "mlx", "transformers", "whisper", "faster_whisper", "pyannote", "speechbrain", "librosa",
    "call1.pipeline", "call1.db", "call1.ingest", "call1.process", "call1.adapters", "call1.ui",
)

_PROBE = """
import json, sys
import call1.store.app, call1.store.__main__, call1.store.maintenance
from call1.store.app import create_app
print(json.dumps(sorted(sys.modules)))
"""


def _loaded_modules() -> list:
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, capture_output=True, text=True, timeout=120, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_store_loads_no_model_runtime_process_or_legacy_modules():
    loaded = _loaded_modules()
    bad = sorted(m for m in loaded if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_PREFIXES))
    assert bad == [], f"call1.store pulled in {bad}"
    assert "call1.store.app" in loaded and "call1.contracts.api" in loaded


def test_store_source_never_names_process_or_legacy_packages():
    """A static backstop for imports the probe cannot reach (lazy imports inside functions)."""
    offenders = []
    for path in sorted((REPO / "call1" / "store").rglob("*.py")):
        if " 2." in path.name:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            words = stripped.replace(",", " ").split()
            module = words[1]
            if any(module == p or module.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(REPO)}:{number}: {stripped}")
    assert offenders == []


def test_the_search_embedder_is_the_only_model_path_and_is_lazy():
    """Store reaches a model only through ``call1.embedding`` (decision 18), and reading its status
    or selecting it never imports torch or transformers."""
    probe = (
        "import json, sys\n"
        "from call1.store.results import search\n"
        "search.embedder_status(); search.search_embedder()\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers'))))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True, timeout=120, check=True,
                         env={"PATH": "/usr/bin:/bin", "CALL1_EMBEDDING_BACKEND": "nemotron", "CALL1_EMBEDDING_PATH": "/nonexistent"})
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
    model_imports = []
    for path in sorted((REPO / "call1" / "store").rglob("*.py")):
        if " 2." in path.name:
            continue
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        if any(line.startswith(("import call1.embedding", "from call1.embedding", "from call1 import embedding")) for line in lines):
            model_imports.append(path.relative_to(REPO).as_posix())
    assert model_imports == ["call1/store/results/search.py"]
