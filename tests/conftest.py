"""Suite-wide defaults.

Semantic search (contract 1.2.0) embeds with ``call1.embedding``. Tests run without the model
weights, so every in-process Store and Process defaults to the deterministic fake embedder; tests
that need the real model (``real_models``) set ``CALL1_EMBEDDING_BACKEND=nemotron`` themselves.

The same holds for the PII masking model (team decision 19, ``call1.pii_model``): tests default to
the labelled deterministic stub; ``real_models`` tests select ``privacy-filter`` explicitly.
"""

from __future__ import annotations

import os

# The repo lives in iCloud, which leaves conflict copies such as "test_x 2.py" (gitignored). Never
# collect them: they are stale duplicates and inflate or break the suite.
collect_ignore_glob = ["* [0-9].py", "*/* [0-9].py"]

os.environ.setdefault("CALL1_EMBEDDING_BACKEND", "fake")
os.environ.setdefault("CALL1_PII_MODEL_BACKEND", "stub")


def pytest_sessionstart(session):
    from pathlib import Path
    import pytest
    if os.getenv("CALL1_REAL_MODELS") == "1" and (Path(__file__).resolve().parents[1] / "sample_audio/.test-audio.json").exists():
        raise pytest.UsageError("Generated test tones cannot qualify real models; supply cleared speech fixtures first.")
