"""Opt-in runs on the real models (Apple Silicon, weights under data/models). Marked
``real_models`` and skipped unless ``CALL1_REAL_MODELS=1``:

    CALL1_REAL_MODELS=1 .venv-local/bin/python -m pytest tests/process/real -m real_models -s -p no:warnings
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line("markers", "real_models: runs the real model stack (opt-in with CALL1_REAL_MODELS=1)")
