"""Shared inference components for Process; exports load lazily."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "RubricEvaluator": "call1.pipeline.evaluator",
    "verify_quoted_evidence": "call1.pipeline.evaluator",
}

__all__ = ["RubricEvaluator", "verify_quoted_evidence"]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
