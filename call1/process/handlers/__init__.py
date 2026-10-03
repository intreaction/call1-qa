"""The handler registry: which ``Handler`` runs each job type on this Process installation.

``build_registry(mode)`` always registers the **code handlers** (``handlers/code.py``: scorecard,
summary and contact-signal assembly), which are deterministic and identical in every mode. Then:

* ``mode == "fake"`` (``CALL1_PROCESS_HANDLERS=fake``) registers the fake handlers
  (``handlers/fake.py``) for every other job type. They produce contract-valid content without any
  model, for tests and for machines without Apple Silicon. ``embeddings`` uses the fake search
  embedder (``handlers/embeddings.py``, scheme ``fake-embedding-v1``) unless
  ``CALL1_EMBEDDING_BACKEND`` says otherwise; Store must resolve the same backend.
* ``mode == "real"`` imports ``call1.process.handlers.real`` when it exists and calls its
  ``register(registry, context)``; that package (built separately) wraps the pipeline modules. A
  job type with no handler is simply not offered to Store, so its jobs wait for a worker that has
  one; the console lists the missing types. Real mode never falls back to fake content.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from call1.contracts.catalog import CatalogEntryStatus, ModelPurpose
from call1.contracts.jobs import JobType

from .base import Handler, HandlerError, HandlerJob, HandlerResult, InputArtifact, JobCancelled, Output, ReleaseJob, Usage

log = logging.getLogger("call1.process.handlers")

REAL_HANDLERS_MODULE = "call1.process.handlers.real"


@dataclass(frozen=True)
class RegistryContext:
    """Handed to ``call1.process.handlers.real.register``: the Process config (models root, data
    dir, stages) and catalog (entries and their frozen revisions)."""

    config: Any
    catalog: Any


class HandlerRegistry:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._handlers: Dict[JobType, Handler] = {}
        self.entry_status: Optional[Callable[[Any], Tuple[CatalogEntryStatus, List[ModelPurpose]]]] = None
        """Optional hook a real-handler package sets: the status of a catalog entry on this host
        (installed, qualified). Defaults to the catalog's own check."""
        self.entry_detail: Optional[Callable[[Any], Optional[str]]] = None
        """Optional hook a real-handler package sets: why an entry is not installed on this host."""
        self.notes: List[str] = []

    def register(self, handler: Handler) -> None:
        job_type = JobType(handler.job_type)
        self._handlers[job_type] = handler

    def get(self, job_type: JobType) -> Optional[Handler]:
        return self._handlers.get(JobType(job_type))

    def job_types(self) -> FrozenSet[JobType]:
        return frozenset(self._handlers)

    def missing(self) -> List[JobType]:
        return [t for t in JobType if t not in self._handlers]

    def describe(self) -> List[Dict[str, Any]]:
        return [{"job_type": t.value, "adapter_id": h.adapter_id, "adapter_version": h.adapter_version}
                for t, h in sorted(self._handlers.items(), key=lambda kv: kv[0].value)]


def load_real_handlers(registry: HandlerRegistry, context: RegistryContext) -> bool:
    try:
        module = importlib.import_module(REAL_HANDLERS_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == REAL_HANDLERS_MODULE:
            registry.notes.append("Real handlers are not installed in this build (call1/process/handlers/real).")
            return False
        log.exception("real handlers failed to load")
        registry.notes.append(f"Real handlers failed to load ({type(exc).__name__}: missing {exc.name}).")
        return False
    except Exception as exc:  # a broken real-handler package must not stop the console
        log.exception("real handlers failed to load")
        registry.notes.append(f"Real handlers failed to load ({type(exc).__name__}).")
        return False
    register = getattr(module, "register", None)
    if register is None:
        registry.notes.append(f"{REAL_HANDLERS_MODULE} has no register(registry, context).")
        return False
    register(registry, context)
    return True


def build_registry(mode: str, *, config: Any = None, catalog: Any = None, fake_behavior: Any = None) -> HandlerRegistry:
    from .code import code_handlers

    registry = HandlerRegistry(mode)
    for handler in code_handlers():
        registry.register(handler)
    if mode == "fake":
        from .fake import FakeBehavior, fake_handlers

        from .embeddings import embeddings_handler

        for handler in fake_handlers(fake_behavior or FakeBehavior.from_env()):
            registry.register(handler)
        registry.register(embeddings_handler("fake"))
        registry.notes.append("Fake handlers: every result is synthetic test content, not an analysis of the recording.")
    elif mode == "real":
        load_real_handlers(registry, RegistryContext(config=config, catalog=catalog))
    else:
        raise ValueError(f"unknown handler mode {mode!r}")
    return registry


__all__ = [
    "Handler", "HandlerError", "HandlerJob", "HandlerResult", "HandlerRegistry", "InputArtifact", "JobCancelled", "Output",
    "RegistryContext", "ReleaseJob", "Usage", "build_registry", "load_real_handlers",
]
