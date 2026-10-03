"""Call1 Store: the only owner of call, queue and review data (docs/SplitBuild.md).

Serves the frozen contract at ``/store/v1`` (``call1/contracts/``), Evaluate at ``/`` and the Store
console at ``/console/``. Run it with ``python -m call1.store serve``. Package map and ownership:
``call1/store/README.md``.

Process never imports this package (``tests/test_split_boundaries.py``); it reaches Store only
over HTTP with its service key.
"""

from __future__ import annotations


def create_app(*args, **kwargs):
    """``call1.store.app.create_app`` (imported lazily so area modules can import the package)."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
