"""Core routes: status, contract, the change feed and the audit log (contract operations owned by
the core), plus the non-contract byte-transfer endpoints behind upload and download grants."""

from call1.store.routing import Area, AreaRouter

router = AreaRouter(Area.CORE)

from . import audit, changes, status  # noqa: E402,F401  (registers the core handlers on ``router``)
