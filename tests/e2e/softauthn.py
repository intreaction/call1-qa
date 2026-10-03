"""The Store tests' software WebAuthn authenticator, reused as is.

``tests/store/auth_softauthn.py`` builds P-256 credentials with ``none`` attestation and signs
assertions exactly as ``@simplewebauthn/browser`` would send them. It depends only on ``cbor2`` and
``cryptography``, so it is loaded here by path: the e2e harness then works both under pytest and
from ``serve_stack.py`` (a plain script), without depending on how pytest names the ``tests/store``
package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "store" / "auth_softauthn.py"
_NAME = "call1_e2e_auth_softauthn"


def _load():
    if _NAME in sys.modules:
        return sys.modules[_NAME]
    spec = importlib.util.spec_from_file_location(_NAME, _SOURCE)
    if spec is None or spec.loader is None:  # pragma: no cover - the file ships with the repo
        raise ImportError(f"cannot load the software authenticator from {_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
    return module


_module = _load()

SoftAuthenticator = _module.SoftAuthenticator
SoftCredential = _module.SoftCredential
b64url = _module.b64url
b64url_decode = _module.b64url_decode

__all__ = ["SoftAuthenticator", "SoftCredential", "b64url", "b64url_decode"]
