"""A private Store+Process stack, remote-controlled over stdin, for tests that need to force a
real Store outage (e.g. the Process console's "Store unreachable" degraded state).

This is test infrastructure the harness (``tests/e2e/serve_stack.py``) does not provide: that
script starts one stack and only stops it on teardown. This one starts a private ``Stack`` (never
the shared Playwright one) and accepts commands on stdin so a Node test can flip Store on and off
around it, for exactly one spec file.

    .venv-local/bin/python tests/e2e/outage_stack.py --info-file PATH

Prints one JSON line (``Stack.info()`` plus ``ready: true``) on start, the same shape
``serve_stack.py`` prints. After that, each newline-delimited JSON command on stdin
(``{"cmd": "stop_store"}``, ``{"cmd": "start_store"}``, ``{"cmd": "stop_process"}``,
``{"cmd": "start_process"}``) gets one JSON response line (``{"ok": true, "cmd": ...}`` or
``{"ok": false, "cmd": ..., "error": ...}``). Closing stdin (or a fatal error) stops the stack and
deletes its directory, exactly like ``serve_stack.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, so "e2e" is a package

from e2e.stack import Stack  # noqa: E402

def _downgrade_scope(stack: Stack) -> None:
    """Re-issue Process's service key with the *default* scopes (no ``jobs:control``) and restart
    Process so it loads the weaker key — for testing the 403 ``insufficient_scope`` console path
    (call1/process/README.md "Without --scope, issue-service-key leaves out jobs:control")."""
    stack.store_cli("issue-service-key", "--installation", f"e2e-{stack.name}-downgraded", "--config", str(stack.process_config_path))
    stack.stop_process()
    stack.start_process()


COMMANDS = {
    "stop_store": lambda s: s.stop_store(),
    "start_store": lambda s: s.start_store(),
    "stop_process": lambda s: s.stop_process(),
    "start_process": lambda s: s.start_process(),
    "downgrade_scope": _downgrade_scope,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="outage")
    parser.add_argument("--info-file")
    args = parser.parse_args()

    stack = Stack(name=args.name, handlers="fake")
    try:
        stack.start()
    except Exception as exc:  # the parent reads this line and gives up
        print(json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 1

    info = dict(stack.info(), ready=True, pid=os.getpid())
    if args.info_file:
        path = Path(args.info_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)
    print(json.dumps(info), flush=True)

    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                cmd = json.loads(raw)
                action = cmd.get("cmd")
                fn = COMMANDS.get(action)
                if fn is None:
                    raise ValueError(f"unknown cmd {action!r}")
                fn(stack)
                print(json.dumps({"ok": True, "cmd": action}), flush=True)
            except Exception as exc:  # report and keep serving further commands
                print(json.dumps({"ok": False, "cmd": raw, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        stack.close()
        print(json.dumps({"stopped": True}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
