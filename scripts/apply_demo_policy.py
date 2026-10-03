"""Apply the demo's rubric policy, alert rule and queue rule to a RUNNING demo Store (no reset).

``python -m call1.launch --demo`` applies these itself on a fresh demo root (``call1.demo_setup``).
A demo started before that, or one whose ``demo-setup.json`` marker was removed, gets them here:

    .venv-local/bin/python scripts/apply_demo_policy.py                 # policy + alert + queue rule
    .venv-local/bin/python scripts/apply_demo_policy.py --reanalyze     # ... and rescore every call's QA
    .venv-local/bin/python scripts/apply_demo_policy.py --store http://localhost:8010

It signs in with demo mode's admin persona (``/demo/sign-in``, localhost only), publishes the next
``call1_standard_v2`` version with the retail verification and disclosure policy for SEC-01 and
COMP-01, and creates the ``stock-check`` alert rule (intent > Check stock / availability) and the
``signal-stock-check`` SIGNAL queue rule. Every step is idempotent. Existing scorecards keep the
version they were scored with until ``--reanalyze`` asks Process to rescore each call's QA with the
current version (one ``qa`` reanalysis request per call); new signals on those calls then reach the
alert, the queue and Metrics as they are projected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    import httpx

    from call1.demo_setup import DemoSetupError, apply_demo_setup

    default_store = f"http://localhost:{os.environ.get('CALL1_STORE_PORT') or 8010}"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=default_store, help=f"the demo Store's URL (default {default_store}); must be localhost")
    parser.add_argument("--reanalyze", action="store_true", help="request a QA reanalysis of every call so scorecards use the new policy")
    args = parser.parse_args(argv)
    host = httpx.URL(args.store).host
    if host not in ("localhost", "127.0.0.1", "::1"):
        parser.error("demo sign-in answers on localhost only")
    try:
        with httpx.Client(base_url=args.store, timeout=30) as client:
            summary = apply_demo_setup(client, reanalyze=args.reanalyze)
    except DemoSetupError as exc:
        print(f"apply_demo_policy: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"apply_demo_policy: cannot reach Store at {args.store} ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
