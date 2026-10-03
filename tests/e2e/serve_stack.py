"""Start a Store + Process stack, print its facts as one JSON line, and keep it alive.

    .venv-local/bin/python tests/e2e/serve_stack.py [--handlers fake|real] [--real-models]
        [--fake-behavior JSON] [--store-parameters JSON] [--process-config JSON]
        [--store-env JSON] [--process-env JSON] [--seed-training-labels] [--no-process] [--signals-pipeline v1|shadow|v2] [--signals-seed PATH]
        [--vocabulary-seed PATH] [--name NAME] [--info-file PATH] [--keep]

The first line on stdout is ``{"ready": true, "store_url": ..., "process_url": ...,
"console_token": ..., ...}`` (``Stack.info()``). The stack stops, and its directory under
``/private/tmp/call1-e2e/`` is deleted, when stdin closes (the parent went away), on SIGTERM or
SIGINT. The Playwright global setup (``frontend/e2e/global-setup.ts``) runs it; it is also handy by
hand for poking at a throwaway stack. On a start-up failure it prints ``{"ready": false, "error":
...}`` and exits 1.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, so "e2e" is a package

from e2e.stack import Stack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="playwright")
    parser.add_argument("--handlers", choices=["fake", "real"], default="fake")
    parser.add_argument("--real-models", action="store_true", help="real handlers with CALL1_BACKEND=mlx (Apple Silicon)")
    parser.add_argument("--fake-behavior", help="CALL1_FAKE_BEHAVIOR JSON")
    parser.add_argument("--store-parameters", help="ContractParameters overrides, JSON")
    parser.add_argument("--process-config", help="extra Process config keys, JSON")
    parser.add_argument("--store-env", help="extra Store environment variables, JSON object of strings (e.g. demo mode's CALL1_STORE_DEMO)")
    parser.add_argument("--process-env", help="extra Process environment variables, JSON object of strings (e.g. on-device training's "
                        "CALL1_FAKE_TRAINING_OUTCOMES and CALL1_FAKE_TRAINER_SECONDS)")
    parser.add_argument("--seed-training-labels", action="store_true",
                        help="before reporting ready, ingest calls and override their QA verdicts as a reviewer until both on-device "
                             "training splits (train and held out) have a labelled call (training_seed.py)")
    parser.add_argument("--no-process", action="store_true", help="Store only — no Process server (demo mode's sign-in screens need no queue data)")
    parser.add_argument("--signals-pipeline", choices=["v1", "shadow", "v2"],
                        help="the Contact Signals pipeline setting (default CALL1_SIGNALS_PIPELINE, else v1)")
    parser.add_argument("--signals-seed", help="a SignalTaxonomySave JSON to publish first (the --demo retail seed)")
    parser.add_argument("--vocabulary-seed", help="an ASR vocabulary seed JSON to install as the industry pack first (the --demo retail seed)")
    parser.add_argument("--info-file", help="also write the JSON facts to this file")
    parser.add_argument("--keep", action="store_true", help="keep the stack directory on exit")
    args = parser.parse_args()

    stack = Stack(name=args.name, handlers=args.handlers, real_models=args.real_models,
                  fake_behavior=json.loads(args.fake_behavior) if args.fake_behavior else None,
                  store_parameters=json.loads(args.store_parameters) if args.store_parameters else None,
                  process_config=json.loads(args.process_config) if args.process_config else None,
                  store_env=json.loads(args.store_env) if args.store_env else None,
                  process_env=json.loads(args.process_env) if args.process_env else None,
                  with_process=not args.no_process, signals_pipeline=args.signals_pipeline,
                  signals_seed=args.signals_seed, vocabulary_seed=args.vocabulary_seed,
                  keep=True if args.keep else None)
    stop = threading.Event()

    def on_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    seeded = None
    try:
        stack.start()
        if args.seed_training_labels:
            from e2e.training_seed import seed_training_labels

            seeded = seed_training_labels(stack)
    except Exception as exc:  # report and exit: the parent reads this line
        print(json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        stack.close()
        return 1

    info = dict(stack.info(), ready=True, pid=os.getpid())
    if seeded is not None:
        info["training_labels"] = seeded
    if args.info_file:
        path = Path(args.info_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)
    print(json.dumps(info), flush=True)
    print(f"stack up: Evaluate {stack.evaluate_url}  Process {stack.process_url}  (dir {stack.dir})", file=sys.stderr, flush=True)

    def watch_stdin():
        try:
            while sys.stdin.read(4096):
                pass
        except (OSError, ValueError):
            pass
        stop.set()

    if not sys.stdin.isatty():
        threading.Thread(target=watch_stdin, daemon=True).start()
    try:
        while not stop.wait(0.5):
            pass
    finally:
        stack.close()
        print("stack stopped", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
