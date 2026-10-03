"""``python -m call1.process <command>``

Commands
    serve          the operator API and console on 127.0.0.1:8020, plus the worker loop
    ingest FILE    register a recording with Store, upload it and create its job graph
                   (--agent-id, --agent-name, --agent-extension, --agent-channel, --external-ref;
                   the same FILE again with different values updates the call's metadata)
    drain          claim and run every ready job once, then exit (no HTTP)
    console-token  issue (or --rotate) the loopback console credential; prints it once
    status         check the Store connection and print the overview as JSON
    training status   on-device training: settings, next run, labels, active adapter, runs (JSON)
    training run      run on-device training now in this process; refused while a serve owns
                      training on this host (use Train now in its console, whose claim pause
                      covers the run)

Configuration: ``CALL1_PROCESS_CONFIG`` (default ``data/process/config.json``), written by
``python -m call1.store issue-service-key --installation <name>`` on the Store host.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import ConfigError, ProcessConfig


def _load(args: argparse.Namespace) -> ProcessConfig:
    config = ProcessConfig.load(Path(args.config) if getattr(args, "config", None) else None)
    overrides = {}
    if getattr(args, "port", None):
        overrides["port"] = args.port
    if getattr(args, "handlers", None):
        overrides["handlers"] = args.handlers
    return config.with_overrides(**overrides) if overrides else config


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from . import console
    from .app import create_app
    from .runtime import ProcessRuntime

    config = _load(args)
    if not config.configured:
        print(f"Process is not configured yet ({config.config_path} has no Store URL or service key). The console will say so.\n"
              "On the Store host run: python -m call1.store issue-service-key --installation <name>", file=sys.stderr)
    try:
        token, _ = console.issue(config)
        print(f"Console credential issued (shown once; Process keeps only its hash). Open: {console.console_url(config, token)}",
              file=sys.stderr)
    except FileExistsError:
        pass
    runtime = ProcessRuntime(config)
    app = create_app(runtime, start_background=not args.no_worker)
    print(f"Call1 Process on {console.console_url(config)} (handlers: {config.handlers}, Store: {config.store_url or 'not configured'})",
          file=sys.stderr)
    uvicorn.run(app, host=config.bind_host, port=config.port, log_level=args.log_level)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from .runtime import ProcessRuntime

    config = _load(args)
    runtime = ProcessRuntime(config)
    runtime.check_store()
    assert runtime.ingestor is not None
    from .ingest import InvalidCallMetadata

    try:
        result = runtime.ingestor.ingest_file(Path(args.file), agent_id=args.agent_id, agent_display_name=args.agent_name,
                                              agent_extension=args.agent_extension, agent_channel=args.agent_channel,
                                              external_call_ref=args.external_ref)
    except InvalidCallMetadata as exc:
        print(f"Invalid call metadata: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2))
    if result.metadata_updated:
        print(f"Metadata updated for the existing call ({', '.join(result.updated_fields)}); nothing was reprocessed.", file=sys.stderr)
    if args.wait:
        assert runtime.client is not None
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            progress = runtime.client.get_progress(result.conversation_id)
            if progress.settled:
                print(json.dumps({"settled": True, "groups": {g.kind.value: g.state.value for g in progress.groups}}, indent=2))
                return 0
            time.sleep(2)
        print("Not settled yet; is python -m call1.process serve running?", file=sys.stderr)
        return 3
    return 0


def cmd_drain(args: argparse.Namespace) -> int:
    from .runtime import ProcessRuntime

    runtime = ProcessRuntime(_load(args))
    worker = runtime.connect()
    done = worker.drain()
    print(json.dumps({"jobs_run": done, "stats": worker.stats.__dict__}, indent=2))
    return 0


def cmd_console_token(args: argparse.Namespace) -> int:
    from . import console

    config = _load(args)
    try:
        token, _ = console.issue(config, rotate=args.rotate)
    except FileExistsError as exc:
        print(f"{exc} (python -m call1.process console-token --rotate).", file=sys.stderr)
        return 2
    print(console.console_url(config, token))
    print("The console credential is shown once; Process keeps only its hash.", file=sys.stderr)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .runtime import ProcessRuntime
    from .store_client import ContractMismatch, StoreError

    runtime = ProcessRuntime(_load(args))
    code = 0
    try:
        runtime.check_store()
        runtime.state = "store_reachable"
    except (StoreError, ContractMismatch, ConfigError) as exc:
        runtime.state = ("contract_mismatch" if isinstance(exc, ContractMismatch) else "not_configured" if isinstance(exc, ConfigError)
                         else "store_unreachable" if exc.code == "store_unavailable" else "store_refused")
        runtime.store_error = {"message": str(exc)}
        code = 1
    print(json.dumps(runtime.overview(), indent=2, default=str))
    return code


def cmd_training(args: argparse.Namespace) -> int:
    from .runtime import ProcessRuntime
    from .training.scheduler import OWNED_ELSEWHERE, TrainingBusy, TrainingUnavailable

    runtime = ProcessRuntime(_load(args))
    if args.training_command == "status":
        print(json.dumps(runtime.training.describe(), indent=2, default=str))
        return 0
    if not runtime.training.claim_ownership():  # a running serve owns training: its claim pause must cover the run
        print(f"training: {OWNED_ELSEWHERE}", file=sys.stderr)
        return 2
    worker = runtime.connect()
    try:
        run = runtime.training.request_run("manual")
    except (TrainingBusy, TrainingUnavailable) as exc:
        print(f"training: {exc}", file=sys.stderr)
        return 2
    print(f"training run {run['run_id']} queued", file=sys.stderr)
    record = runtime.training.wait(timeout=args.timeout) or {}
    print(json.dumps(record, indent=2, default=str))
    del worker
    runtime.training.release_ownership()
    return 0 if record.get("status") in ("promoted", "rejected", "skipped") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m call1.process", description="Call1 Process")
    parser.add_argument("--config", help="config file (default CALL1_PROCESS_CONFIG or data/process/config.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the operator API, console and worker")
    serve.add_argument("--port", type=int, help="default 8020")
    serve.add_argument("--handlers", choices=["fake", "real"], help="override CALL1_PROCESS_HANDLERS")
    serve.add_argument("--no-worker", action="store_true", help="serve the console without connecting or running jobs")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    ingest = sub.add_parser("ingest", help="ingest one recording")
    ingest.add_argument("file")
    ingest.add_argument("--agent-id", help="the stable agent identifier (filters, metrics and queue rules key on it)")
    ingest.add_argument("--agent-name", help="the agent's display name; Evaluate shows 'Name (ext)'")
    ingest.add_argument("--agent-extension", help="the agent's PBX or phone extension (1-20 of 0-9 A-Z a-z * # + . _ -)")
    ingest.add_argument("--agent-channel", type=int, choices=[0, 1])
    ingest.add_argument("--external-ref", help="the recorder's call identifier")
    ingest.add_argument("--wait", type=float, default=0, help="seconds to wait for the conversation to settle (needs a running worker)")
    ingest.set_defaults(func=cmd_ingest)

    drain = sub.add_parser("drain", help="run every ready job once, then exit")
    drain.add_argument("--handlers", choices=["fake", "real"])
    drain.set_defaults(func=cmd_drain)

    token = sub.add_parser("console-token", help="issue the loopback console credential")
    token.add_argument("--rotate", action="store_true")
    token.set_defaults(func=cmd_console_token)

    status = sub.add_parser("status", help="check the Store connection")
    status.set_defaults(func=cmd_status)

    training = sub.add_parser("training", help="on-device training of the customer adapter")
    training_sub = training.add_subparsers(dest="training_command", required=True)
    training_sub.add_parser("status", help="print the training view as JSON")
    run = training_sub.add_parser("run", help="train now, in this process, and print the run record")
    run.add_argument("--timeout", type=float, default=6 * 3600, help="seconds to wait for the run (default 6 hours)")
    training.set_defaults(func=cmd_training)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Store request (claim polls) is noise
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # a clear one-line error for the operator; details are in the log
        from .store_client import ContractMismatch, StoreError

        if isinstance(exc, (StoreError, ContractMismatch)):
            print(f"store: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
