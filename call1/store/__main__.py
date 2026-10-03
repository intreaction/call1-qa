"""``python -m call1.store <command>``: run Store and its host-only commands.

Commands
    serve              run the Store app (uvicorn) on CALL1_STORE_BIND:CALL1_STORE_PORT
    migrate            apply pending database migrations and exit
    setup-code         print a single-use first-admin or break-glass enrollment code
    issue-service-key  register a Process installation, issue its key and write Process's config
    request-reembedding
                       ask Process to re-embed every call whose search vectors are of another
                       scheme than Store's search embedder (contract 1.2.0)
    apply-signals-seed <path> [--pipeline v1|shadow|v2]
                       apply a seed signal taxonomy (a SignalTaxonomySave JSON, e.g.
                       call1/store/seeds/signals_retail_v1.json) as an audited admin save; used by
                       --demo. Installs start with the built-ins only (contract 1.3.0)
    apply-vocabulary-seed <path>
                       install an ASR vocabulary pack (an AsrVocabularyPack JSON, e.g.
                       call1/store/seeds/asr_vocabulary_retail_v1.json) as the industry pack for
                       dual transcription; audited, idempotent; used by --demo (contract 1.3.0,
                       decision 33)
    signals-pipeline v1|shadow|v2
                       set the Contact Signals pipeline setting (audited, installer actor)
    apply-signals-recipes <path> [--detection model|rules]
                       install a seed's rules-engine recipes (and its example-bank pin) into the
                       current signal taxonomy as an audited admin save, keeping every other edit
                       (contract 1.4.0, docs/SignalsEmbeddings.md)
    signals-detection model|rules
                       set Contact Signals detection: model (Gemma decides; the default) or rules
                       (categories with a rules recipe are decided by the rules engine)
    project-signals [--limit N]
                       project contact signals published before contract 1.3.0 (idempotent)

The host commands run as the Store OS user on the Store host; no HTTP route does what they do.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from call1.contracts.auth import SetupCodeIssueRequest, SetupCodePurpose
from call1.contracts.common import ServiceScope

from .config import ConfigError, StoreConfig

DEFAULT_PROCESS_CONFIG = Path("data/process/config.json")

# What a Stage 2 Process worker calls (least privilege). admin-state:read stays in: Process reads
# getStatusDetail and getAdminState (501 until Stage 4, which Process treats as the contract
# defaults). training:read (1.3.0) lets Process page the reviewer-label log for on-device training
# (IDs, enums and versions only). jobs:control (retry/cancel), key-release:write and
# release-trust:write need --scope.
PROCESS_DEFAULT_SCOPES = (
    ServiceScope.CALLS_WRITE,
    ServiceScope.ARTIFACTS_READ,
    ServiceScope.ARTIFACTS_WRITE,
    ServiceScope.JOBS_WRITE,
    ServiceScope.JOBS_CLAIM,
    ServiceScope.REANALYSIS_CLAIM,
    ServiceScope.CHANGES_READ,
    ServiceScope.HARDWARE_WRITE,
    ServiceScope.CATALOG_PUBLISH,
    ServiceScope.USAGE_READ,
    ServiceScope.ADMIN_STATE_READ,
    ServiceScope.TRAINING_READ,
)


def _config(args: argparse.Namespace) -> StoreConfig:
    config = StoreConfig.from_env()
    overrides = {}
    if getattr(args, "port", None):
        overrides["port"] = args.port
    if getattr(args, "bind", None):
        overrides["bind_host"] = args.bind
    return config.with_overrides(**overrides) if overrides else config


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app

    config = _config(args)
    tls = {}
    if not config.dev_mode:
        if not (config.tls_cert_file and config.tls_key_file):
            print("Outside dev mode Store serves HTTPS only: set CALL1_STORE_TLS_CERT and CALL1_STORE_TLS_KEY "
                  "(certificate management is Stage 4).", file=sys.stderr)
            return 2
        tls = {"ssl_certfile": str(config.tls_cert_file), "ssl_keyfile": str(config.tls_key_file)}
    app = create_app(config)
    print(f"Call1 Store on {config.public_base_url} (data: {config.data_dir}{', dev mode' if config.dev_mode else ''})", file=sys.stderr)
    from .results.search import embedder_status

    embedder = embedder_status()
    if embedder is not None and embedder["state"] == "not_installed":
        print(f"Semantic search is unavailable: the search embedder ({embedder['model']}) is not installed. "
              "Run `python -m call1.embedding download` (or set CALL1_EMBEDDING_PATH) and restart Store.", file=sys.stderr)
    elif embedder is not None:
        print(f"Search embedder: {embedder['scheme']} ({embedder['state']})", file=sys.stderr)
    if config.demo_mode:
        print("DEMO MODE: /demo/sign-in signs in Demo Admin, Demo Supervisor or Demo Reviewer without a passkey "
              "(localhost only; every demo sign-in is audited). Do not use with real data.", file=sys.stderr)
    uvicorn.run(app, host=config.bind_host, port=config.port, log_level=args.log_level, **tls)
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    from .db import Database

    config = _config(args)
    applied = Database(config.db_path).initialize()
    print("applied migrations: " + (", ".join(f"{v:03d}" for v in applied) if applied else "none (up to date)"))
    return 0


def _open_store(args: argparse.Namespace):
    from .context import Store

    return Store.open(_config(args))


def cmd_seed_demo_history(args: argparse.Namespace) -> int:
    from .results.demo_history import seed_demo_history

    config = _config(args)
    if not config.demo_mode:
        print("seed-demo-history requires CALL1_STORE_DEMO=1; refusing to seed a normal Store", file=sys.stderr)
        return 2
    store = _open_store(args)
    try:
        added = seed_demo_history(store, count=args.count)
    except ValueError as exc:
        print(f"seed-demo-history: {exc}", file=sys.stderr)
        return 2
    print(f"Demo history: added {added} synthetic sessions (target {args.count}); no audio or model inference.")
    return 0


def cmd_setup_code(args: argparse.Namespace) -> int:
    from .auth import cli as auth_cli

    request = SetupCodeIssueRequest(
        purpose=SetupCodePurpose(args.purpose),
        email=args.email,
        display_name=args.display_name,
        target_account_id=args.target_account_id,
        os_user=getpass.getuser() or "unknown",
    )
    store = _open_store(args)
    try:
        code, record = auth_cli.issue_setup_code(store, request)
    except NotImplementedError as exc:
        print(f"setup-code: {exc}", file=sys.stderr)
        return 2
    print(f"Setup code for {record.email} ({record.purpose.value}), valid until {record.expires_at.isoformat()}:")
    print(code)
    print(f"Open {store.config.public_base_url}/ and enroll with this code. It works once.", file=sys.stderr)
    return 0


def write_process_config(path: Path, values: dict) -> Path:
    """Merge ``values`` into Process's JSON config and leave it mode 0600."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            raise SystemExit(f"{path} is not JSON; refusing to overwrite it")
    current.update(values)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(current, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def cmd_issue_service_key(args: argparse.Namespace) -> int:
    from .auth import cli as auth_cli

    scopes = [ServiceScope(s) for s in args.scope] if args.scope else list(PROCESS_DEFAULT_SCOPES)
    store = _open_store(args)
    try:
        issued = auth_cli.issue_service_key(store, installation_label=args.installation, scopes=scopes, primary_host=args.primary_host)
    except NotImplementedError as exc:
        print(f"issue-service-key: {exc}", file=sys.stderr)
        return 2
    if args.print_token:
        print(issued.token)
        return 0
    path = Path(args.config or os.environ.get("CALL1_PROCESS_CONFIG") or DEFAULT_PROCESS_CONFIG)
    write_process_config(path, {
        "store_url": store.config.public_base_url,
        "installation_id": issued.key.installation_id,
        "service_key_id": issued.key.id,
        "service_key": issued.token,
    })
    print(f"Issued {issued.key.key_prefix} to installation {issued.key.installation_id}; wrote {path} (mode 0600).", file=sys.stderr)
    return 0


def cmd_request_reembedding(args: argparse.Namespace) -> int:
    """One reanalysis request of kind ``embeddings`` per call indexed under another scheme (e.g.
    ``hashing-projection-v1`` from before contract 1.2.0). Idempotent per call and target scheme."""
    from call1.contracts.events import AuditAction
    from call1.contracts.jobs import ReanalysisKind

    from . import audit, db
    from .queue import api as queue_api
    from .results.search import search_embedder

    scheme = search_embedder().scheme
    actor = audit.installer_actor(getpass.getuser() or "unknown")
    store = _open_store(args)
    created = 0
    with store.connection() as conn:
        stale = [r[0] for r in conn.execute(
            "SELECT DISTINCT conversation_id FROM results_search_vectors WHERE scheme != ? ORDER BY conversation_id", (scheme,)).fetchall()]
        for conversation_id in stale:
            key = f"reembed.{scheme.replace('@', '.')}.{conversation_id}"[:128]
            with db.transaction(conn):
                request = queue_api.create_reanalysis_request(conn, conversation_id=conversation_id, kind=ReanalysisKind.EMBEDDINGS,
                                                              requested_by=actor, idempotency_key=key, reason=f"re-embed for {scheme}")
                audit.append(conn, actor=actor, action=AuditAction.REANALYSIS_REQUESTED, target_kind="call", target_id=request.call_id,
                             details={"kind": ReanalysisKind.EMBEDDINGS.value, "reanalysis_request_id": request.id})
            created += 1
    print(f"{created} call(s) indexed under another scheme than {scheme}: re-embedding requested; Process fulfils the requests.",
          file=sys.stderr)
    return 0


def _installer():
    from . import audit

    return audit.installer_actor(getpass.getuser() or "unknown")


def cmd_apply_signals_seed(args: argparse.Namespace) -> int:
    """Apply a seed taxonomy as an admin save (contract 1.3.0, decision 22: ``--demo`` only)."""
    from pydantic import ValidationError

    from call1.contracts.signals import SignalTaxonomySave

    from .errors import StoreError
    from .results import signal_store

    try:
        body = SignalTaxonomySave.model_validate(json.loads(Path(args.path).read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        print(f"apply-signals-seed: {args.path} is not a valid SignalTaxonomySave: {str(exc)[:500]}", file=sys.stderr)
        return 2
    store = _open_store(args)
    with store.connection() as conn:
        try:
            record, version = signal_store.apply_seed(conn, body, actor=_installer(), parameters=store.config.parameters, pipeline=args.pipeline)
        except StoreError as exc:
            print(f"apply-signals-seed: {exc.code.value}: {exc} {exc.details}", file=sys.stderr)
            return 2
    what = f"published as taxonomy v{version}" if version is not None else f"already current (taxonomy v{record.current.version})"
    print(f"signal taxonomy seed {what}; pipeline {record.settings.pipeline}", file=sys.stderr)
    return 0


def cmd_apply_vocabulary_seed(args: argparse.Namespace) -> int:
    """Install an ASR vocabulary pack (contract 1.3.0, decision 33: ``--demo`` applies the retail seed)."""
    from pydantic import ValidationError

    from call1.contracts.vocabulary import AsrVocabularyPack

    from .errors import StoreError
    from .results import vocabulary

    try:
        pack = AsrVocabularyPack.model_validate(json.loads(Path(args.path).read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        # Pydantic's message names the failing position and rule; a term itself is never printed.
        reason = exc.errors()[0].get("msg", "invalid") if isinstance(exc, ValidationError) else type(exc).__name__
        print(f"apply-vocabulary-seed: {args.path} is not a valid AsrVocabularyPack: {str(reason)[:300]}", file=sys.stderr)
        return 2
    store = _open_store(args)
    with store.connection() as conn:
        try:
            record, changed = vocabulary.install_pack(conn, pack, actor=_installer(), parameters=store.config.parameters)
        except StoreError as exc:
            print(f"apply-vocabulary-seed: {exc.code.value}: {exc} {exc.details}", file=sys.stderr)
            return 2
    what = "installed" if changed else "already installed"
    state = "on" if record.active else ("off" if not record.settings.enabled else "off (empty)")
    print(f"vocabulary pack {pack.pack_id} v{pack.version} {what} ({len(pack.terms)} terms; record v{record.record_version}; "
          f"{len(record.effective_terms)} effective terms; dual transcription {state})", file=sys.stderr)
    return 0


def cmd_signals_pipeline(args: argparse.Namespace) -> int:
    from . import db
    from .results import signal_store

    store = _open_store(args)
    with store.connection() as conn, db.transaction(conn):
        record = signal_store.set_pipeline(conn, args.pipeline, actor=_installer())
    print(f"signal pipeline is {record.settings.pipeline}", file=sys.stderr)
    return 0


def cmd_apply_signals_recipes(args: argparse.Namespace) -> int:
    """Install a seed's recipes into the current taxonomy (contract 1.4.0)."""
    from pydantic import ValidationError

    from call1.contracts.signals import SignalTaxonomySave

    from .errors import StoreError
    from .results import signal_store

    try:
        body = SignalTaxonomySave.model_validate(json.loads(Path(args.path).read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        print(f"apply-signals-recipes: {args.path} is not a valid SignalTaxonomySave: {str(exc)[:500]}", file=sys.stderr)
        return 2
    store = _open_store(args)
    with store.connection() as conn:
        try:
            record, version = signal_store.apply_recipes(conn, body.taxonomy, actor=_installer(), parameters=store.config.parameters,
                                                         detection=args.detection)
        except StoreError as exc:
            print(f"apply-signals-recipes: {exc.code.value}: {exc} {exc.details}", file=sys.stderr)
            return 2
    what = f"published as taxonomy v{version}" if version is not None else f"already current (taxonomy v{record.current.version})"
    ruled = sum(1 for c in record.current.taxonomy.categories if c.recipe is not None and c.recipe.engine == "rules")
    print(f"signal recipes {what}: {ruled} categories with a rules recipe; detection {record.settings.detection}", file=sys.stderr)
    return 0


def cmd_signals_detection(args: argparse.Namespace) -> int:
    from . import db
    from .results import signal_store

    store = _open_store(args)
    with store.connection() as conn, db.transaction(conn):
        record = signal_store.set_detection(conn, args.detection, actor=_installer())
    print(f"signal detection is {record.settings.detection}", file=sys.stderr)
    return 0


def cmd_project_signals(args: argparse.Namespace) -> int:
    """Project contact signals published before 1.3.0 (idempotent, bounded, audited)."""
    from call1.contracts.events import AuditAction

    from . import audit, db
    from .results import projections

    store = _open_store(args)
    with store.connection() as conn:
        count = projections.project_existing_signals(conn, limit=args.limit)
        with db.transaction(conn):
            audit.append(conn, actor=_installer(), action=AuditAction.SIGNAL_BACKFILL_REQUESTED, target_kind="signal_backfill",
                         target_id="project-signals", details={"backfill_id": "project-signals", "mode": "project", "calls_projected": count,
                                                               "limit": args.limit})
    print(f"{count} call(s) projected", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m call1.store", description="Call1 Store")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the Store app")
    serve.add_argument("--port", type=int, help="default CALL1_STORE_PORT or 8010")
    serve.add_argument("--bind", help="default CALL1_STORE_BIND or 127.0.0.1")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    migrate = sub.add_parser("migrate", help="apply pending database migrations")
    migrate.set_defaults(func=cmd_migrate)

    setup = sub.add_parser("setup-code", help="print a single-use first-admin or break-glass enrollment code")
    setup.add_argument("--purpose", choices=[p.value for p in SetupCodePurpose], default=SetupCodePurpose.FIRST_ADMIN.value)
    setup.add_argument("--email", required=True)
    setup.add_argument("--display-name", required=True)
    setup.add_argument("--target-account-id", help="break_glass only: re-enroll this account")
    setup.set_defaults(func=cmd_setup_code)

    key = sub.add_parser("issue-service-key", help="issue a Process service key and write Process's config")
    key.add_argument("--installation", required=True, help="installation label, e.g. the Process host's name")
    key.add_argument("--scope", action="append", choices=[s.value for s in ServiceScope], help="repeatable; default the Stage 2 Process worker scopes (" + ", ".join(s.value for s in PROCESS_DEFAULT_SCOPES) + ")")
    key.add_argument("--primary-host", dest="primary_host", action="store_true", default=None,
                     help="the designated primary Process host (runs ML stages); default: primary when no active installation is")
    key.add_argument("--no-primary-host", dest="primary_host", action="store_false")
    key.add_argument("--config", help="Process config file (default CALL1_PROCESS_CONFIG or data/process/config.json)")
    key.add_argument("--print-token", action="store_true", help="print the token once instead of writing the config file")
    key.set_defaults(func=cmd_issue_service_key)

    reembed = sub.add_parser("request-reembedding", help="re-embed calls whose search vectors are of an older scheme")
    reembed.set_defaults(func=cmd_request_reembedding)

    history = sub.add_parser("seed-demo-history", help="seed fictional call history, demo mode only (idempotent)")
    history.add_argument("--count", type=int, default=560, help="target number of synthetic sessions (1–5000; default 560)")
    history.set_defaults(func=cmd_seed_demo_history)

    seed = sub.add_parser("apply-signals-seed", help="apply a seed signal taxonomy as an audited admin save (--demo)")
    seed.add_argument("path", help="a SignalTaxonomySave JSON file, e.g. call1/store/seeds/signals_retail_v1.json")
    seed.add_argument("--pipeline", choices=["v1", "shadow", "v2"], help="also set the signal pipeline")
    seed.set_defaults(func=cmd_apply_signals_seed)

    vocab = sub.add_parser("apply-vocabulary-seed", help="install an ASR vocabulary pack for dual transcription (--demo)")
    vocab.add_argument("path", help="an AsrVocabularyPack JSON file, e.g. call1/store/seeds/asr_vocabulary_retail_v1.json")
    vocab.set_defaults(func=cmd_apply_vocabulary_seed)

    pipeline = sub.add_parser("signals-pipeline", help="set the Contact Signals pipeline setting")
    pipeline.add_argument("pipeline", choices=["v1", "shadow", "v2"])
    pipeline.set_defaults(func=cmd_signals_pipeline)

    recipes = sub.add_parser("apply-signals-recipes", help="install a seed's rules-engine recipes into the current taxonomy (1.4.0)")
    recipes.add_argument("path", help="a SignalTaxonomySave JSON file whose categories carry recipes, e.g. call1/store/seeds/signals_retail_v1.json")
    recipes.add_argument("--detection", choices=["model", "rules"], help="also set Contact Signals detection")
    recipes.set_defaults(func=cmd_apply_signals_recipes)

    detection = sub.add_parser("signals-detection", help="set Contact Signals detection: model (default) or rules (1.4.0)")
    detection.add_argument("detection", choices=["model", "rules"])
    detection.set_defaults(func=cmd_signals_detection)

    project = sub.add_parser("project-signals", help="project contact signals published before contract 1.3.0")
    project.add_argument("--limit", type=int, default=1000)
    project.set_defaults(func=cmd_project_signals)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
