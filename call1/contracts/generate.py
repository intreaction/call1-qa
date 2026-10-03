"""Regenerate the committed contract artifacts deterministically.

    python -m call1.contracts.generate            # write openapi.json and the TypeScript types
    python -m call1.contracts.generate --check    # exit 1 if either committed file is stale

Both refuse to run under generator versions other than ``api.GENERATOR_PINS`` (FastAPI and
Pydantic minor versions, the exact ``openapi-typescript``), because schema output differs between
versions. ``constraints.txt`` beside this file pins the Python side for pip
(``pip install -r requirements.txt -c call1/contracts/constraints.txt``). The TypeScript step runs
the pinned ``openapi-typescript`` from ``frontend/node_modules``; it is skipped with a notice when
that is not installed (``npm --prefix frontend ci``), unless ``--require-typescript`` is given or
``CALL1_REQUIRE_TS_CHECK=1`` is set, as CI does.

The byte-for-byte drift tests skip, rather than fail, under unpinned generators, because the
difference would be generator noise and not contract drift. ``CALL1_REQUIRE_GENERATOR_PINS=1`` (or
``CALL1_REQUIRE_TS_CHECK=1``) turns that skip into a failure; CI sets it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONSTRAINTS_PATH = Path(__file__).resolve().parent / "constraints.txt"
OPENAPI_PATH = Path(__file__).resolve().parent / "openapi.json"
TS_PATH = REPO / "frontend" / "src" / "contracts" / "store-v1.ts"
TS_BIN = REPO / "frontend" / "node_modules" / ".bin" / "openapi-typescript"
TS_PACKAGE = REPO / "frontend" / "node_modules" / "openapi-typescript" / "package.json"


def generator_version_problems() -> list[str]:
    """Differences between the installed generators and the pins; empty when they match."""
    import fastapi
    import pydantic

    from .api import GENERATOR_PINS

    problems = []
    for name, installed in (("fastapi", fastapi.__version__), ("pydantic", pydantic.VERSION)):
        pinned = GENERATOR_PINS[name]
        if not (installed == pinned or installed.startswith(pinned + ".")):
            problems.append(f"{name} {installed} is installed; the contract is generated with {pinned}.x")
    if TS_PACKAGE.exists():
        installed = json.loads(TS_PACKAGE.read_text())["version"]
        if installed != GENERATOR_PINS["openapi-typescript"]:
            problems.append(f"openapi-typescript {installed} is installed; the contract pins {GENERATOR_PINS['openapi-typescript']}")
    return problems


def render_openapi() -> str:
    from .api import build_openapi

    return json.dumps(build_openapi(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def render_typescript(openapi_path: Path = OPENAPI_PATH) -> str | None:
    """Return the generated TypeScript, or None when the generator is not installed."""
    if not TS_BIN.exists():
        return None
    result = subprocess.run(
        [str(TS_BIN), str(openapi_path), "--alphabetize", "--export-type"],
        check=True, capture_output=True, text=True, cwd=str(REPO / "frontend"),
    )
    return result.stdout


def _flag(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false")


def typescript_required() -> bool:
    return _flag("CALL1_REQUIRE_TS_CHECK")


def pins_required() -> bool:
    """True when generator versions other than the pins must fail the drift tests (CI)."""
    return _flag("CALL1_REQUIRE_GENERATOR_PINS") or typescript_required()


def constraint_pins() -> dict[str, str]:
    """The ``name==X.Y.*`` lines of constraints.txt, as ``{name: "X.Y"}``."""
    pins = {}
    for line in CONSTRAINTS_PATH.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, spec = line.partition("==")
            pins[name.strip()] = spec.strip().removesuffix(".*")
    return pins


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed files match")
    parser.add_argument("--no-typescript", action="store_true", help="skip the TypeScript step")
    parser.add_argument("--require-typescript", action="store_true", help="fail when openapi-typescript is not installed")
    args = parser.parse_args(argv)

    problems = generator_version_problems()
    if problems:
        print("wrong generator versions: " + "; ".join(problems), file=sys.stderr)
        return 1

    openapi = render_openapi()
    stale: list[str] = []
    if args.check:
        if not OPENAPI_PATH.exists() or OPENAPI_PATH.read_text() != openapi:
            stale.append(str(OPENAPI_PATH))
    else:
        OPENAPI_PATH.write_text(openapi)
        print(f"wrote {OPENAPI_PATH}")

    if not args.no_typescript:
        typescript = render_typescript()
        if typescript is None:
            if args.require_typescript or typescript_required():
                print("openapi-typescript is not installed and the TypeScript check is required (npm --prefix frontend ci)", file=sys.stderr)
                return 1
            print("openapi-typescript is not installed; skipped TypeScript (npm --prefix frontend ci)", file=sys.stderr)
        elif args.check:
            if not TS_PATH.exists() or TS_PATH.read_text() != typescript:
                stale.append(str(TS_PATH))
        else:
            TS_PATH.parent.mkdir(parents=True, exist_ok=True)
            TS_PATH.write_text(typescript)
            print(f"wrote {TS_PATH}")

    if stale:
        print("stale: " + ", ".join(stale) + " (run: python -m call1.contracts.generate)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
