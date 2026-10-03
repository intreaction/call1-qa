"""Serve the two browser apps Store hosts: Evaluate at ``/`` and the Store console at ``/console/``.

The frontend build (``npm --prefix frontend run build``) writes them under
``call1/store/static/``. Either layout works:

* one directory per app: ``static/evaluate/index.html`` and ``static/console/index.html`` (each
  with its own ``assets/``), or
* one Vite multi-page output: ``static/evaluate.html``, ``static/store-console.html`` and a shared
  ``static/assets/``.

Unknown paths without a file extension fall back to the app's ``index.html`` (client-side routing).
Until an app is built, its pages show a plain "not built yet" placeholder. Paths under ``/store/``
are never served from here: an unknown API path is a 404 envelope, not the Evaluate page.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional, Sequence

from fastapi import FastAPI, Request
from starlette.responses import FileResponse, HTMLResponse, Response

from call1.contracts.errors import ErrorCode

from .errors import StoreError

STATIC_ROOT = Path(__file__).resolve().parent / "static"


def _within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _find(roots: Sequence[Path], relative: str) -> Optional[Path]:
    if not relative or relative.endswith("/"):
        return None
    for root in roots:
        candidate = root / relative
        if _within(root, candidate) and candidate.is_file():
            return candidate
    return None


def _placeholder(app_name: str, detail: str) -> HTMLResponse:
    name = html.escape(app_name)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name}</title>
<style>
:root {{ color-scheme: light dark; --bg: #f6f8fb; --fg: #111827; --muted: #4b5563; --line: #d7dde6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #0b0f17; --fg: #e6e9ef; --muted: #9aa3b2; --line: #243044; }} }}
body {{ margin: 0; background: var(--bg); color: var(--fg); font-family: "IBM Plex Sans", system-ui, sans-serif; }}
header {{ padding: 16px 24px; border-bottom: 1px solid var(--line); font-weight: 600; }}
main {{ padding: 32px 24px; max-width: 640px; line-height: 1.5; }}
p {{ color: var(--muted); }}
</style></head>
<body><header>{name}</header>
<main><h1>Not built yet</h1><p>{html.escape(detail)}</p>
<p>Build the browser apps with <strong>npm --prefix frontend run build</strong>, then reload this page.
The Store API is running: <a href="/store/v1/status">/store/v1/status</a>.</p></main></body></html>"""
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


class SpaSite:
    def __init__(self, name: str, roots: Sequence[Path], index_candidates: Sequence[Path], detail: str) -> None:
        self.name = name
        self.roots = list(roots)
        self.index_candidates = list(index_candidates)
        self.detail = detail

    def index(self) -> Optional[Path]:
        return next((p for p in self.index_candidates if p.is_file()), None)

    def serve(self, relative: str) -> Response:
        found = _find(self.roots, relative)
        if found is not None:
            return FileResponse(found)
        if "." in relative.rsplit("/", 1)[-1]:
            raise StoreError(ErrorCode.NOT_FOUND, "No such file")
        index = self.index()
        if index is None:
            return _placeholder(self.name, self.detail)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def evaluate_site(root: Path = STATIC_ROOT) -> SpaSite:
    return SpaSite("Call1 Evaluate", [root / "evaluate", root], [root / "evaluate" / "index.html", root / "evaluate.html"],
                   "Evaluate is the reviewer app: passkey sign-in, Workbench, Rubric Studio, review queue and metrics.")


def console_site(root: Path = STATIC_ROOT) -> SpaSite:
    return SpaSite("Call1 Store", [root / "console"], [root / "console" / "index.html", root / "store-console.html"],
                   "The Store console shows Store's health and operations.")


def mount_frontends(app: FastAPI, root: Path = STATIC_ROOT) -> None:
    evaluate = evaluate_site(root)
    console = console_site(root)

    @app.get("/console", include_in_schema=False)
    @app.get("/console/", include_in_schema=False)
    def console_index() -> Response:
        return console.serve("")

    @app.get("/console/{path:path}", include_in_schema=False)
    def console_page(path: str) -> Response:
        return console.serve(path)

    @app.get("/", include_in_schema=False)
    def evaluate_index() -> Response:
        return evaluate.serve("")

    @app.get("/{path:path}", include_in_schema=False)
    def evaluate_page(path: str, request: Request) -> Response:
        if path == "store" or path.startswith("store/"):
            raise StoreError(ErrorCode.NOT_FOUND, "No such route")
        return evaluate.serve(path)
