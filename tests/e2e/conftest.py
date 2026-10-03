"""Fixtures for end-to-end tests: real Store and Process server processes (see README.md).

    stack               the stack for this test: the shared fake-handler stack, or the real-model
                        stack when the test is marked ``real_models`` (and CALL1_REAL_MODELS=1)
    e2e_stack           the shared fake-handler stack (session scope)
    real_stack          the shared real-model stack (session scope; skips without CALL1_REAL_MODELS=1)
    stack_factory       stack_factory(**Stack kwargs) -> a private, started Stack closed after the test
    admin_session       the stack's first admin (setup code via CLI + real passkey ceremony)
    reviewer_session    a reviewer invited by the admin and enrolled by passkey (one per stack)
    supervisor_session  the same for a supervisor
    new_user            new_user(role="reviewer", email=None) -> a NEW enrolled, signed-in account
    ingest              ingest(sample_or_path, unique=True, agent_id=..., ...) -> Process's receipt
    wait_until_settled  wait_until_settled(call_id, timeout=None) -> Store's JobGroupProgress
    store_get/store_post  stack.store_get / stack.store_post (session=StoreSession | "service" | None)
    process_get/process_post  Process's loopback API with the console token

Every test under tests/e2e is marked ``e2e`` automatically; ``-m "not e2e"`` skips them.
"""

from __future__ import annotations

import os

import pytest

from .stack import Stack, StackError, StoreSession, sample_path, unique_email  # noqa: F401  (re-exported for tests)

_HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end test against real Store and Process server processes (tests/e2e)")
    config.addinivalue_line("markers", "real_models: runs the real model stack (opt-in with CALL1_REAL_MODELS=1)")


def pytest_collection_modifyitems(config, items):
    real = os.environ.get("CALL1_REAL_MODELS") == "1"
    skip_real = pytest.mark.skip(reason="real models are opt-in: set CALL1_REAL_MODELS=1 (Apple Silicon, weights in data/models)")
    for item in items:
        if not str(item.fspath).startswith(_HERE + os.sep):
            continue
        item.add_marker(pytest.mark.e2e)
        if item.get_closest_marker("real_models") and not real:
            item.add_marker(skip_real)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """On failure, attach the Store and Process log tails of every stack the test used."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    seen = set()
    for value in getattr(item, "funcargs", {}).values():
        stacks = [value] if isinstance(value, Stack) else getattr(value, "_e2e_stacks", [])
        for stack in stacks:
            if id(stack) in seen or not isinstance(stack, Stack):
                continue
            seen.add(id(stack))
            for name in ("store", "process"):
                report.sections.append((f"{stack.name} {name}.log (tail)", stack.log_tail(name, 60)))


@pytest.fixture(scope="session")
def e2e_stack():
    stack = Stack(name="session")
    stack.start()
    try:
        yield stack
    finally:
        stack.close()


@pytest.fixture(scope="session")
def real_stack():
    if os.environ.get("CALL1_REAL_MODELS") != "1":
        pytest.skip("real models are opt-in: set CALL1_REAL_MODELS=1")
    stack = Stack(name="real", real_models=True)
    stack.start()
    try:
        yield stack
    finally:
        stack.close()


@pytest.fixture
def stack(request) -> Stack:
    chosen = request.getfixturevalue("real_stack" if request.node.get_closest_marker("real_models") else "e2e_stack")
    chosen.ensure_running()
    return chosen


class _StackFactory:
    def __init__(self) -> None:
        self._e2e_stacks = []

    def __call__(self, **kwargs) -> Stack:
        kwargs.setdefault("name", "private")
        stack = Stack(**kwargs)
        self._e2e_stacks.append(stack)
        return stack.start()

    def close(self) -> None:
        for stack in self._e2e_stacks:
            stack.close()


@pytest.fixture
def stack_factory():
    """``stack_factory(handlers="fake", fake_behavior={...}, store_parameters={...}, process_config={...},
    with_process=True, ...)``: a private stack for tests that need isolation (an empty Store, a
    first-admin enrollment, a Store outage, scripted fake failures). Closed after the test."""
    factory = _StackFactory()
    try:
        yield factory
    finally:
        factory.close()


@pytest.fixture
def admin_session(stack) -> StoreSession:
    return stack.admin()


@pytest.fixture
def reviewer_session(stack) -> StoreSession:
    return stack.user("reviewer")


@pytest.fixture
def supervisor_session(stack) -> StoreSession:
    return stack.user("supervisor")


@pytest.fixture
def new_user(stack):
    def make(role: str = "reviewer", *, email=None, display_name=None) -> StoreSession:
        return stack.user(role, email=email, display_name=display_name, cached=False)
    return make


@pytest.fixture
def ingest(stack):
    return stack.ingest


@pytest.fixture
def wait_until_settled(stack):
    return stack.wait_until_settled


@pytest.fixture
def store_get(stack):
    return stack.store_get


@pytest.fixture
def store_post(stack):
    return stack.store_post


@pytest.fixture
def process_get(stack):
    return stack.process_get


@pytest.fixture
def process_post(stack):
    return stack.process_post
