"""Reviewer labels for on-device training e2e runs (docs/OnDeviceTraining.md section 7.3): recordings
through the fake-handler pipeline, then a signed-in reviewer overrides every QA verdict through
Store's review API, as Evaluate does. Test infrastructure only.

On-device training splits by call (``split_of``: 20% held out, 10% validation), so the seed keeps
ingesting until at least ``min_train_calls`` calls land in the training split and ``min_eval_calls``
in the held-out split; a run over these labels then reaches a decision instead of ``skipped``.

Every fake-handler call has the same transcript, so its training prompts are identical and the
dataset keeps only the newest label's example per prompt (``dataset.assemble``). The seed therefore
labels the held-out calls first, then the validation calls, and the training calls last, so the
kept examples belong to a training-split call.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

from .stack import REPO, Stack, StackError


def _split_of(installation_id: str, call_id: str) -> str:
    if str(REPO) not in sys.path:  # serve_stack.py runs with tests/ on the path, not the repo root
        sys.path.insert(0, str(REPO))
    from call1.process.training.examples import split_of

    return split_of(installation_id, call_id)


def override_every_verdict(stack: Stack, reviewer, call_id: str) -> int:
    """Flip every QA verdict of the call (PASS -> FAIL, anything else -> PASS). Returns the count."""
    evaluation = reviewer.get(f"/calls/{call_id}/evaluation")
    if evaluation.status_code != 200:
        raise StackError(f"getEvaluation {call_id} answered {evaluation.status_code}: {evaluation.text}")
    done = 0
    for verdict in evaluation.json().get("verdicts", []):
        review = reviewer.get(f"/calls/{call_id}/review").json()
        body = {"status": "FAIL" if verdict["status"] != "FAIL" else "PASS", "reason_code": "model_misread_evidence",
                "evaluation_version": review["current_evaluation_version"], "expected_version": review["review_version"]}
        response = reviewer.post(f"/calls/{call_id}/verdicts/{verdict['criterion_id']}", json=body)
        if response.status_code != 200:
            raise StackError(f"override {call_id}/{verdict['criterion_id']} answered {response.status_code}: {response.text}")
        done += 1
    return done


def wait_until_no_jobs(stack: Stack, timeout: float = 60.0) -> None:
    """Until Store lists no QUEUED or RUNNING job. A conversation reads as settled while its
    supporting jobs (summaries) can still be finishing, and Train now's start rule waits a full
    recheck interval (60 s) for any QUEUED or RUNNING model job; a test must not race it."""

    def idle() -> bool:
        for status in ("QUEUED", "RUNNING"):
            response = stack.store_get("/jobs", session="service", params={"status": status, "limit": 1})
            if response.status_code != 200:
                raise StackError(f"listJobs answered {response.status_code}: {response.text}")
            if response.json().get("items"):
                return False
        return True

    stack.wait_for(idle, timeout=timeout, interval=0.25, what="no queued or running jobs")


def seed_training_labels(stack: Stack, *, min_train_calls: int = 1, min_eval_calls: int = 1, batch: int = 4,
                         max_calls: int = 40) -> Dict[str, Any]:
    """Ingest, settle and override calls until both splits are covered. Returns the counts."""
    reviewer = stack.user("reviewer")
    installation = str(stack.installation_id)
    splits: Dict[str, List[str]] = {"train": [], "valid": [], "eval": []}
    while len(splits["train"]) < min_train_calls or len(splits["eval"]) < min_eval_calls:
        ingested = sum(len(v) for v in splits.values())
        if ingested >= max_calls:
            raise StackError(f"no call landed in both training splits after {ingested} calls: {splits}")
        calls = [stack.ingest("call_01_compliant")["call_id"] for _ in range(min(batch, max_calls - ingested))]
        for call_id in calls:
            stack.wait_until_settled(call_id)
            splits[_split_of(installation, call_id)].append(call_id)
    labels = 0
    for split in ("eval", "valid", "train"):  # the training calls' labels are the newest (see above)
        for call_id in splits[split]:
            labels += override_every_verdict(stack, reviewer, call_id)
    wait_until_no_jobs(stack)
    return {"calls": sum(len(v) for v in splits.values()), "train_calls": len(splits["train"]), "valid_calls": len(splits["valid"]),
            "eval_calls": len(splits["eval"]), "labels": labels}


__all__ = ["override_every_verdict", "seed_training_labels", "wait_until_no_jobs"]
