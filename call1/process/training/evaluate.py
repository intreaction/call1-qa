"""Scoring held-out answers with the engines' parsers, and the promotion rule
(docs/OnDeviceTraining.md section 4.4).

An unparseable answer counts as wrong and as ``invalid``. The candidate is promoted only when:

1. overall accuracy (micro, over every held-out item) is at least the active adapter's;
2. every task with at least 10 held-out items loses at most 0.05;
3. invalid answers are no more than the active adapter's;
4. there are at least ``min_eval_items`` held-out items.

Ties promote, so newer labels get in. The reason is a sentence with the numbers, for example
``promoted: overall 0.842 >= 0.815 (n=146)`` or ``rejected: qa_verdict fell 0.72 -> 0.64 (n=25)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .examples import EvalItem

TASK_MIN_ITEMS = 10
TASK_MAX_DROP = 0.05


@dataclass
class Score:
    n: int = 0
    correct: int = 0
    invalid: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


@dataclass
class Scores:
    tasks: Dict[str, Score] = field(default_factory=dict)

    @property
    def overall(self) -> Score:
        total = Score()
        for score in self.tasks.values():
            total.n += score.n
            total.correct += score.correct
            total.invalid += score.invalid
        return total


def score(items: List[EvalItem], answers: Dict[str, Optional[str]]) -> Scores:
    out = Scores()
    for item in items:
        task = out.tasks.setdefault(item.task, Score())
        task.n += 1
        try:
            correct, invalid = item.judge(answers)
        except Exception:
            correct, invalid = False, True
        task.correct += int(bool(correct))
        task.invalid += int(bool(invalid))
    return out


def decide(active: Scores, candidate: Scores, *, min_eval_items: int) -> Tuple[bool, str]:
    a, c = active.overall, candidate.overall
    if c.n < min_eval_items:
        return False, f"rejected: {c.n} held-out items; at least {min_eval_items} are needed"
    for task, cand in sorted(candidate.tasks.items()):
        base = active.tasks.get(task, Score())
        if cand.n >= TASK_MIN_ITEMS and cand.accuracy < base.accuracy - TASK_MAX_DROP - 1e-12:
            return False, f"rejected: {task} fell {base.accuracy:.2f} → {cand.accuracy:.2f} (n={cand.n})"
    if c.invalid > a.invalid:
        return False, f"rejected: invalid answers {c.invalid} > {a.invalid}"
    if c.accuracy + 1e-12 < a.accuracy:
        return False, f"rejected: overall {c.accuracy:.3f} < {a.accuracy:.3f} (n={c.n})"
    return True, f"promoted: overall {c.accuracy:.3f} ≥ {a.accuracy:.3f} (n={c.n})"


def adapter_scope(active: Scores, candidate: Scores) -> Tuple[List[str], List[str]]:
    """(the tasks a promoted adapter applies to, the held-out tasks it is held back from). A task
    joins the scope only when the candidate is not worse on it: accuracy at least the baseline's and
    no more invalid answers. The per-task drop rule in ``decide`` protects only tasks with at least
    ``TASK_MIN_ITEMS`` held-out items, so without this a small task could fall (qualification L1: QA
    went 6/6 -> 1/6 on six items) and still be served by the adapter; a held-back task stays on the
    base until a later candidate is not worse on it."""
    tasks: List[str] = []
    held_back: List[str] = []
    for task, cand in sorted(candidate.tasks.items()):
        if cand.n == 0:
            continue
        base = active.tasks.get(task, Score())
        if cand.accuracy + 1e-12 >= base.accuracy and cand.invalid <= base.invalid:
            tasks.append(task)
        else:
            held_back.append(task)
    return tasks, held_back


def metrics(active: Scores, candidate: Scores) -> Dict[str, object]:
    """The run record's ``eval`` block: numbers only."""
    tasks = {}
    for task in sorted(set(active.tasks) | set(candidate.tasks)):
        a, c = active.tasks.get(task, Score()), candidate.tasks.get(task, Score())
        tasks[task] = {"n": c.n or a.n, "active": round(a.accuracy, 4), "candidate": round(c.accuracy, 4), "invalid_active": a.invalid,
                       "invalid_candidate": c.invalid}
    a, c = active.overall, candidate.overall
    return {"tasks": tasks, "overall": {"n": c.n or a.n, "active": round(a.accuracy, 4), "candidate": round(c.accuracy, 4)}}


__all__ = ["Score", "Scores", "TASK_MAX_DROP", "TASK_MIN_ITEMS", "adapter_scope", "decide", "metrics", "score"]
