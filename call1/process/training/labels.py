"""Paging the label log and applying supersession (docs/OnDeviceTraining.md sections 1.1 and 4.2).

Every run rebuilds its dataset from all current labels, starting from ``seq`` 0: training is never
incremental on the previous adapter. The newest label (highest ``seq``) for a subject wins, and a
``withdrawn`` label removes the subject.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from call1.contracts.training import TrainingLabel, TrainingLabelKind

PAGE = 500


@dataclass
class CollectedLabels:
    current: List[TrainingLabel]
    """The newest live label per subject, in ``seq`` order."""
    cursor: int
    """The last ``seq`` read: the run's label cursor (the next scheduled threshold counts from it)."""
    rows: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    """Current labels per kind, plus ``withdrawn``: subjects a withdrawal removed."""


def collect(client, *, page: int = PAGE, cancelled=lambda: False) -> CollectedLabels:
    """Page ``listTrainingLabels`` from ``after=0``, ``page`` at a time, then supersede."""
    after = 0
    rows: List[TrainingLabel] = []
    cursor = 0
    for _ in range(1_000_000):
        if cancelled():
            break
        result = client.list_training_labels(after=after, limit=page)
        rows.extend(result.items)
        cursor = max(cursor, result.next_after)
        if not result.items or result.count_after <= len(result.items):
            break
        after = result.next_after
    return supersede(rows, cursor)


def supersede(rows: List[TrainingLabel], cursor: int = 0) -> CollectedLabels:
    latest: Dict[str, TrainingLabel] = {}
    withdrawn = 0
    for label in sorted(rows, key=lambda r: r.seq):
        if label.withdrawn:
            if latest.pop(label.subject, None) is not None:
                withdrawn += 1
            continue
        latest[label.subject] = label
    current = sorted(latest.values(), key=lambda r: r.seq)
    counts = Counter(label.kind.value for label in current)
    out = {kind.value: counts.get(kind.value, 0) for kind in TrainingLabelKind}
    out["withdrawn"] = withdrawn
    return CollectedLabels(current=current, cursor=max(cursor, max((r.seq for r in rows), default=0)), rows=len(rows), counts=out)


__all__ = ["CollectedLabels", "PAGE", "collect", "supersede"]
