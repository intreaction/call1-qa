"""From built examples to the files the trainer reads (docs/OnDeviceTraining.md sections 1.1, 1.7, 4.2).

* **Dedup.** Examples with the same digest (``canonical_digest`` of the three messages) collapse into
  one; the same prompt with different answers keeps the answer from the newest label.
* **Split by call** (``examples.split_of``): held-out calls became evaluation items; the validation
  bucket feeds mlx_lm's validation loss only. When it is empty, a deterministic 10% of the training
  examples (at least 1) move to validation.
* **Cap.** At most ``max_train_examples`` training examples, newest labels first.
* **Minimums.** ``min_labeled_calls``, ``min_train_examples`` and ``min_eval_items``: a run that
  misses one ends ``skipped`` with the counts and never touches the GPU.
* **Files.** ``train.jsonl`` and ``valid.jsonl`` hold only ``{"messages": [system, user,
  assistant]}``; the metadata (label seqs, task, call hash, example digest) goes in a parallel
  ``index.jsonl``. ``eval-prompts.jsonl`` holds the held-out prompts for the generation worker.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from call1.contracts.common import canonical_digest

from .examples import BuildResult, EvalItem, EvalPrompt, Example


@dataclass
class Dataset:
    train: List[Example]
    valid: List[Example]
    prompts: Dict[str, EvalPrompt]
    items: List[EvalItem]
    skipped: Counter = field(default_factory=Counter)
    notes: Counter = field(default_factory=Counter)
    duplicates: int = 0

    @property
    def digest(self) -> str:
        return canonical_digest(sorted(e.digest for e in self.train + self.valid))

    def by_task(self) -> Dict[str, int]:
        return dict(Counter(e.task for e in self.train))

    def eval_by_task(self) -> Dict[str, int]:
        return dict(Counter(i.task for i in self.items))

    def labeled_calls(self) -> int:
        return len({e.call_id for e in self.train})

    def counts(self) -> Dict[str, Any]:
        return {"train": len(self.train), "valid": len(self.valid), "eval_items": len(self.items), "by_task": self.by_task(),
                "eval_by_task": self.eval_by_task(), "labeled_calls": self.labeled_calls(), "duplicates": self.duplicates}


def assemble(build: BuildResult, *, max_train_examples: int) -> Dataset:
    newest_by_prompt: Dict[str, Example] = {}
    duplicates = 0
    for example in sorted(build.examples, key=lambda e: e.newest):
        key = example.prompt_digest
        if key in newest_by_prompt:
            duplicates += 1
        newest_by_prompt[key] = example  # the newest label's answer wins
    kept = list(newest_by_prompt.values())
    train = [e for e in kept if e.split == "train"]
    valid = [e for e in kept if e.split == "valid"]
    train.sort(key=lambda e: (-e.newest, e.digest))
    if len(train) > max_train_examples:
        train = train[:max_train_examples]
    if not valid and train:
        count = max(1, math.ceil(len(train) * 0.1)) if len(train) > 1 else 0
        chosen = set(sorted(e.digest for e in train)[:count])
        valid = [e for e in train if e.digest in chosen]
        train = [e for e in train if e.digest not in chosen]
    return Dataset(train=train, valid=valid, prompts=dict(build.prompts), items=list(build.items), skipped=Counter(build.skipped),
                   notes=Counter(build.notes), duplicates=duplicates)


def unmet_minimum(dataset: Dataset, *, min_labeled_calls: int, min_train_examples: int, min_eval_items: int) -> Optional[str]:
    """The reason a run is skipped, or None when every minimum is met."""
    if dataset.labeled_calls() < min_labeled_calls:
        return f"{dataset.labeled_calls()} labelled calls in the training split; at least {min_labeled_calls} are needed"
    if len(dataset.train) < min_train_examples:
        return f"{len(dataset.train)} training examples; at least {min_train_examples} are needed"
    if len(dataset.items) < min_eval_items:
        return f"{len(dataset.items)} held-out items; at least {min_eval_items} are needed to judge a candidate"
    return None


def _call_hash(call_id: str) -> str:
    return hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:16]


def _write_jsonl(path: Path, rows) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write(dataset: Dataset, data_dir: Path) -> Dict[str, Path]:
    """Write the trainer's files into ``data_dir`` (created 0700)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(data_dir, 0o700)
    paths = {name: data_dir / f"{name}.jsonl" for name in ("train", "valid", "index")}
    _write_jsonl(paths["train"], ({"messages": e.messages} for e in dataset.train))
    _write_jsonl(paths["valid"], ({"messages": e.messages} for e in dataset.valid))
    _write_jsonl(paths["index"], ({"split": split, "task": e.task, "label_seqs": e.seqs, "call": _call_hash(e.call_id), "digest": e.digest,
                                   "tokens": e.tokens} for split, group in (("train", dataset.train), ("valid", dataset.valid)) for e in group))
    return paths


def write_eval_prompts(dataset: Dataset, path: Path, tasks: Optional[Iterable[str]] = None) -> Path:
    """The held-out prompts the generation worker answers; with ``tasks``, only the prompts of those
    tasks (the active pass answers each task on what production uses for it)."""
    wanted = None if tasks is None else set(tasks)
    _write_jsonl(Path(path), (prompt.record() for prompt in dataset.prompts.values() if wanted is None or prompt.task in wanted))
    return Path(path)


__all__ = ["Dataset", "assemble", "unmet_minimum", "write", "write_eval_prompts"]
