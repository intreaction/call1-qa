"""The ``training`` key of the Process config (docs/OnDeviceTraining.md section 3.1).

| Key | Default | Meaning |
|---|---|---|
| ``enabled`` | ``false`` | Scheduled training on or off. **Train now** works either way |
| ``schedule.frequency`` | ``daily`` | ``daily`` or ``weekly`` |
| ``schedule.weekday`` | ``6`` (Sunday) | Weekly only: 0 = Monday ... 6 = Sunday |
| ``schedule.time`` | ``02:00`` | ``HH:MM``, 24-hour, in the Process host's local time zone |
| ``min_new_labels`` | 25 | Scheduled runs start only when this many label rows arrived since the last run (1-10000) |
| ``max_duration_minutes`` | 120 | From pausing claims to resuming them (15-720); past it the run is killed as ``timed_out`` |
| ``only_when_idle`` | ``true`` | Scheduled runs also wait for the whole Process to be idle |
| ``max_seq_length`` | 2600 | The training budget per example (read-only in the console; raise after qualification L3) |
| ``keep_versions`` | 5 | Promoted adapters kept for rollback (2-20) |
| ``trainer`` | ``mlx_lm`` | ``mlx_lm`` or ``fake``. Fake-handler mode forces ``fake``; ``CALL1_PROCESS_TRAINER`` overrides |
| ``min_labeled_calls`` / ``min_train_examples`` / ``min_eval_items`` / ``max_train_examples`` | 8 / 40 / 20 / 2000 | Minimums and the cap (section 1.7) |

The console writes the first six (``PUT /process/api/training/settings``); the rest are config-only
knobs. ``save`` merges the key into the config file atomically at mode 0600 under a process-wide
lock.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

FREQUENCIES = ("daily", "weekly")
TRAINERS = ("mlx_lm", "fake")
CONSOLE_FIELDS = ("enabled", "schedule", "min_new_labels", "max_duration_minutes", "only_when_idle")
_TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_lock = threading.Lock()


class SettingsError(ValueError):
    """An invalid training setting. ``field`` is the dotted key the console highlights."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"training.{field}: {message}")
        self.field = field
        self.message = message


@dataclass(frozen=True)
class Schedule:
    frequency: str = "daily"
    weekday: int = 6
    time: str = "02:00"

    @property
    def hour(self) -> int:
        return int(self.time[:2])

    @property
    def minute(self) -> int:
        return int(self.time[3:])


@dataclass(frozen=True)
class TrainingSettings:
    enabled: bool = False
    schedule: Schedule = field(default_factory=Schedule)
    min_new_labels: int = 25
    max_duration_minutes: int = 120
    only_when_idle: bool = True
    max_seq_length: int = 2600
    keep_versions: int = 5
    trainer: str = "mlx_lm"
    min_labeled_calls: int = 8
    min_train_examples: int = 40
    min_eval_items: int = 20
    max_train_examples: int = 2000

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "TrainingSettings":
        raw = dict(raw or {})
        if not isinstance(raw, dict):
            raise SettingsError("training", "must be an object")
        base = cls()
        schedule_raw = raw.get("schedule") or {}
        if not isinstance(schedule_raw, Mapping):
            raise SettingsError("schedule", "must be an object")
        frequency = str(schedule_raw.get("frequency", base.schedule.frequency))
        if frequency not in FREQUENCIES:
            raise SettingsError("schedule.frequency", "must be daily or weekly")
        weekday = _int(schedule_raw.get("weekday", base.schedule.weekday), "schedule.weekday", 0, 6)
        time = str(schedule_raw.get("time", base.schedule.time))
        if not _TIME.match(time):
            raise SettingsError("schedule.time", "must be HH:MM, 24-hour")
        trainer = str(raw.get("trainer", base.trainer))
        if trainer not in TRAINERS:
            raise SettingsError("trainer", "must be mlx_lm or fake")
        return cls(
            enabled=_bool(raw.get("enabled", base.enabled), "enabled"),
            schedule=Schedule(frequency=frequency, weekday=weekday, time=time),
            min_new_labels=_int(raw.get("min_new_labels", base.min_new_labels), "min_new_labels", 1, 10000),
            max_duration_minutes=_int(raw.get("max_duration_minutes", base.max_duration_minutes), "max_duration_minutes", 15, 720),
            only_when_idle=_bool(raw.get("only_when_idle", base.only_when_idle), "only_when_idle"),
            max_seq_length=_int(raw.get("max_seq_length", base.max_seq_length), "max_seq_length", 256, 32768),
            keep_versions=_int(raw.get("keep_versions", base.keep_versions), "keep_versions", 2, 20),
            trainer=trainer,
            min_labeled_calls=_int(raw.get("min_labeled_calls", base.min_labeled_calls), "min_labeled_calls", 0, 100000),
            min_train_examples=_int(raw.get("min_train_examples", base.min_train_examples), "min_train_examples", 0, 1000000),
            min_eval_items=_int(raw.get("min_eval_items", base.min_eval_items), "min_eval_items", 1, 100000),
            max_train_examples=_int(raw.get("max_train_examples", base.max_train_examples), "max_train_examples", 1, 1000000),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def console_view(self) -> Dict[str, Any]:
        """What ``GET /process/api/training`` shows: every setting, with the console-editable ones first."""
        return self.to_dict()

    def with_console_update(self, body: Mapping[str, Any]) -> "TrainingSettings":
        """Apply a ``PUT /process/api/training/settings`` body. Only the console fields change; an
        unknown field is refused so a typo never saves silently. ``schedule.weekday: null`` keeps the
        saved weekday."""
        unknown = set(body) - set(CONSOLE_FIELDS)
        if unknown:
            raise SettingsError(sorted(unknown)[0], "is not a setting the console can change")
        merged = self.to_dict()
        for key, value in body.items():
            if key == "schedule":
                if not isinstance(value, Mapping):
                    raise SettingsError("schedule", "must be an object")
                extra = set(value) - {"frequency", "weekday", "time"}
                if extra:
                    raise SettingsError(f"schedule.{sorted(extra)[0]}", "is not a schedule setting")
                # a null weekday (a daily schedule has no use for one) keeps the saved weekday
                merged["schedule"] = {**merged["schedule"], **{k: v for k, v in dict(value).items() if not (k == "weekday" and v is None)}}
            else:
                merged[key] = value
        return TrainingSettings.from_mapping(merged)

    def effective_trainer(self, handlers: str, env: Optional[Mapping[str, str]] = None) -> str:
        """``CALL1_PROCESS_TRAINER`` overrides; fake-handler mode forces ``fake``; else the setting."""
        env = os.environ if env is None else env
        override = (env.get("CALL1_PROCESS_TRAINER") or "").strip().lower()
        if override in TRAINERS:
            return override
        if handlers == "fake":
            return "fake"
        return self.trainer


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise SettingsError(name, "must be true or false")


def _int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SettingsError(name, f"must be a whole number from {low} to {high}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise SettingsError(name, f"must be a whole number from {low} to {high}") from None
    if isinstance(value, float) and value != number:
        raise SettingsError(name, f"must be a whole number from {low} to {high}")
    if not low <= number <= high:
        raise SettingsError(name, f"must be from {low} to {high}")
    return number


def save(config_path: Path, settings: TrainingSettings) -> TrainingSettings:
    """Persist ``settings`` as the config's ``training`` key (atomic, mode 0600), under the lock."""
    from ..config import update_config_file

    with _lock:
        update_config_file(Path(config_path), {"training": settings.to_dict()})
    return settings


def load(config_path: Path, fallback: Optional[Mapping[str, Any]] = None) -> TrainingSettings:
    """The ``training`` key as the config file holds it now (``fallback`` when the file has none)."""
    from ..config import read_config_file

    with _lock:
        raw = read_config_file(Path(config_path)).get("training")
    return TrainingSettings.from_mapping(raw if raw is not None else fallback)


__all__ = ["CONSOLE_FIELDS", "Schedule", "SettingsError", "TrainingSettings", "load", "save"]
