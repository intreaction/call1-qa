"""On-device training: settings (P1) and schedule semantics (P2), docs/OnDeviceTraining.md 3.1-3.2.
No MLX, torch or real model runs."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from call1.process.config import ConfigError, ProcessConfig
from call1.process.training import settings as settings_mod
from call1.process.training.scheduler import START_WINDOW, TrainingService, latest_occurrence, next_occurrence, occurrence_on
from call1.process.training.settings import SettingsError, TrainingSettings

from .training_support import FakeStore, fake_base, registry_for

PHOENIX = ZoneInfo("America/Phoenix")
NEW_YORK = ZoneInfo("America/New_York")


# --- P1: settings ---------------------------------------------------------------------------------


def test_defaults_are_off_daily_at_two_with_the_documented_limits():
    s = TrainingSettings()
    assert (s.enabled, s.schedule.frequency, s.schedule.weekday, s.schedule.time) == (False, "daily", 6, "02:00")
    assert (s.min_new_labels, s.max_duration_minutes, s.only_when_idle, s.max_seq_length, s.keep_versions, s.trainer) == (25, 120, True, 2600, 5, "mlx_lm")
    assert (s.min_labeled_calls, s.min_train_examples, s.min_eval_items, s.max_train_examples) == (8, 40, 20, 2000)
    assert TrainingSettings.from_mapping(None) == s == TrainingSettings.from_mapping({})
    assert ProcessConfig().training == {}


@pytest.mark.parametrize("raw, field", [
    ({"min_new_labels": 0}, "min_new_labels"), ({"min_new_labels": 10001}, "min_new_labels"), ({"min_new_labels": "many"}, "min_new_labels"),
    ({"max_duration_minutes": 14}, "max_duration_minutes"), ({"max_duration_minutes": 721}, "max_duration_minutes"),
    ({"schedule": {"frequency": "hourly"}}, "schedule.frequency"), ({"schedule": {"weekday": 7}}, "schedule.weekday"),
    ({"schedule": {"time": "24:00"}}, "schedule.time"), ({"schedule": {"time": "2:00"}}, "schedule.time"),
    ({"enabled": "yes"}, "enabled"), ({"only_when_idle": 1}, "only_when_idle"), ({"keep_versions": 1}, "keep_versions"),
    ({"trainer": "torch"}, "trainer"), ({"min_new_labels": 2.5}, "min_new_labels"),
])
def test_invalid_settings_name_the_field(raw, field):
    with pytest.raises(SettingsError) as err:
        TrainingSettings.from_mapping(raw)
    assert err.value.field == field
    with pytest.raises(ConfigError):
        ProcessConfig.from_mapping({"training": raw})


def test_console_updates_change_only_console_fields():
    s = TrainingSettings()
    updated = s.with_console_update({"enabled": True, "schedule": {"frequency": "weekly", "weekday": 2}, "min_new_labels": 10})
    assert updated.enabled and updated.schedule.frequency == "weekly" and updated.schedule.weekday == 2 and updated.schedule.time == "02:00"
    for body, field in (({"keep_versions": 3}, "keep_versions"), ({"trainer": "fake"}, "trainer"), ({"schedule": {"hour": 3}}, "schedule.hour")):
        with pytest.raises(SettingsError) as err:
            s.with_console_update(body)
        assert err.value.field == field


def test_a_daily_schedule_with_a_null_weekday_saves_and_keeps_the_saved_weekday():
    """What the console sent for daily before it always sent the weekday: 0 = Monday ... 6 = Sunday."""
    s = TrainingSettings().with_console_update({"schedule": {"frequency": "weekly", "weekday": 0}})
    body = {"enabled": True, "schedule": {"frequency": "daily", "weekday": None, "time": "02:00"}, "min_new_labels": 25,
            "max_duration_minutes": 120, "only_when_idle": True}
    updated = s.with_console_update(body)
    assert updated.enabled and updated.schedule.frequency == "daily" and updated.schedule.weekday == 0


def test_fake_handlers_force_the_fake_trainer_and_the_env_overrides():
    s = TrainingSettings()
    assert s.effective_trainer("real", env={}) == "mlx_lm"
    assert s.effective_trainer("fake", env={}) == "fake"
    assert s.effective_trainer("real", env={"CALL1_PROCESS_TRAINER": "fake"}) == "fake"
    assert s.effective_trainer("fake", env={"CALL1_PROCESS_TRAINER": "mlx_lm"}) == "mlx_lm"


def test_settings_persist_atomically_at_mode_0600_and_round_trip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"store_url": "http://localhost:8010", "service_key": "c1sk_x"}))
    saved = settings_mod.save(path, TrainingSettings().with_console_update({"enabled": True, "schedule": {"time": "03:30"}}))
    raw = json.loads(path.read_text())
    assert raw["store_url"] == "http://localhost:8010" and raw["training"]["enabled"] is True and raw["training"]["schedule"]["time"] == "03:30"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert settings_mod.load(path) == saved
    assert ProcessConfig.load(path, env={}).training["schedule"]["time"] == "03:30"


# --- P2: the schedule -----------------------------------------------------------------------------


def _settings(**schedule) -> TrainingSettings:
    return TrainingSettings.from_mapping({"enabled": True, "schedule": schedule, "min_new_labels": 3})


def test_daily_occurrences_in_phoenix():
    s = _settings(time="02:00")
    now = datetime(2026, 9, 26, 12, 0, tzinfo=PHOENIX)
    assert latest_occurrence(now, s, PHOENIX) == datetime(2026, 9, 26, 2, 0, tzinfo=PHOENIX)
    assert next_occurrence(now, s, PHOENIX) == datetime(2026, 9, 27, 2, 0, tzinfo=PHOENIX)
    early = datetime(2026, 9, 26, 1, 0, tzinfo=PHOENIX)
    assert latest_occurrence(early, s, PHOENIX) == datetime(2026, 9, 25, 2, 0, tzinfo=PHOENIX)
    assert next_occurrence(early, s, PHOENIX) == datetime(2026, 9, 26, 2, 0, tzinfo=PHOENIX)


def test_weekly_occurrences_fall_on_the_weekday():
    s = _settings(frequency="weekly", weekday=6, time="02:00")  # Sunday
    now = datetime(2026, 9, 26, 12, 0, tzinfo=PHOENIX)  # a Saturday
    assert next_occurrence(now, s, PHOENIX) == datetime(2026, 9, 27, 2, 0, tzinfo=PHOENIX)
    assert latest_occurrence(now, s, PHOENIX) == datetime(2026, 9, 20, 2, 0, tzinfo=PHOENIX)
    assert next_occurrence(now, s, PHOENIX).weekday() == 6


def test_a_nonexistent_dst_time_runs_at_the_first_valid_minute_after_it():
    s = _settings(time="02:30")
    spring = occurrence_on(datetime(2026, 3, 8).date(), s, NEW_YORK)  # 02:00-03:00 does not exist
    assert (spring.hour, spring.minute) == (3, 0)
    assert spring.utcoffset() == timedelta(hours=-4)
    autumn = occurrence_on(datetime(2026, 11, 1).date(), _settings(time="01:30"), NEW_YORK)  # ambiguous: the first one
    assert autumn.utcoffset() == timedelta(hours=-4)
    normal = occurrence_on(datetime(2026, 3, 9).date(), s, NEW_YORK)
    assert (normal.hour, normal.minute) == (2, 30)


def _service(tmp_path, store, now, *, settings=None, tz=PHOENIX):
    base = fake_base(tmp_path)
    config = SimpleNamespace(data_dir=tmp_path / "process", config_path=tmp_path / "process" / "config.json", configured=True, handlers="fake",
                             installation_id="inst", training=(settings or _settings()).to_dict())
    clock = {"now": now}
    service = TrainingService(config, client=store, registry=registry_for(tmp_path, base), base_model=base, tz=tz, now=lambda: clock["now"],
                              worker=lambda: None, recheck_seconds=0.01)
    service._queue = lambda trigger, window_end, scheduled: service.__dict__.setdefault("queued", []).append((trigger, window_end, scheduled))  # type: ignore[method-assign]
    return service, clock


def _labels(store: FakeStore, n: int) -> None:
    from call1.contracts.training import TrainingLabelKind

    from .training_support import timestamp

    for i in range(n):
        store.labels.append(SimpleNamespace(seq=len(store.labels) + 1, kind=TrainingLabelKind.QA_VERDICT, recorded_at=timestamp()))


class _CountStore(FakeStore):
    def list_training_labels(self, *, after=0, limit=200, kinds=None, retry=True, timeout=None):
        from call1.contracts.training import TrainingLabelPage

        self.reads.append(("list_training_labels", (after, limit)))
        rows = [label for label in self.labels if label.seq > after]
        return TrainingLabelPage(items=[], next_after=after, count_after=len(rows), high_water=len(self.labels))


def test_below_the_threshold_an_occurrence_is_a_check_and_writes_no_run(tmp_path):
    store = _CountStore()
    _labels(store, 2)
    service, clock = _service(tmp_path, store, datetime(2026, 9, 26, 2, 0, 30, tzinfo=PHOENIX))
    assert service.tick() == "check"
    check = service.state()["last_check"]
    assert (check["new_labels"], check["min_new_labels"], check["started"]) == (2, 3, False)
    assert not getattr(service, "queued", None) and service.history() == []
    assert service.tick() is None  # checked once per occurrence
    assert store.reads[-1] == ("list_training_labels", (0, 0))  # a count-only read


def test_at_the_threshold_the_occurrence_queues_a_run_with_a_four_hour_window(tmp_path):
    store = _CountStore()
    _labels(store, 3)
    occurrence = datetime(2026, 9, 26, 2, 0, tzinfo=PHOENIX)
    service, clock = _service(tmp_path, store, occurrence + timedelta(minutes=1))
    assert service.tick() == "queued"
    assert service.queued == [("schedule", occurrence + START_WINDOW, True)]
    assert service.tick() is None


def test_a_missed_occurrence_runs_while_its_window_is_open_and_is_skipped_after(tmp_path):
    store = _CountStore()
    _labels(store, 5)
    occurrence = datetime(2026, 9, 26, 2, 0, tzinfo=PHOENIX)
    service, clock = _service(tmp_path, store, occurrence + timedelta(hours=3, minutes=59))  # the host woke up late
    assert service.tick() == "queued"

    store2 = _CountStore()
    _labels(store2, 5)
    other, clock2 = _service(tmp_path / "second", store2, occurrence + timedelta(hours=4, minutes=1))
    assert other.tick() == "missed"
    assert not getattr(other, "queued", None)
    assert other.tick() is None


def test_the_threshold_counts_from_the_last_runs_cursor(tmp_path):
    store = _CountStore()
    _labels(store, 30)
    service, clock = _service(tmp_path, store, datetime(2026, 9, 26, 2, 1, tzinfo=PHOENIX))
    service._write_state(cursor=28)
    assert service.tick() == "check"
    assert service.state()["last_check"]["new_labels"] == 2
    assert store.reads[-1] == ("list_training_labels", (28, 0))


def test_disabled_scheduling_never_checks(tmp_path):
    store = _CountStore()
    _labels(store, 30)
    service, clock = _service(tmp_path, store, datetime(2026, 9, 26, 2, 1, tzinfo=PHOENIX), settings=TrainingSettings())
    assert service.tick() is None and store.reads == []


def test_a_dst_zone_schedules_in_local_time(tmp_path):
    store = _CountStore()
    _labels(store, 5)
    at = datetime(2026, 3, 8, 3, 0, tzinfo=NEW_YORK)  # 02:30 does not exist that night: it runs at 03:00
    service, clock = _service(tmp_path, store, at, settings=_settings(time="02:30"), tz=NEW_YORK)
    assert service.tick() == "queued"
    assert service.queued[0][1] == at + START_WINDOW
    assert service.describe()["timezone"] == "America/New_York"
    assert service.describe()["next_run_at"] == datetime(2026, 3, 9, 2, 30, tzinfo=NEW_YORK).isoformat()
