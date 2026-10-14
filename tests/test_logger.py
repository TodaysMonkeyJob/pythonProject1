import os
from datetime import datetime, timedelta, timezone

from core.events import Event, make_event
from core.logger import JsonlLogger


class FakeClock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now


def _big_event(size=1000):
    return make_event("gas", "gas_reading", {"blob": "x" * size})


def test_writes_jsonl_to_day_directory(tmp_path):
    clock = FakeClock(datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc))
    events = [make_event("gas", "gas_reading", {"i": i}) for i in range(3)]
    with JsonlLogger("gas", base_dir=str(tmp_path), clock=clock) as log:
        for event in events:
            log.write(event)

    path = tmp_path / "2026-10-14" / "gas.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [Event.from_json(line) for line in lines] == events


def test_rotates_daily(tmp_path):
    clock = FakeClock(datetime(2026, 10, 14, 23, 59, 59, tzinfo=timezone.utc))
    with JsonlLogger("audio", base_dir=str(tmp_path), clock=clock) as log:
        log.write(make_event("audio", "speech", {"text": "before"}))
        clock.now += timedelta(seconds=2)
        log.write(make_event("audio", "speech", {"text": "after"}))
        assert log.current_path.endswith(os.path.join("2026-10-15", "audio.jsonl"))

    assert len((tmp_path / "2026-10-14" / "audio.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "2026-10-15" / "audio.jsonl").read_text().splitlines()) == 1


def test_size_cap_deletes_oldest_days_first(tmp_path):
    clock = FakeClock(datetime(2026, 10, 10, tzinfo=timezone.utc))
    cap_mb = 5000 / (1024 * 1024)  # ~5 KB total
    with JsonlLogger("gas", base_dir=str(tmp_path), max_total_mb=cap_mb, clock=clock) as log:
        for _ in range(5):  # 5 days x ~2 KB
            log.write(_big_event())
            log.write(_big_event())
            clock.now += timedelta(days=1)
        log.write(_big_event())
        log.enforce_size_limit()

    remaining = sorted(os.listdir(tmp_path))
    assert remaining == ["2026-10-14", "2026-10-15"]
    total = sum(f.stat().st_size for f in tmp_path.rglob("*.jsonl"))
    assert total <= 5000


def test_size_cap_checked_periodically_within_a_day(tmp_path):
    (tmp_path / "2026-10-01").mkdir()
    (tmp_path / "2026-10-01" / "gas.jsonl").write_text("x" * 10_000)
    clock = FakeClock(datetime(2026, 10, 14, tzinfo=timezone.utc))
    cap_mb = 20_000 / (1024 * 1024)
    with JsonlLogger("gas", base_dir=str(tmp_path), max_total_mb=cap_mb, check_every=5, clock=clock) as log:
        for _ in range(12):
            log.write(_big_event())

    assert os.listdir(tmp_path) == ["2026-10-14"]


def test_never_deletes_current_day(tmp_path):
    clock = FakeClock(datetime(2026, 10, 14, tzinfo=timezone.utc))
    with JsonlLogger("gas", base_dir=str(tmp_path), max_total_mb=0, clock=clock) as log:
        log.write(_big_event())
        log.enforce_size_limit()
        log.write(_big_event())

    assert len((tmp_path / "2026-10-14" / "gas.jsonl").read_text().splitlines()) == 2


def test_ignores_foreign_directories(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "keep.txt").write_text("x" * 10_000)
    clock = FakeClock(datetime(2026, 10, 14, tzinfo=timezone.utc))
    with JsonlLogger("gas", base_dir=str(tmp_path), max_total_mb=0.001, clock=clock) as log:
        log.write(_big_event())

    assert (tmp_path / "archive" / "keep.txt").exists()
