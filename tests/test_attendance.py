from datetime import datetime, timezone

import pytest

from face_attendance.attendance import AttendanceLog
from face_attendance.errors import AttendanceError


def test_attendance_round_trip_and_latest_action(tmp_path):
    log = AttendanceLog(tmp_path / "attendance.csv")
    first_time = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)

    first = log.record("Ada", "in", first_time)
    second = log.record("Ada", "out", first_time.replace(hour=5))

    assert first.timestamp == "2026-01-02T03:04:00+00:00"
    assert log.events() == (first, second)
    assert log.latest_action("Ada") == "out"


def test_attendance_normalizes_naive_timestamps(tmp_path):
    log = AttendanceLog(tmp_path / "attendance.csv")
    event = log.record("Ada", "in", datetime(2026, 1, 2, 3, 4))

    assert event.timestamp == "2026-01-02T03:04:00+00:00"


def test_attendance_rejects_invalid_actions(tmp_path):
    log = AttendanceLog(tmp_path / "attendance.csv")

    with pytest.raises(AttendanceError):
        log.record("Ada", "break")
    with pytest.raises(AttendanceError):
        log.record("Ada", ["in"])


def test_attendance_rejects_invalid_headers(tmp_path):
    path = tmp_path / "attendance.csv"
    path.write_text("name,action\nAda,in\n", encoding="utf-8")

    with pytest.raises(AttendanceError):
        AttendanceLog(path).events()


def test_attendance_enforces_sign_in_and_sign_out_order(tmp_path):
    log = AttendanceLog(tmp_path / "attendance.csv")

    with pytest.raises(AttendanceError):
        log.record("Ada", "out")
    log.record("Ada", "in")
    with pytest.raises(AttendanceError):
        log.record("Ada", "in")
    log.record("Ada", "out")


def test_attendance_rejects_corrupt_rows(tmp_path):
    path = tmp_path / "attendance.csv"
    path.write_text(
        "timestamp,name,action\nnot-a-time,Ada,in\n",
        encoding="utf-8",
    )

    with pytest.raises(AttendanceError):
        AttendanceLog(path).events()


def test_attendance_matches_names_case_insensitively(tmp_path):
    log = AttendanceLog(tmp_path / "attendance.csv")

    log.record("Ada", "in")
    log.record("ADA", "out")

    assert log.latest_action("ada") == "out"
    assert [event.name for event in log.events()] == ["Ada", "ADA"]


def test_attendance_reuses_cached_events_for_appends(tmp_path, monkeypatch):
    log = AttendanceLog(tmp_path / "attendance.csv")
    original = log._read_events_unlocked
    calls = 0

    def counted_read():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(log, "_read_events_unlocked", counted_read)

    log.record("Ada", "in")
    log.record("Ada", "out")
    log.events()

    assert calls == 1


def test_attendance_refreshes_cache_after_external_change(tmp_path):
    path = tmp_path / "attendance.csv"
    log = AttendanceLog(path)
    log.record("Ada", "in")

    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("2026-01-02T05:00:00+00:00,Ada,out\n")

    assert log.latest_action("Ada") == "out"


def test_attendance_rejects_invalid_historical_transition(tmp_path):
    path = tmp_path / "attendance.csv"
    path.write_text(
        "timestamp,name,action\n2026-01-02T03:04:00+00:00,Ada,out\n",
        encoding="utf-8",
    )

    with pytest.raises(AttendanceError):
        AttendanceLog(path).events()
