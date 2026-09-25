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
