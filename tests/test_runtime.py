from face_attendance.attendance import AttendanceLog
from face_attendance.config import AppConfig
from face_attendance.liveness import CallableLivenessChecker
from face_attendance.runtime import build_runtime


def test_build_runtime_initializes_local_data_files(tmp_path):
    config = AppConfig(
        registry_path=tmp_path / "registry.json",
        attendance_path=tmp_path / "attendance.csv",
        require_liveness=False,
    )

    runtime = build_runtime(config)

    assert runtime.config is config
    assert runtime.registry.path == config.registry_path
    assert runtime.attendance.path == config.attendance_path
    assert config.registry_path.is_file()
    assert config.attendance_path.is_file()
    assert runtime.liveness_policy.checker is None
    assert runtime.liveness_policy.required is False


def test_build_runtime_accepts_a_liveness_checker(tmp_path):
    config = AppConfig(
        registry_path=tmp_path / "registry.json",
        attendance_path=tmp_path / "attendance.csv",
    )
    checker = CallableLivenessChecker(lambda frame: True)

    runtime = build_runtime(config, checker)

    result = runtime.liveness_policy.evaluate(object())

    assert result.allowed is True
    assert runtime.liveness_policy.checker is checker


def test_runtime_attendance_is_ready_for_events(tmp_path):
    config = AppConfig(
        registry_path=tmp_path / "registry.json",
        attendance_path=tmp_path / "attendance.csv",
        require_liveness=False,
    )
    runtime = build_runtime(config)

    event = runtime.attendance.record("Ada", "in")

    assert event.name == "Ada"
    assert event.action == "in"
    assert AttendanceLog(config.attendance_path).events() == (event,)
