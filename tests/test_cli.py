import json

import pytest

from face_attendance.attendance import AttendanceLog
from face_attendance.cli import _load_image, _print_attendance, _print_result, main
from face_attendance.config import AppConfig
from face_attendance.errors import FaceAttendanceError
from face_attendance.recognition import RecognitionResult, RecognitionStatus
from face_attendance.registry import FaceRegistry


def write_config(tmp_path, **values):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(values), encoding="utf-8")
    return config_path


def test_init_command_creates_data_files(tmp_path, capsys):
    config_path = write_config(
        tmp_path,
        registry_path=(tmp_path / "registry.json").as_posix(),
        attendance_path=(tmp_path / "attendance.csv").as_posix(),
    )

    result = main(["--config", str(config_path), "init"])

    assert result == 0
    assert (tmp_path / "registry.json").is_file()
    assert (tmp_path / "attendance.csv").is_file()
    output = capsys.readouterr().out
    assert "Registry:" in output
    assert "Attendance log:" in output


def test_list_command_reports_users_and_empty_state(tmp_path, capsys):
    config_path = write_config(tmp_path, registry_path=(tmp_path / "registry.json").as_posix())
    registry = FaceRegistry(tmp_path / "registry.json")
    registry.register("Ada", [0.1, 0.2])

    assert main(["--config", str(config_path), "list"]) == 0
    assert capsys.readouterr().out == "Ada\n"

    empty_path = write_config(
        tmp_path / "empty",
        registry_path=(tmp_path / "empty" / "registry.json").as_posix(),
    )
    assert main(["--config", str(empty_path), "list"]) == 0
    assert "No users registered" in capsys.readouterr().out


def test_attendance_command_prints_events_or_empty_state(tmp_path, capsys):
    config = AppConfig(
        registry_path=tmp_path / "registry.json",
        attendance_path=tmp_path / "attendance.csv",
    )
    config_path = write_config(
        tmp_path,
        registry_path=config.registry_path.as_posix(),
        attendance_path=config.attendance_path.as_posix(),
    )

    assert main(["--config", str(config_path), "attendance"]) == 0
    assert "No attendance events recorded" in capsys.readouterr().out

    AttendanceLog(config.attendance_path).record("Ada", "in")
    assert main(["--config", str(config_path), "attendance"]) == 0
    assert "\tAda\tin" in capsys.readouterr().out


def test_main_reports_invalid_configuration(tmp_path, capsys):
    result = main(["--config", str(tmp_path / "missing.json"), "doctor"])

    assert result == 1
    assert "Error:" in capsys.readouterr().err


def test_load_image_rejects_missing_file(tmp_path, monkeypatch):
    class FakeNumpy:
        uint8 = "uint8"

        @staticmethod
        def fromfile(path, dtype):
            return FakeNumpy()

        size = 0

    monkeypatch.setattr("face_attendance.cli.importlib.import_module", lambda name: FakeNumpy())

    with pytest.raises(FaceAttendanceError, match="Unable to read image"):
        _load_image(tmp_path / "missing.jpg")


def test_load_image_reports_missing_vision_dependencies(tmp_path, monkeypatch):
    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr("face_attendance.cli.importlib.import_module", unavailable)

    with pytest.raises(FaceAttendanceError, match="Vision dependencies are unavailable"):
        _load_image(tmp_path / "missing.jpg")


def test_print_helpers_emit_machine_readable_output(capsys):
    result = RecognitionResult(
        status=RecognitionStatus.MATCH,
        face_count=1,
        name="Ada",
        distance=0.2,
        confidence=0.8,
    )
    _print_result(result)
    _print_attendance(())

    output = capsys.readouterr().out
    payload = json.loads(output.splitlines()[0])
    assert payload["status"] == "match"
    assert payload["name"] == "Ada"
    assert payload["match_quality"] == 0.8
    assert "No attendance events recorded" in output
