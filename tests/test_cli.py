from face_attendance.cli import main
from face_attendance.config import AppConfig
from face_attendance.registry import FaceRegistry


def test_remove_command_does_not_require_vision_dependencies(tmp_path, capsys):
    registry_path = tmp_path / "registry.json"
    registry = FaceRegistry(registry_path)
    registry.register("Ada", [0.1, 0.2])
    config_path = tmp_path / "config.json"
    config_path.write_text(
        f'{{"registry_path": "{registry_path.as_posix()}"}}',
        encoding="utf-8",
    )

    result = main(["--config", str(config_path), "remove", "ada"])

    assert result == 0
    assert "Removed Ada" in capsys.readouterr().out
    assert registry.names() == ()


def test_remove_command_reports_unknown_user(tmp_path, capsys):
    config = AppConfig(
        registry_path=tmp_path / "registry.json",
        attendance_path=tmp_path / "attendance.csv",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        f'{{"registry_path": "{config.registry_path.as_posix()}"}}',
        encoding="utf-8",
    )

    result = main(["--config", str(config_path), "remove", "Nobody"])

    assert result == 0
    assert "No registered user" in capsys.readouterr().out


def test_doctor_command_validates_data_files(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        f'{{"registry_path": "{(tmp_path / "registry.json").as_posix()}", '
        f'"attendance_path": "{(tmp_path / "attendance.csv").as_posix()}"}}',
        encoding="utf-8",
    )

    result = main(["--config", str(config_path), "doctor"])

    assert result == 0
    output = capsys.readouterr().out
    assert "Registry OK" in output
    assert "Attendance log OK" in output
