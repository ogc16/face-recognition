import face_attendance.main as entrypoint
from face_attendance.errors import FaceAttendanceError


def test_main_delegates_to_gui(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        f'{{"registry_path": "{(tmp_path / "registry.json").as_posix()}"}}',
        encoding="utf-8",
    )
    received = []

    def fake_run_gui(config):
        received.append(config)

    monkeypatch.setattr("face_attendance.gui.run_gui", fake_run_gui)

    result = entrypoint.main(["--config", str(config_path)])

    assert result == 0
    assert received[0].registry_path == tmp_path / "registry.json"


def test_main_reports_gui_startup_errors(monkeypatch, capsys):
    def fail(config):
        raise FaceAttendanceError("camera unavailable")

    monkeypatch.setattr("face_attendance.gui.run_gui", fail)

    result = entrypoint.main([])

    assert result == 1
    assert "Unable to start face attendance" in capsys.readouterr().err
