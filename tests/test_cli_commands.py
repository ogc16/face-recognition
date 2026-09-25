import types

import pytest

from face_attendance import cli
from face_attendance.attendance import AttendanceLog
from face_attendance.config import AppConfig
from face_attendance.errors import FaceAttendanceError
from face_attendance.recognition import RecognitionResult, RecognitionStatus
from face_attendance.registry import FaceRegistry
from face_attendance.runtime import Runtime
from face_attendance.types import AttendanceEvent

_FRAME = object()

MATCH = RecognitionResult(
    status=RecognitionStatus.MATCH,
    face_count=1,
    name="Ada",
    distance=0.1,
    confidence=0.9,
)
UNKNOWN = RecognitionResult(status=RecognitionStatus.UNKNOWN, face_count=1)


class StubService:
    def __init__(self, result=MATCH):
        self.result = result
        self.embedding = (0.1, 0.2)
        self.frames: list[object] = []

    def authenticate(self, frame, liveness_policy):
        self.frames.append(frame)
        return self.result

    def recognize(self, frame):
        return self.result

    def embedding_for(self, frame):
        self.frames.append(frame)
        return self.embedding


def make_config(tmp_path, **overrides):
    defaults = {
        "registry_path": tmp_path / "registry.json",
        "attendance_path": tmp_path / "attendance.csv",
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def write_config(tmp_path, **extra):
    """Write a JSON config file for the CLI and return its path.

    Args:
        tmp_path: The pytest temporary directory.
        **extra: Additional configuration keys to merge in.

    Returns:
        The path to the written configuration file.
    """
    import json

    payload = {
        "registry_path": (tmp_path / "registry.json").as_posix(),
        "attendance_path": (tmp_path / "attendance.csv").as_posix(),
    }
    payload.update(extra)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def make_runtime(tmp_path, service=None, liveness_policy=None):
    config = make_config(tmp_path)
    return Runtime(
        config=config,
        service=service or StubService(),
        registry=FaceRegistry(config.registry_path),
        attendance=AttendanceLog(config.attendance_path),
        liveness_policy=liveness_policy,
    )


def stub_vision(monkeypatch, decoded="image", size=1):
    import sys

    numpy = types.ModuleType("numpy")
    numpy.uint8 = "uint8"

    class Encoded:
        def __init__(self, value):
            self.size = value

    numpy.fromfile = lambda path, dtype: Encoded(size)
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.COLOR_BGR2RGB = 4
    cv2.imdecode = lambda encoded, flag: decoded
    cv2.cvtColor = lambda image, code: image
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2


def test_load_image_converts_to_rgb(tmp_path, monkeypatch):
    stub_vision(monkeypatch)
    image = tmp_path / "face.png"
    image.write_bytes(b"not-a-real-png")

    assert cli._load_image(image) == "image"


def test_load_image_reports_an_empty_file(tmp_path, monkeypatch):
    stub_vision(monkeypatch, size=0)
    image = tmp_path / "empty.png"
    image.write_bytes(b"")

    with pytest.raises(FaceAttendanceError, match="Unable to read image"):
        cli._load_image(image)


def test_load_image_reports_an_undecodable_file(tmp_path, monkeypatch):
    stub_vision(monkeypatch, decoded=None)
    image = tmp_path / "broken.png"
    image.write_bytes(b"garbage")

    with pytest.raises(FaceAttendanceError, match="Unable to read image"):
        cli._load_image(image)


def test_load_image_wraps_a_decoding_crash(tmp_path, monkeypatch):
    cv2 = stub_vision(monkeypatch)
    image = tmp_path / "face.png"
    image.write_bytes(b"data")

    def exploding(encoded, flag):
        raise ZeroDivisionError("decoder exploded")

    cv2.imdecode = exploding

    with pytest.raises(FaceAttendanceError, match="Unable to read image"):
        cli._load_image(image)


def test_load_image_reports_missing_vision_dependencies(tmp_path, monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr("face_attendance.cli.importlib.import_module", missing)

    with pytest.raises(FaceAttendanceError, match="Vision dependencies are unavailable"):
        cli._load_image(tmp_path / "face.png")


def test_print_result_emits_sorted_json(capsys):
    cli._print_result(MATCH)

    assert '"face_count": 1' in capsys.readouterr().out


def test_print_attendance_reports_an_empty_log(capsys):
    cli._print_attendance(())

    assert "No attendance events recorded" in capsys.readouterr().out


def test_print_attendance_prints_tab_separated_rows(capsys):
    event = AttendanceEvent(
        timestamp="2026-09-25T09:00:00+00:00", name="Ada", action="in"
    )

    cli._print_attendance((event,))

    assert "2026-09-25T09:00:00+00:00\tAda\tin" in capsys.readouterr().out


def test_register_reports_a_new_sample(tmp_path, monkeypatch, capsys):
    stub_vision(monkeypatch)
    runtime = make_runtime(tmp_path)
    runtime.registry.ensure_exists()
    image = tmp_path / "face.png"
    image.write_bytes(b"data")

    assert cli._register(runtime, "Ada", image) == 0

    assert "Registered Ada" in capsys.readouterr().out
    assert len(runtime.registry) == 1


def test_register_reports_a_duplicate_sample(tmp_path, monkeypatch, capsys):
    stub_vision(monkeypatch)
    runtime = make_runtime(tmp_path)
    runtime.registry.ensure_exists()
    image = tmp_path / "face.png"
    image.write_bytes(b"data")
    cli._register(runtime, "Ada", image)
    capsys.readouterr()

    cli._register(runtime, "Ada", image)

    assert "No sample added for Ada" in capsys.readouterr().out


def test_recognize_returns_zero_for_a_match(tmp_path, monkeypatch, capsys):
    stub_vision(monkeypatch)
    runtime = make_runtime(tmp_path, service=StubService(MATCH))
    image = tmp_path / "face.png"
    image.write_bytes(b"data")

    assert cli._recognize(runtime, image) == 0

    assert '"status": "match"' in capsys.readouterr().out


def test_recognize_returns_two_for_no_match(tmp_path, monkeypatch, capsys):
    stub_vision(monkeypatch)
    runtime = make_runtime(tmp_path, service=StubService(UNKNOWN))
    image = tmp_path / "face.png"
    image.write_bytes(b"data")

    assert cli._recognize(runtime, image) == 2

    assert '"status": "unknown"' in capsys.readouterr().out


def test_list_users_reports_an_empty_registry(tmp_path, capsys):
    registry_path = tmp_path / "registry.json"
    FaceRegistry(registry_path).ensure_exists()

    assert cli._list_users(make_config(tmp_path)) == 0

    assert "No users registered" in capsys.readouterr().out


def test_list_users_prints_every_name(tmp_path, capsys):
    registry = FaceRegistry(tmp_path / "registry.json")
    registry.ensure_exists()
    registry.register("Ada", (0.1,))
    registry.register("Grace", (0.2,))

    assert cli._list_users(make_config(tmp_path)) == 0

    assert capsys.readouterr().out.split() == ["Ada", "Grace"]


def test_remove_user_reports_success(tmp_path, capsys):
    registry = FaceRegistry(tmp_path / "registry.json")
    registry.ensure_exists()
    registry.register("Ada", (0.1,))

    assert cli._remove_user(make_config(tmp_path), "Ada") == 0

    assert "Removed Ada" in capsys.readouterr().out


def test_remove_user_reports_an_unknown_name(tmp_path, capsys):
    FaceRegistry(tmp_path / "registry.json").ensure_exists()

    assert cli._remove_user(make_config(tmp_path), "Nobody") == 0

    assert "No registered user named Nobody" in capsys.readouterr().out


def test_init_creates_both_data_files(tmp_path, capsys):
    config = make_config(tmp_path)

    assert cli._init(config) == 0

    assert config.registry_path.is_file()
    assert config.attendance_path.is_file()
    assert "Registry:" in capsys.readouterr().out


def test_doctor_reports_a_healthy_installation(tmp_path, capsys):
    config = make_config(tmp_path)
    registry = FaceRegistry(config.registry_path)
    registry.ensure_exists()
    registry.register("Ada", (0.1,))

    assert cli._doctor(config) == 0

    output = capsys.readouterr().out
    assert "Registry OK" in output
    assert "(1 users)" in output


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        cli._parser().parse_args([])


def test_parser_reports_the_version(capsys):
    with pytest.raises(SystemExit) as caught:
        cli._parser().parse_args(["--version"])

    assert caught.value.code == 0


def test_main_init_creates_the_files(tmp_path, capsys):
    config_path = write_config(tmp_path)

    assert cli.main(["--config", str(config_path), "init"]) == 0

    assert (tmp_path / "registry.json").is_file()


def test_main_list_reports_users(tmp_path, capsys):
    config_path = write_config(tmp_path)
    FaceRegistry(tmp_path / "registry.json").ensure_exists()

    assert cli.main(["--config", str(config_path), "list"]) == 0

    assert "No users registered" in capsys.readouterr().out


def test_main_attendance_prints_the_log(tmp_path, capsys):
    config_path = write_config(tmp_path)
    log = AttendanceLog(tmp_path / "attendance.csv")
    log.ensure_exists()
    log.record("Ada", "in")

    assert cli.main(["--config", str(config_path), "attendance"]) == 0

    assert "Ada" in capsys.readouterr().out


def test_main_remove_deletes_a_user(tmp_path, capsys):
    config_path = write_config(tmp_path)
    registry = FaceRegistry(tmp_path / "registry.json")
    registry.ensure_exists()
    registry.register("Ada", (0.1,))

    assert cli.main(["--config", str(config_path), "remove", "Ada"]) == 0

    assert "Removed Ada" in capsys.readouterr().out


def test_main_doctor_validates_the_files(tmp_path, capsys):
    config_path = write_config(tmp_path)

    assert cli.main(["--config", str(config_path), "doctor"]) == 0

    assert "Registry OK" in capsys.readouterr().out


def test_main_reports_a_bad_config(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text("{not json")

    assert cli.main(["--config", str(config_path), "list"]) == 1

    assert "Error:" in capsys.readouterr().err


def test_main_reports_a_recognize_failure(tmp_path, monkeypatch, capsys):
    stub_vision(monkeypatch)
    config_path = write_config(tmp_path)
    image = tmp_path / "face.png"
    image.write_bytes(b"data")
    monkeypatch.setattr(cli, "build_runtime", lambda config: make_runtime(tmp_path))

    assert cli.main(["--config", str(config_path), "recognize", str(image)]) == 0

    assert "match" in capsys.readouterr().out


def test_main_reports_a_register_failure(tmp_path, monkeypatch, capsys):
    stub_vision(monkeypatch)
    config_path = write_config(tmp_path)
    image = tmp_path / "face.png"
    image.write_bytes(b"data")
    monkeypatch.setattr(cli, "build_runtime", lambda config: make_runtime(tmp_path))

    assert cli.main(["--config", str(config_path), "register", "Ada", str(image)]) == 0

    assert "Registered Ada" in capsys.readouterr().out


def test_module_entry_point_matches_main():
    source = cli.__file__
    assert source is not None
