import pytest

from face_attendance.config import AppConfig
from face_attendance.errors import ConfigurationError


def test_config_reads_file_and_environment_overrides(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"camera_index": 1, "tolerance": 0.55, "registry_path": "registry.json"}',
        encoding="utf-8",
    )

    config = AppConfig.from_file(
        config_path,
        environ={"FACE_ATTENDANCE_CAMERA_INDEX": "2"},
    )

    assert config.camera_index == 2
    assert config.tolerance == 0.55
    assert config.registry_path == tmp_path / "registry.json"
    assert config.attendance_path == tmp_path / "data" / "attendance.csv"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("camera_index", "not-an-int"),
        ("tolerance", "0"),
        ("require_liveness", "maybe"),
        ("window_width", "100"),
    ],
)
def test_config_rejects_invalid_values(tmp_path, key, value):
    config_path = tmp_path / "config.json"
    config_path.write_text(f'{{"{key}": "{value}"}}', encoding="utf-8")

    with pytest.raises(ConfigurationError):
        AppConfig.from_file(config_path)


def test_config_rejects_same_data_paths(tmp_path):
    with pytest.raises(ConfigurationError):
        AppConfig(registry_path=tmp_path / "data", attendance_path=tmp_path / "data")


def test_config_rejects_unknown_json_keys(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"require_livness": true}', encoding="utf-8")

    with pytest.raises(ConfigurationError):
        AppConfig.from_file(config_path)


def test_config_rejects_unknown_environment_keys(tmp_path):
    with pytest.raises(ConfigurationError):
        AppConfig.from_file(
            tmp_path / "config.json", environ={"FACE_ATTENDANCE_REQUIRE_LIVNESS": "true"}
        )


def test_config_defaults_to_required_liveness():
    assert AppConfig().require_liveness is True


def test_config_rejects_non_finite_tolerance():
    with pytest.raises(ConfigurationError):
        AppConfig(tolerance=float("nan"))


def test_config_rejects_unrepresentable_tolerance():
    with pytest.raises(ConfigurationError):
        AppConfig(tolerance=10**10000)


def test_config_rejects_fractional_direct_integer_values():
    with pytest.raises(ConfigurationError):
        AppConfig(camera_index=1.5)


def test_config_rejects_non_path_data_values(tmp_path):
    with pytest.raises(ConfigurationError):
        AppConfig(registry_path="registry.json", attendance_path=tmp_path / "attendance.csv")


def test_config_rejects_non_boolean_liveness_setting():
    with pytest.raises(ConfigurationError):
        AppConfig(require_liveness="false")
