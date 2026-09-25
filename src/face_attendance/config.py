import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

ENVIRONMENT_KEYS = {
    "camera_index": "FACE_ATTENDANCE_CAMERA_INDEX",
    "registry_path": "FACE_ATTENDANCE_REGISTRY_PATH",
    "attendance_path": "FACE_ATTENDANCE_ATTENDANCE_PATH",
    "tolerance": "FACE_ATTENDANCE_TOLERANCE",
    "require_liveness": "FACE_ATTENDANCE_REQUIRE_LIVENESS",
    "max_embeddings_per_user": "FACE_ATTENDANCE_MAX_EMBEDDINGS",
    "window_width": "FACE_ATTENDANCE_WINDOW_WIDTH",
    "window_height": "FACE_ATTENDANCE_WINDOW_HEIGHT",
    "storage_backend": "FACE_ATTENDANCE_STORAGE_BACKEND",
    "postgres_dsn": "FACE_ATTENDANCE_POSTGRES_DSN",
}

STORAGE_BACKENDS = ("json", "csv", "sqlite", "postgres")


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera_index: int = 0
    registry_path: Path = Path("data/registry.json")
    attendance_path: Path = Path("data/attendance.csv")
    tolerance: float = 0.6
    require_liveness: bool = True
    max_embeddings_per_user: int = 5
    window_width: int = 1200
    window_height: int = 720
    storage_backend: str = "json"
    postgres_dsn: str | None = None

    def __post_init__(self) -> None:
        _validate_integer("camera_index", self.camera_index, minimum=0)
        _validate_number("tolerance", self.tolerance, minimum=0.01, maximum=1.0)
        _validate_integer(
            "max_embeddings_per_user", self.max_embeddings_per_user, minimum=1, maximum=100
        )
        _validate_integer("window_width", self.window_width, minimum=800)
        _validate_integer("window_height", self.window_height, minimum=600)
        if not isinstance(self.require_liveness, bool):
            raise ConfigurationError("require_liveness must be a boolean")
        if not isinstance(self.registry_path, Path) or not isinstance(self.attendance_path, Path):
            raise ConfigurationError("Data paths must be Path objects")
        if _paths_refer_to_same_file(self.registry_path, self.attendance_path):
            raise ConfigurationError("Registry and attendance paths must be different")
        _validate_storage_backend(self.storage_backend)
        if self.storage_backend == "postgres" and not self.postgres_dsn:
            raise ConfigurationError("postgres_dsn is required when storage_backend is 'postgres'")

    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "AppConfig":
        environment = os.environ if environ is None else environ
        known_environment_keys = set(ENVIRONMENT_KEYS.values())
        unknown_environment_keys = sorted(
            key
            for key in environment
            if key.startswith("FACE_ATTENDANCE_") and key not in known_environment_keys
        )
        if unknown_environment_keys:
            unknown = ", ".join(unknown_environment_keys)
            raise ConfigurationError(f"Unknown face attendance environment variable: {unknown}")
        defaults = cls()
        config_path = Path(path).expanduser() if path is not None else None
        values: dict[str, Any] = {}

        if config_path is not None:
            if not config_path.is_file():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            try:
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ConfigurationError(f"Unable to read configuration: {config_path}") from exc
            if not isinstance(loaded, dict):
                raise ConfigurationError("Configuration must be a JSON object")
            unknown_keys = sorted(set(loaded) - set(ENVIRONMENT_KEYS))
            if unknown_keys:
                unknown = ", ".join(unknown_keys)
                raise ConfigurationError(f"Unknown configuration key: {unknown}")
            values.update(loaded)

        base_directory = config_path.parent if config_path is not None else Path.cwd()

        def get_value(key: str, default: Any) -> Any:
            environment_key = ENVIRONMENT_KEYS[key]
            if environment_key in environment:
                return environment[environment_key]
            if key in values:
                return values[key]
            return default

        registry_path = _resolve_path(
            get_value("registry_path", str(defaults.registry_path)),
            base_directory,
            "registry_path",
        )
        attendance_path = _resolve_path(
            get_value("attendance_path", str(defaults.attendance_path)),
            base_directory,
            "attendance_path",
        )
        return cls(
            camera_index=_parse_integer(
                get_value("camera_index", defaults.camera_index), "camera_index", minimum=0
            ),
            registry_path=registry_path,
            attendance_path=attendance_path,
            tolerance=_parse_number(
                get_value("tolerance", defaults.tolerance),
                "tolerance",
                minimum=0.01,
                maximum=1.0,
            ),
            require_liveness=_parse_boolean(
                get_value("require_liveness", defaults.require_liveness), "require_liveness"
            ),
            max_embeddings_per_user=_parse_integer(
                get_value("max_embeddings_per_user", defaults.max_embeddings_per_user),
                "max_embeddings_per_user",
                minimum=1,
                maximum=100,
            ),
            window_width=_parse_integer(
                get_value("window_width", defaults.window_width), "window_width", minimum=800
            ),
            window_height=_parse_integer(
                get_value("window_height", defaults.window_height), "window_height", minimum=600
            ),
            storage_backend=_parse_choice(
                get_value("storage_backend", defaults.storage_backend),
                "storage_backend",
                STORAGE_BACKENDS,
            ),
            postgres_dsn=_parse_optional_string(
                get_value("postgres_dsn", defaults.postgres_dsn), "postgres_dsn"
            ),
        )


def _resolve_path(value: Any, base_directory: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_directory / path


def _validate_storage_backend(value: str) -> str:
    """Return the normalized backend name, or raise if unsupported.

    Args:
        value: The configured backend name.

    Returns:
        The lowercased, stripped backend name.

    Raises:
        ConfigurationError: If the name is not a supported backend.
    """
    if not isinstance(value, str):
        raise ConfigurationError("storage_backend must be a string")
    normalized = value.strip().lower()
    if normalized not in STORAGE_BACKENDS:
        supported = ", ".join(STORAGE_BACKENDS)
        raise ConfigurationError(f"storage_backend must be one of: {supported}")
    return normalized


def _parse_choice(value: Any, name: str, allowed: tuple[str, ...]) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in allowed:
            return normalized
    supported = ", ".join(allowed)
    raise ConfigurationError(f"{name} must be one of: {supported}")


def _parse_optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ConfigurationError(f"{name} must not be empty when set")
        return stripped
    raise ConfigurationError(f"{name} must be a string")


def _parse_boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigurationError(f"{name} must be a boolean")


def _parse_integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ConfigurationError(f"{name} must be an integer")
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value.strip(), 10)
        except ValueError as exc:
            raise ConfigurationError(f"{name} must be an integer") from exc
    else:
        raise ConfigurationError(f"{name} must be an integer")
    if parsed < minimum or (maximum is not None and parsed > maximum):
        bounds = (
            f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        )
        raise ConfigurationError(f"{name} must be {bounds}")
    return parsed


def _parse_number(value: Any, name: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ConfigurationError(f"{name} must be a finite number")
    if parsed < minimum or (maximum is not None and parsed > maximum):
        bounds = (
            f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        )
        raise ConfigurationError(f"{name} must be {bounds}")
    return parsed


def _paths_refer_to_same_file(first: Path, second: Path) -> bool:
    first_path = first.expanduser().resolve()
    second_path = second.expanduser().resolve()
    if first_path == second_path:
        return True
    try:
        return (
            first_path.exists()
            and second_path.exists()
            and os.path.samefile(first_path, second_path)
        )
    except OSError:
        return False


def _validate_integer(name: str, value: int, minimum: int, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = (
            f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        )
        raise ConfigurationError(f"{name} must be {bounds}")


def _validate_number(name: str, value: float, minimum: float, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric_value):
        raise ConfigurationError(f"{name} must be a finite number")
    if numeric_value < minimum or (maximum is not None and numeric_value > maximum):
        bounds = (
            f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        )
        raise ConfigurationError(f"{name} must be {bounds}")
