"""Storage factory and storage configuration tests.

The factory is the only place in the codebase that branches on a backend name,
so these tests cover name resolution, the config surface that feeds it, and the
guarantee that every backend is substitutable behind the shared protocols.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from face_attendance.attendance import AttendanceLog
from face_attendance.config import STORAGE_BACKENDS, AppConfig
from face_attendance.errors import ConfigurationError
from face_attendance.protocols import AttendanceStore, EmbeddingStore
from face_attendance.registry import FaceRegistry
from face_attendance.storage import (
    PostgresAttendanceStore,
    PostgresEmbeddingStore,
    SqliteAttendanceStore,
    SqliteEmbeddingStore,
    StorageFactory,
    create_attendance_store,
    create_embedding_store,
    resolve_factory,
)


def test_every_advertised_backend_resolves() -> None:
    for backend in STORAGE_BACKENDS:
        assert isinstance(resolve_factory(backend), StorageFactory)


def test_backend_names_are_case_insensitive() -> None:
    assert isinstance(resolve_factory("SQLite"), StorageFactory)
    assert isinstance(resolve_factory("  Postgres  "), StorageFactory)


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as error:
        resolve_factory("mysql")
    assert "sqlite" in str(error.value)


def test_the_default_backend_is_json() -> None:
    assert AppConfig().storage_backend == "json"


def test_json_serves_the_bundled_file_stores(tmp_path: Path) -> None:
    config = AppConfig()
    assert isinstance(create_embedding_store(config, tmp_path / "r.json"), FaceRegistry)
    assert isinstance(create_attendance_store(config, tmp_path / "a.csv"), AttendanceLog)


def test_csv_serves_the_bundled_file_stores(tmp_path: Path) -> None:
    config = AppConfig(storage_backend="csv")
    assert isinstance(create_embedding_store(config, tmp_path / "r.json"), FaceRegistry)
    assert isinstance(create_attendance_store(config, tmp_path / "a.csv"), AttendanceLog)


def test_sqlite_serves_the_sqlite_stores(tmp_path: Path) -> None:
    config = AppConfig(storage_backend="sqlite")
    embeddings = create_embedding_store(config, tmp_path / "faces.db")
    attendance = create_attendance_store(config, tmp_path / "faces.db")
    assert isinstance(embeddings, SqliteEmbeddingStore)
    assert isinstance(attendance, SqliteAttendanceStore)


def test_postgres_serves_the_postgres_stores() -> None:
    config = AppConfig(storage_backend="postgres", postgres_dsn="postgresql://h/faces")
    assert isinstance(create_embedding_store(config), PostgresEmbeddingStore)
    assert isinstance(create_attendance_store(config), PostgresAttendanceStore)


def test_postgres_ignores_the_file_path() -> None:
    config = AppConfig(
        storage_backend="postgres",
        postgres_dsn="postgresql://h/faces",
        registry_path=Path("ignored/registry.json"),
    )
    store = create_embedding_store(config)
    assert isinstance(store, PostgresEmbeddingStore)
    assert store.dsn == "postgresql://h/faces"


def test_the_factory_returns_protocol_conforming_stores(tmp_path: Path) -> None:
    for backend in ("json", "csv", "sqlite"):
        config = AppConfig(storage_backend=backend)
        assert isinstance(create_embedding_store(config, tmp_path / "s"), EmbeddingStore)
        assert isinstance(create_attendance_store(config, tmp_path / "s"), AttendanceStore)
    config = AppConfig(storage_backend="postgres", postgres_dsn="postgresql://h/f")
    assert isinstance(create_embedding_store(config), EmbeddingStore)
    assert isinstance(create_attendance_store(config), AttendanceStore)


def test_paths_default_to_the_configured_ones(tmp_path: Path) -> None:
    config = AppConfig(
        storage_backend="sqlite",
        registry_path=tmp_path / "custom-registry.db",
        attendance_path=tmp_path / "custom-attendance.db",
    )
    assert create_embedding_store(config).path == tmp_path / "custom-registry.db"
    assert create_attendance_store(config).path == tmp_path / "custom-attendance.db"


def test_the_sample_cap_is_forwarded(tmp_path: Path) -> None:
    config = AppConfig(storage_backend="sqlite")
    store = create_embedding_store(config, tmp_path / "r.db", max_embeddings_per_user=2)
    assert isinstance(store, SqliteEmbeddingStore)
    assert store.max_embeddings_per_user == 2


def test_an_unknown_backend_is_rejected_by_the_config() -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(storage_backend="mysql")


def test_postgres_requires_a_dsn() -> None:
    with pytest.raises(ConfigurationError) as error:
        AppConfig(storage_backend="postgres")
    assert "postgres_dsn" in str(error.value)


def test_the_backend_can_come_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"storage_backend": "sqlite", "max_embeddings_per_user": 3}',
        encoding="utf-8",
    )
    config = AppConfig.from_file(path, environ={})
    assert config.storage_backend == "sqlite"
    assert config.max_embeddings_per_user == 3


def test_the_dsn_can_come_from_the_environment() -> None:
    config = AppConfig.from_file(
        None,
        environ={
            "FACE_ATTENDANCE_STORAGE_BACKEND": "postgres",
            "FACE_ATTENDANCE_POSTGRES_DSN": "postgresql://h/faces",
        },
    )
    assert config.storage_backend == "postgres"
    assert config.postgres_dsn == "postgresql://h/faces"


def test_the_environment_overrides_the_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"storage_backend": "sqlite"}', encoding="utf-8")
    config = AppConfig.from_file(path, environ={"FACE_ATTENDANCE_STORAGE_BACKEND": "json"})
    assert config.storage_backend == "json"


def test_an_empty_dsn_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        AppConfig.from_file(None, environ={"FACE_ATTENDANCE_POSTGRES_DSN": "   "})


def test_an_invalid_backend_name_is_rejected_from_the_environment() -> None:
    with pytest.raises(ConfigurationError):
        AppConfig.from_file(None, environ={"FACE_ATTENDANCE_STORAGE_BACKEND": "mysql"})


def test_a_factory_serves_both_store_kinds(tmp_path: Path) -> None:
    factory = resolve_factory("sqlite")
    assert isinstance(factory.create_embedding_store(tmp_path / "a.db"), EmbeddingStore)
    assert isinstance(factory.create_attendance_store(tmp_path / "a.db"), AttendanceStore)
