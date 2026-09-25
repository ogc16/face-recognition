"""SQLite backend failure-path tests.

The happy paths and the shared policy live in
:mod:`tests.test_storage_policy`. These tests cover what happens when the
database cannot be opened, is not a database, or holds a corrupt payload.

The file-backed stores convert both :class:`OSError` and database errors into
the domain error types, and the SQLite backend must do the same; these tests
pin that contract down, because a raw ``sqlite3`` or ``FileExistsError``
leaking to a caller breaks the documented error handling.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from face_attendance.errors import AttendanceError, RegistryError
from face_attendance.storage import SqliteAttendanceStore, SqliteEmbeddingStore

ALPHA = (0.1, 0.2, 0.3)


def test_ensure_exists_creates_the_database(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "faces.db"
    SqliteEmbeddingStore(path).ensure_exists()
    assert path.exists()


def test_ensure_exists_is_idempotent(tmp_path: Path) -> None:
    store = SqliteEmbeddingStore(tmp_path / "faces.db")
    store.ensure_exists()
    store.ensure_exists()
    assert store.register("Ada", ALPHA) is True


def test_ensure_exists_creates_missing_parent_directories(tmp_path: Path) -> None:
    SqliteAttendanceStore(tmp_path / "a" / "b" / "log.db").ensure_exists()
    assert (tmp_path / "a" / "b" / "log.db").exists()


def test_a_blocked_directory_becomes_a_registry_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    store = SqliteEmbeddingStore(blocker / "sub" / "faces.db")
    with pytest.raises(RegistryError):
        store.ensure_exists()


def test_a_blocked_directory_becomes_an_attendance_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    store = SqliteAttendanceStore(blocker / "sub" / "log.db")
    with pytest.raises(AttendanceError):
        store.ensure_exists()


def test_a_corrupt_database_becomes_a_registry_error(tmp_path: Path) -> None:
    path = tmp_path / "faces.db"
    path.write_bytes(b"this is definitely not a sqlite database")
    store = SqliteEmbeddingStore(path)
    with pytest.raises(RegistryError):
        store.get("Ada")
    with pytest.raises(RegistryError):
        store.register("Ada", ALPHA)
    with pytest.raises(RegistryError):
        store.remove("Ada")
    with pytest.raises(RegistryError):
        store.names()


def test_a_corrupt_database_becomes_an_attendance_error(tmp_path: Path) -> None:
    path = tmp_path / "log.db"
    path.write_bytes(b"this is definitely not a sqlite database")
    store = SqliteAttendanceStore(path)
    with pytest.raises(AttendanceError):
        store.events()
    with pytest.raises(AttendanceError):
        store.record("Ada", "in")
    with pytest.raises(AttendanceError):
        store.ensure_exists()


def test_a_corrupt_embedding_payload_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "faces.db"
    SqliteEmbeddingStore(path).ensure_exists()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO users (name_key, name, embeddings) VALUES (?, ?, ?)",
            ("ada", "Ada", "{not json"),
        )
    with pytest.raises(RegistryError) as error:
        SqliteEmbeddingStore(path).get("Ada")
    assert "corrupt" in str(error.value)


def test_a_non_list_embedding_payload_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "faces.db"
    SqliteEmbeddingStore(path).ensure_exists()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO users (name_key, name, embeddings) VALUES (?, ?, ?)",
            ("ada", "Ada", '{"not": "a list"}'),
        )
    with pytest.raises(RegistryError) as error:
        SqliteEmbeddingStore(path).get("Ada")
    assert "corrupt" in str(error.value)


def test_an_empty_user_table_reads_as_no_users(tmp_path: Path) -> None:
    store = SqliteEmbeddingStore(tmp_path / "faces.db")
    store.ensure_exists()
    assert store.names() == ()
    assert store.records() == ()
    assert store.get("Nobody") is None
    assert len(store) == 0
    assert list(store.iter_embeddings()) == []


def test_an_empty_attendance_table_reads_as_no_events(tmp_path: Path) -> None:
    store = SqliteAttendanceStore(tmp_path / "log.db")
    store.ensure_exists()
    assert store.events() == ()
    assert store.latest_action("Nobody") is None


def test_data_survives_a_new_store_instance(tmp_path: Path) -> None:
    path = tmp_path / "faces.db"
    SqliteEmbeddingStore(path).register("Ada", ALPHA)
    record = SqliteEmbeddingStore(path).get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA,)


def test_data_survives_a_new_attendance_instance(tmp_path: Path) -> None:
    path = tmp_path / "log.db"
    SqliteAttendanceStore(path).record("Ada", "in")
    assert SqliteAttendanceStore(path).latest_action("Ada") == "in"


def test_an_invalid_sample_cap_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RegistryError):
        SqliteEmbeddingStore(tmp_path / "faces.db", max_embeddings_per_user=0)
    with pytest.raises(RegistryError):
        SqliteEmbeddingStore(tmp_path / "faces.db", max_embeddings_per_user=True)
    with pytest.raises(RegistryError):
        SqliteEmbeddingStore(tmp_path / "faces.db", max_embeddings_per_user="3")


def test_removing_the_last_user_clears_the_table(tmp_path: Path) -> None:
    store = SqliteEmbeddingStore(tmp_path / "faces.db")
    store.register("Ada", ALPHA)
    assert store.remove("Ada") is True
    assert store.records() == ()
    assert SqliteEmbeddingStore(tmp_path / "faces.db").names() == ()


def test_the_store_recreates_a_dropped_table(tmp_path: Path) -> None:
    path = tmp_path / "faces.db"
    store = SqliteEmbeddingStore(path)
    store.register("Ada", ALPHA)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE users")
    assert store.register("Ada", (0.4, 0.5, 0.6)) is True
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT name FROM users").fetchall()
    assert [str(row[0]) for row in rows] == ["Ada"]


def test_a_failed_write_leaves_the_previous_data_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "faces.db"
    store = SqliteEmbeddingStore(path)
    store.register("Ada", ALPHA)

    original = SqliteEmbeddingStore._connect

    @contextmanager
    def closed_database(store_self: SqliteEmbeddingStore) -> Iterator[sqlite3.Connection]:
        with original(store_self) as connection:
            connection.close()
            yield connection

    monkeypatch.setattr(SqliteEmbeddingStore, "_connect", closed_database)
    with pytest.raises(RegistryError):
        store.register("Ada", (0.4, 0.5, 0.6))
    monkeypatch.undo()

    record = SqliteEmbeddingStore(path).get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA,)


def test_wal_mode_is_enabled(tmp_path: Path) -> None:
    path = tmp_path / "faces.db"
    store = SqliteEmbeddingStore(path)
    store.ensure_exists()
    store.register("Ada", ALPHA)
    with sqlite3.connect(path) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
