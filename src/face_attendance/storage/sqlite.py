"""SQLite storage backend.

Uses the standard library's :mod:`sqlite3` module, so it adds no dependency.
This is the recommended backend for single-machine deployments that want
atomic writes and concurrent access beyond what a JSON file offers.

The database is opened per operation with a short-lived connection and WAL
journaling, which avoids the "cannot start a transaction within a transaction"
problem that arises from sharing one connection across threads.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import cast

from ..errors import AttendanceError, RegistryError
from ..types import AttendanceAction, AttendanceEvent, Embedding, UserRecord
from ..validation import name_key
from .base import AttendanceStoreBase, EmbeddingStoreBase

__all__ = ["SqliteAttendanceStore", "SqliteEmbeddingStore"]

_EMBEDDING_SQL = """
CREATE TABLE IF NOT EXISTS users (
    name_key TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    embeddings TEXT NOT NULL
)
"""

_ATTENDANCE_SQL = """
CREATE TABLE IF NOT EXISTS attendance (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    action TEXT NOT NULL
)
"""


class SqliteConnectionMixin:
    """Shared connection handling for the SQLite stores.

    Attributes:
        path: Filesystem location of the database file.
    """

    path: Path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection, closing it on exit.

        Yields:
            An open :class:`sqlite3.Connection` with foreign keys enabled.

        Raises:
            OSError: If the database cannot be opened.
        """
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        try:
            connection.row_factory = sqlite3.Row
            yield connection
        finally:
            connection.close()


class SqliteEmbeddingStore(EmbeddingStoreBase, SqliteConnectionMixin):
    """An :class:`~face_attendance.protocols.EmbeddingStore` backed by SQLite.

    Args:
        path: Path to the SQLite database file.
        max_embeddings_per_user: Maximum samples retained per identity.

    Raises:
        RegistryError: If ``max_embeddings_per_user`` is not a positive
            integer.
    """

    def __init__(self, path: str | Path, max_embeddings_per_user: int = 5) -> None:
        if isinstance(max_embeddings_per_user, bool) or not isinstance(
            max_embeddings_per_user, int
        ):
            raise RegistryError("Maximum embeddings per user must be an integer")
        if max_embeddings_per_user < 1:
            raise RegistryError("At least one embedding per user is required")
        self.path = Path(path)
        self.max_embeddings_per_user = max_embeddings_per_user
        self._lock = RLock()

    def ensure_exists(self) -> None:
        """Create the database file and ``users`` table if absent.

        Raises:
            RegistryError: If the database cannot be created.
        """
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self._connect() as connection:
                    self._ensure_schema(connection)
            except (sqlite3.Error, OSError) as exc:
                raise RegistryError(f"Unable to access registry: {self.path}") from exc

    def _read_records(self) -> dict[str, UserRecord]:
        with self._lock:
            try:
                with self._connect() as connection:
                    self._ensure_schema(connection)
                    rows = connection.execute(
                        "SELECT name_key, name, embeddings FROM users"
                    ).fetchall()
            except (sqlite3.Error, OSError) as exc:
                raise RegistryError(f"Unable to access registry: {self.path}") from exc
        records: dict[str, UserRecord] = {}
        for row in rows:
            records[str(row["name_key"])] = UserRecord(
                name=str(row["name"]),
                embeddings=self._decode(row["embeddings"]),
            )
        return records

    def _write_records(self, records: dict[str, UserRecord]) -> None:
        with self._lock:
            try:
                with self._connect() as connection:
                    self._ensure_schema(connection)
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("DELETE FROM users")
                    connection.executemany(
                        "INSERT INTO users (name_key, name, embeddings) VALUES (?, ?, ?)",
                        [
                            (key, record.name, self._encode(record.embeddings))
                            for key, record in records.items()
                        ],
                    )
                    connection.commit()
            except (sqlite3.Error, OSError) as exc:
                raise RegistryError(f"Unable to access registry: {self.path}") from exc

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        """Create the ``users`` table when absent.

        Called on every read and write so a freshly constructed store behaves
        like the file-backed stores, which create their file on first use.

        Args:
            connection: An open connection.
        """
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(_EMBEDDING_SQL)
        connection.commit()

    @staticmethod
    def _encode(embeddings: tuple[Embedding, ...]) -> str:
        return json.dumps([list(embedding) for embedding in embeddings])

    @staticmethod
    def _decode(payload: object) -> tuple[Embedding, ...]:
        try:
            decoded = json.loads(str(payload))
        except (TypeError, ValueError) as exc:
            raise RegistryError("Stored embeddings are corrupt") from exc
        if not isinstance(decoded, list):
            raise RegistryError("Stored embeddings are corrupt")
        return tuple(tuple(float(value) for value in item) for item in decoded)


class SqliteAttendanceStore(AttendanceStoreBase, SqliteConnectionMixin):
    """An :class:`~face_attendance.protocols.AttendanceStore` backed by SQLite.

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def ensure_exists(self) -> None:
        """Create the database file and ``attendance`` table if absent.

        Raises:
            AttendanceError: If the database cannot be created.
        """
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self._connect() as connection:
                    self._ensure_schema(connection)
            except (sqlite3.Error, OSError) as exc:
                raise AttendanceError(f"Unable to access attendance log: {self.path}") from exc

    def _read_events(self) -> list[AttendanceEvent]:
        with self._lock:
            try:
                with self._connect() as connection:
                    self._ensure_schema(connection)
                    rows = connection.execute(
                        "SELECT timestamp, name, action FROM attendance ORDER BY sequence"
                    ).fetchall()
            except (sqlite3.Error, OSError) as exc:
                raise AttendanceError(f"Unable to access attendance log: {self.path}") from exc
        return [
            AttendanceEvent(
                timestamp=str(row["timestamp"]),
                name=str(row["name"]),
                action=cast(AttendanceAction, str(row["action"])),
            )
            for row in rows
        ]

    def _append_event(self, event: AttendanceEvent) -> None:
        with self._lock:
            try:
                with self._connect() as connection:
                    self._ensure_schema(connection)
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "INSERT INTO attendance (timestamp, name, name_key, action) "
                        "VALUES (?, ?, ?, ?)",
                        (event.timestamp, event.name, name_key(event.name), event.action),
                    )
                    connection.commit()
            except (sqlite3.Error, OSError) as exc:
                raise AttendanceError(f"Unable to access attendance log: {self.path}") from exc

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        """Create the ``attendance`` table when absent.

        Invoked on every read and write so a freshly constructed store behaves
        like the file-backed stores, which create their file on first use.

        Args:
            connection: An open connection.
        """
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(_ATTENDANCE_SQL)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS attendance_name_key ON attendance (name_key)"
        )
        connection.commit()
