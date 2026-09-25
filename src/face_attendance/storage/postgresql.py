"""PostgreSQL storage backend.

PostgreSQL support is optional: this module imports cleanly without
``psycopg`` installed, and raises a clear dependency error only when a store is
actually constructed. The driver is loaded lazily so that the dependency-free
core keeps working.

Install the extra with::

    pip install "face-attendance[postgres]"

Embedding vectors are stored as JSONB rather than a ``pgvector`` column, which
keeps the schema portable and avoids an extension dependency. Similarity search
for a deployment with many identities should move the matching step into
PostgreSQL (``ORDER BY embeddings <=> %s`` with the ``pgvector`` extension) and
load the result into the recognizer's cache.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, cast

from ..errors import AttendanceError, RegistryError
from ..types import AttendanceAction, AttendanceEvent, Embedding, UserRecord
from ..validation import name_key
from .base import AttendanceStoreBase, EmbeddingStoreBase

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["PostgresAttendanceStore", "PostgresEmbeddingStore"]

_EMBEDDING_DDL = """
CREATE TABLE IF NOT EXISTS face_users (
    name_key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embeddings JSONB NOT NULL
)
"""

_ATTENDANCE_DDL = """
CREATE TABLE IF NOT EXISTS face_attendance_log (
    sequence BIGSERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('in', 'out'))
)
"""


class _Cursor(Protocol):
    """The subset of a DB-API cursor this backend relies on."""

    def execute(self, operation: str, parameters: object = ...) -> object: ...

    def fetchall(self) -> list[tuple[Any, ...]]: ...

    def __enter__(self) -> _Cursor: ...

    def __exit__(self, *exc_info: object) -> object: ...


class _Connection(Protocol):
    """The subset of a DB-API connection this backend relies on."""

    def cursor(self) -> _Cursor: ...

    def commit(self) -> object: ...

    def rollback(self) -> object: ...

    def close(self) -> object: ...


class _Driver(Protocol):
    """The subset of the ``psycopg`` module this backend relies on."""

    def connect(self, dsn: str) -> _Connection: ...


class PostgresConnectionMixin:
    """Shared connection handling for the PostgreSQL stores.

    Attributes:
        dsn: The libpq connection string or URI.
        error_type: The domain error raised for driver and connection faults.
        label: A human-readable name for the store, used in error messages.
    """

    dsn: str
    error_type: type[Exception]
    label: str

    def _load_driver(self) -> _Driver:
        """Import ``psycopg`` on demand.

        Returns:
            The imported driver module.

        Raises:
            Exception: ``self.error_type``, when ``psycopg`` is not installed.
        """
        try:
            module: ModuleType = importlib.import_module("psycopg")
        except ImportError as exc:
            raise self.error_type(
                "PostgreSQL storage requires the 'psycopg' package. "
                'Install it with: pip install "face-attendance[postgres]"'
            ) from exc
        return cast(_Driver, module)

    @contextmanager
    def _operation(self) -> Iterator[_Connection]:
        """Yield a connection, mapping driver faults to a domain error.

        Every call path goes through here so that a database outage surfaces
        as :class:`~face_attendance.errors.RegistryError` or
        :class:`~face_attendance.errors.AttendanceError` rather than a raw
        DB-API exception, matching what the file-backed stores raise.

        The DSN is deliberately omitted from the message because it commonly
        carries a password.

        Yields:
            An open connection. The transaction is committed when the body
            returns and rolled back when it raises.
        """
        try:
            with self._connect() as connection:
                yield connection
        except self.error_type:
            raise
        except Exception as exc:
            raise self.error_type(f"Unable to access the {self.label}") from exc

    @contextmanager
    def _connect(self) -> Iterator[_Connection]:
        """Yield a committed-or-rolled-back connection.

        Yields:
            An open connection. The transaction is committed when the body
            returns and rolled back when it raises.
        """
        connection = self._load_driver().connect(self.dsn)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class PostgresEmbeddingStore(EmbeddingStoreBase, PostgresConnectionMixin):
    """An :class:`~face_attendance.protocols.EmbeddingStore` backed by PostgreSQL.

    Args:
        dsn: libpq connection string, e.g.
            ``postgresql://user:pass@localhost/faces``.
        max_embeddings_per_user: Maximum samples retained per identity.

    Raises:
        RegistryError: If ``max_embeddings_per_user`` is not a positive
            integer.
    """

    def __init__(self, dsn: str, max_embeddings_per_user: int = 5) -> None:
        if isinstance(max_embeddings_per_user, bool) or not isinstance(
            max_embeddings_per_user, int
        ):
            raise RegistryError("Maximum embeddings per user must be an integer")
        if max_embeddings_per_user < 1:
            raise RegistryError("At least one embedding per user is required")
        self.dsn = dsn
        self.max_embeddings_per_user = max_embeddings_per_user
        self.error_type = RegistryError
        self.label = "embedding store"

    def ensure_exists(self) -> None:
        """Create the ``face_users`` table if absent.

        Raises:
            RegistryError: If the driver is missing or the DDL fails.
        """
        with self._operation() as connection, connection.cursor() as cursor:
            cursor.execute(_EMBEDDING_DDL)

    def _read_records(self) -> dict[str, UserRecord]:
        with self._operation() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT name_key, name, embeddings FROM face_users")
            rows = cursor.fetchall()
        records: dict[str, UserRecord] = {}
        for key, name, payload in rows:
            records[str(key)] = UserRecord(name=str(name), embeddings=self._decode(payload))
        return records

    def _write_records(self, records: dict[str, UserRecord]) -> None:
        with self._operation() as connection, connection.cursor() as cursor:
            cursor.execute(_EMBEDDING_DDL)
            cursor.execute("DELETE FROM face_users")
            for key, record in records.items():
                cursor.execute(
                    "INSERT INTO face_users (name_key, name, embeddings) "
                    "VALUES (%s, %s, %s::jsonb)",
                    (key, record.name, self._encode(record.embeddings)),
                )

    @staticmethod
    def _encode(embeddings: tuple[Embedding, ...]) -> str:
        return json.dumps([list(embedding) for embedding in embeddings])

    @staticmethod
    def _decode(payload: object) -> tuple[Embedding, ...]:
        if isinstance(payload, (bytes, bytearray, str)):
            try:
                decoded = json.loads(payload)
            except ValueError as exc:
                raise RegistryError("Stored embeddings are corrupt") from exc
        else:
            decoded = payload
        if not isinstance(decoded, list):
            raise RegistryError("Stored embeddings are corrupt")
        return tuple(tuple(float(value) for value in item) for item in decoded)


class PostgresAttendanceStore(AttendanceStoreBase, PostgresConnectionMixin):
    """An :class:`~face_attendance.protocols.AttendanceStore` backed by PostgreSQL.

    Args:
        dsn: libpq connection string, e.g.
            ``postgresql://user:pass@localhost/faces``.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.error_type = AttendanceError
        self.label = "attendance store"

    def ensure_exists(self) -> None:
        """Create the ``face_attendance_log`` table if absent.

        Raises:
            AttendanceError: If the driver is missing or the DDL fails.
        """
        with self._operation() as connection, connection.cursor() as cursor:
            cursor.execute(_ATTENDANCE_DDL)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS face_attendance_name_key "
                "ON face_attendance_log (name_key)"
            )

    def _read_events(self) -> list[AttendanceEvent]:
        with self._operation() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT timestamp, name, action FROM face_attendance_log ORDER BY sequence"
            )
            rows = cursor.fetchall()
        return [
            AttendanceEvent(
                timestamp=str(timestamp),
                name=str(name),
                action=cast(AttendanceAction, str(action)),
            )
            for timestamp, name, action in rows
        ]

    def _append_event(self, event: AttendanceEvent) -> None:
        with self._operation() as connection, connection.cursor() as cursor:
            cursor.execute(_ATTENDANCE_DDL)
            cursor.execute(
                "INSERT INTO face_attendance_log "
                "(timestamp, name, name_key, action) VALUES (%s, %s, %s, %s)",
                (event.timestamp, event.name, name_key(event.name), event.action),
            )
