"""Storage factory.

Resolves the configured storage backend name to a concrete implementation, so
callers never import a backend class or branch on a string themselves.

Supported values for ``embeddings`` and ``attendance``:

``json`` / ``csv``
    The dependency-free defaults: :class:`~face_attendance.registry.FaceRegistry`
    and :class:`~face_attendance.attendance.AttendanceLog`.
``sqlite``
    :class:`~face_attendance.storage.sqlite.SqliteEmbeddingStore` and
    :class:`~face_attendance.storage.sqlite.SqliteAttendanceStore`, backed by
    the standard library.
``postgres``
    :class:`~face_attendance.storage.postgresql.PostgresEmbeddingStore` and
    :class:`~face_attendance.storage.postgresql.PostgresAttendanceStore`.
    Requires the ``psycopg`` package.

Every store implements :class:`~face_attendance.protocols.EmbeddingStore` or
:class:`~face_attendance.protocols.AttendanceStore`, so swapping backends does
not change any caller's types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..attendance import AttendanceLog
from ..config import STORAGE_BACKENDS, AppConfig
from ..crypto import PlaintextCipher, build_cipher
from ..errors import ConfigurationError
from ..protocols import AttendanceStore, EmbeddingStore
from ..registry import FaceRegistry
from .postgresql import PostgresAttendanceStore, PostgresEmbeddingStore
from .sqlite import SqliteAttendanceStore, SqliteEmbeddingStore

__all__ = [
    "STORAGE_BACKENDS",
    "StorageFactory",
    "create_attendance_store",
    "create_embedding_store",
    "resolve_factory",
]


class StorageFactory(ABC):
    """Creates stores for a single deployment.

    Implementations resolve one backend name to concrete classes, which keeps
    the ``if backend == "sqlite"`` branching in exactly one place.
    """

    @abstractmethod
    def create_embedding_store(
        self,
        path: str | Path,
        max_embeddings_per_user: int = 5,
        config: AppConfig | None = None,
    ) -> EmbeddingStore:
        """Create the embedding store for this deployment.

        Args:
            path: Backend-specific location. A file path for ``json`` and
                ``sqlite``, or a libpq DSN for ``postgres``.
            max_embeddings_per_user: Maximum samples retained per identity.

        Returns:
            A store satisfying :class:`~face_attendance.protocols.EmbeddingStore`.
        """

    @abstractmethod
    def create_attendance_store(self, path: str | Path) -> AttendanceStore:
        """Create the attendance store for this deployment.

        Args:
            path: Backend-specific location. A file path for ``csv`` and
                ``sqlite``, or a libpq DSN for ``postgres``.

        Returns:
            A store satisfying
            :class:`~face_attendance.protocols.AttendanceStore`.
        """


class _JsonFactory(StorageFactory):
    """Serves the file-backed defaults used by the bundled config.

    The cipher only applies to this backend: SQLite and PostgreSQL store
    embeddings in their own column types, so confidentiality there depends on
    the database's own encryption at rest.
    """

    def create_embedding_store(
        self,
        path: str | Path,
        max_embeddings_per_user: int = 5,
        config: AppConfig | None = None,
    ) -> EmbeddingStore:
        cipher = build_cipher(config.registry_cipher) if config is not None else PlaintextCipher()
        return FaceRegistry(path, max_embeddings_per_user=max_embeddings_per_user, cipher=cipher)

    def create_attendance_store(self, path: str | Path) -> AttendanceStore:
        return AttendanceLog(path)


class _SqliteFactory(StorageFactory):
    """Serves the standard-library SQLite backends."""

    def create_embedding_store(
        self,
        path: str | Path,
        max_embeddings_per_user: int = 5,
        config: AppConfig | None = None,
    ) -> EmbeddingStore:
        return SqliteEmbeddingStore(path, max_embeddings_per_user=max_embeddings_per_user)

    def create_attendance_store(self, path: str | Path) -> AttendanceStore:
        return SqliteAttendanceStore(path)


class _PostgresFactory(StorageFactory):
    """Serves the PostgreSQL backends, which require the ``psycopg`` driver."""

    def create_embedding_store(
        self,
        path: str | Path,
        max_embeddings_per_user: int = 5,
        config: AppConfig | None = None,
    ) -> EmbeddingStore:
        return PostgresEmbeddingStore(str(path), max_embeddings_per_user=max_embeddings_per_user)

    def create_attendance_store(self, path: str | Path) -> AttendanceStore:
        return PostgresAttendanceStore(str(path))


_FACTORIES: dict[str, type[StorageFactory]] = {
    "json": _JsonFactory,
    "csv": _JsonFactory,
    "sqlite": _SqliteFactory,
    "postgres": _PostgresFactory,
}


def resolve_factory(backend: str) -> StorageFactory:
    """Return the factory serving a backend name.

    Args:
        backend: One of :data:`STORAGE_BACKENDS`, matched case-insensitively.

    Returns:
        The matching :class:`StorageFactory`.

    Raises:
        ConfigurationError: If ``backend`` is not a supported backend name.
    """
    key = backend.strip().lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        supported = ", ".join(sorted(_FACTORIES))
        raise ConfigurationError(
            f"Unknown storage backend {backend!r}. Expected one of: {supported}"
        )
    return factory()


def _dsn_for(config: AppConfig) -> str:
    """Return the configured PostgreSQL DSN.

    Args:
        config: Application configuration.

    Returns:
        The libpq connection string.

    Raises:
        ConfigurationError: If no DSN was configured.
    """
    if not config.postgres_dsn:
        raise ConfigurationError("postgres_dsn is required when storage_backend is 'postgres'")
    return config.postgres_dsn


def create_embedding_store(
    config: AppConfig,
    path: str | Path | None = None,
    max_embeddings_per_user: int = 5,
) -> EmbeddingStore:
    """Build the embedding store named by ``config``.

    Args:
        config: Application configuration naming the backend.
        path: File path for ``json`` and ``sqlite``. Ignored for ``postgres``,
            which uses ``config.postgres_dsn``. Defaults to
            ``config.registry_path`` for file backends.
        max_embeddings_per_user: Maximum samples retained per identity.

    Returns:
        A store satisfying :class:`~face_attendance.protocols.EmbeddingStore`.
    """
    if config.storage_backend == "postgres":
        return PostgresEmbeddingStore(
            _dsn_for(config), max_embeddings_per_user=max_embeddings_per_user
        )
    target = config.registry_path if path is None else path
    return resolve_factory(config.storage_backend).create_embedding_store(
        target, max_embeddings_per_user, config
    )


def create_attendance_store(
    config: AppConfig,
    path: str | Path | None = None,
) -> AttendanceStore:
    """Build the attendance store named by ``config``.

    Args:
        config: Application configuration naming the backend.
        path: File path for ``csv`` and ``sqlite``. Ignored for ``postgres``,
            which uses ``config.postgres_dsn``. Defaults to
            ``config.attendance_path`` for file backends.

    Returns:
        A store satisfying :class:`~face_attendance.protocols.AttendanceStore`.
    """
    if config.storage_backend == "postgres":
        return PostgresAttendanceStore(_dsn_for(config))
    target = config.attendance_path if path is None else path
    return resolve_factory(config.storage_backend).create_attendance_store(target)
