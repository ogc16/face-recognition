"""Storage backends for embeddings and attendance.

Each backend implements only *persistence*; the registration policy and the
sign-in/sign-out state machine live once in :mod:`face_attendance.storage.base`
so that every backend enforces identical rules.

Backends are chosen through the abstract factory in
:mod:`face_attendance.storage.factory`, which resolves the configured backend
name to a concrete implementation.
"""

from __future__ import annotations

from .base import AttendanceStoreBase, EmbeddingStoreBase
from .factory import (
    STORAGE_BACKENDS,
    StorageFactory,
    create_attendance_store,
    create_embedding_store,
    resolve_factory,
)
from .postgresql import PostgresAttendanceStore, PostgresEmbeddingStore
from .sqlite import SqliteAttendanceStore, SqliteEmbeddingStore

__all__ = [
    "STORAGE_BACKENDS",
    "AttendanceStoreBase",
    "EmbeddingStoreBase",
    "PostgresAttendanceStore",
    "PostgresEmbeddingStore",
    "SqliteAttendanceStore",
    "SqliteEmbeddingStore",
    "StorageFactory",
    "create_attendance_store",
    "create_embedding_store",
    "resolve_factory",
]
