"""Backend-independent storage policy.

:class:`EmbeddingStoreBase` and :class:`AttendanceStoreBase` implement every
rule the application cares about -- duplicate detection, the per-user sample
cap, cross-user sample conflicts, and the sign-in/sign-out state machine -- in
one place. A concrete backend supplies only the two or three persistence
primitives, so no backend can accidentally implement a different policy.

The two ABCs here are the *implementation* base classes. Consumers should
depend on :class:`~face_attendance.protocols.EmbeddingStore` and
:class:`~face_attendance.protocols.AttendanceStore`, which are structural
:class:`~typing.Protocol` types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from typing import cast

from ..errors import AttendanceError, RegistryError
from ..types import VALID_ACTIONS, AttendanceAction, AttendanceEvent, Embedding, UserRecord
from ..validation import name_key, normalize_name, validate_embedding

__all__ = ["AttendanceStoreBase", "EmbeddingStoreBase"]


class EmbeddingStoreBase(ABC):
    """Registration policy shared by every embedding store.

    Attributes:
        max_embeddings_per_user: The maximum number of samples retained for a
            single identity.
    """

    max_embeddings_per_user: int

    @abstractmethod
    def ensure_exists(self) -> None:
        """Create the backing storage if it does not already exist."""

    @abstractmethod
    def _read_records(self) -> dict[str, UserRecord]:
        """Return every user record, keyed by case-folded name."""

    @abstractmethod
    def _write_records(self, records: dict[str, UserRecord]) -> None:
        """Replace the entire stored set with ``records``."""

    def register(self, name: str, embedding: Embedding | Sequence[float]) -> bool:
        """Store a face sample for a user.

        Args:
            name: Human-readable identity. Normalized before storage.
            embedding: Face embedding to persist.

        Returns:
            ``True`` when a new sample was written. ``False`` when the sample
            was already present for this user, or the per-user limit was
            reached.

        Raises:
            RegistryError: If the name or embedding is invalid, or the sample
                already belongs to a different identity.
        """
        normalized_name = normalize_name(name)
        normalized_embedding = validate_embedding(embedding)
        records = self._read_records()
        key = name_key(normalized_name)
        existing = records.get(key)
        if existing is not None and normalized_embedding in existing.embeddings:
            return False
        if any(
            normalized_embedding in record.embeddings
            for other_key, record in records.items()
            if other_key != key
        ):
            raise RegistryError("Face sample is already registered to another user")
        if existing is None:
            records[key] = UserRecord(name=normalized_name, embeddings=(normalized_embedding,))
        elif len(existing.embeddings) < self.max_embeddings_per_user:
            records[key] = UserRecord(
                name=existing.name,
                embeddings=(*existing.embeddings, normalized_embedding),
            )
        else:
            return False
        self._write_records(records)
        return True

    def remove(self, name: str) -> bool:
        """Delete every stored sample for a user.

        Args:
            name: Human-readable identity. Normalized before lookup.

        Returns:
            ``True`` when a user was removed, ``False`` when none existed.
        """
        normalized_name = normalize_name(name)
        records = self._read_records()
        key = name_key(normalized_name)
        if key not in records:
            return False
        del records[key]
        self._write_records(records)
        return True

    def get(self, name: str) -> UserRecord | None:
        """Return the record for a user, or ``None`` when absent.

        Args:
            name: Human-readable identity. Normalized before lookup.
        """
        return self._read_records().get(name_key(normalize_name(name)))

    def names(self) -> tuple[str, ...]:
        """Return every registered name, sorted case-insensitively."""
        records = self._read_records()
        return tuple(sorted((record.name for record in records.values()), key=name_key))

    def records(self) -> tuple[UserRecord, ...]:
        """Return every user record, ordered by case-folded name."""
        records = self._read_records()
        return tuple(records[key] for key in sorted(records))

    def iter_embeddings(self) -> Iterator[tuple[str, Embedding]]:
        """Yield each ``(name, embedding)`` pair exactly once."""
        for record in self.records():
            for embedding in record.embeddings:
                yield record.name, embedding

    def __len__(self) -> int:
        """Return the number of registered users."""
        return len(self._read_records())


class AttendanceStoreBase(ABC):
    """Sign-in/sign-out policy shared by every attendance store."""

    @abstractmethod
    def ensure_exists(self) -> None:
        """Create the backing storage and any required header or schema."""

    @abstractmethod
    def _read_events(self) -> list[AttendanceEvent]:
        """Return every stored event in append order."""

    @abstractmethod
    def _append_event(self, event: AttendanceEvent) -> None:
        """Persist one new event."""

    def record(
        self,
        name: str,
        action: str,
        timestamp: datetime | None = None,
    ) -> AttendanceEvent:
        """Append an attendance event after validating the transition.

        Args:
            name: Human-readable identity. Normalized before storage.
            action: Either ``"in"`` or ``"out"``.
            timestamp: Optional event time. Defaults to the current UTC time.
                Naive values are interpreted as UTC.

        Returns:
            The persisted :class:`~face_attendance.types.AttendanceEvent`.

        Raises:
            AttendanceError: If the action is invalid, the timestamp is not a
                datetime, or the transition is not permitted.
        """
        normalized_name = normalize_name(name)
        if not isinstance(action, str) or action not in VALID_ACTIONS:
            raise AttendanceError("Attendance action must be 'in' or 'out'")
        if timestamp is not None and not isinstance(timestamp, datetime):
            raise AttendanceError("Attendance timestamp must be a datetime")
        event_time = timestamp or datetime.now(timezone.utc)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        event = AttendanceEvent(
            timestamp=event_time.astimezone(timezone.utc).isoformat(),
            name=normalized_name,
            action=cast(AttendanceAction, action),
        )
        previous = self.latest_action(normalized_name)
        if action == "in" and previous == "in":
            raise AttendanceError(f"{normalized_name} is already signed in")
        if action == "out" and previous != "in":
            raise AttendanceError(f"{normalized_name} is not signed in")
        self._append_event(event)
        return event

    def events(self) -> tuple[AttendanceEvent, ...]:
        """Return every recorded event in append order."""
        return tuple(self._read_events())

    def latest_action(self, name: str) -> AttendanceAction | None:
        """Return the user's most recent action.

        Args:
            name: Human-readable identity. Normalized before lookup.

        Returns:
            ``"in"``, ``"out"``, or ``None`` when the user has no events.
        """
        key = name_key(normalize_name(name))
        latest: AttendanceAction | None = None
        for event in self._read_events():
            if name_key(event.name) == key:
                latest = event.action
        return latest
