"""Interface contracts shared across the application.

Every long-lived collaborator is consumed through a :class:`typing.Protocol`
defined here rather than through a concrete class. Protocols are used instead of
abstract base classes so implementations do not need a common ancestor, which
keeps third-party adapters and test doubles trivially substitutable.

============================  ============================================
Contract                      Implementations
============================  ============================================
:class:`EmbeddingStore`       ``FaceRegistry``, ``SqliteEmbeddingStore``,
                              ``PostgresEmbeddingStore``
:class:`AttendanceStore`      ``AttendanceLog``, ``SqliteAttendanceStore``,
                              ``PostgresAttendanceStore``
:class:`FrameSource`          ``OpenCVCamera``, ``VideoFileFrameSource``
:class:`FaceEncoder`          ``DefaultFaceRecognitionBackend``
:class:`FaceDistance`         ``DefaultFaceRecognitionBackend.distance``
:class:`LivenessChecker`      ``CallableLivenessChecker``,
                              ``MotionLivenessChecker``
:class:`RecognitionService`   ``FaceRecognitionService``
============================  ============================================

Shared value types such as :class:`~face_attendance.types.Embedding` and
:class:`~face_attendance.types.UserRecord` are defined in
:mod:`face_attendance.types`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from .types import (
    VALID_ACTIONS,
    AttendanceAction,
    AttendanceEvent,
    Embedding,
    Frame,
    LivenessResult,
    RecognitionResult,
    UserRecord,
)

__all__ = [
    "VALID_ACTIONS",
    "AttendanceAction",
    "AttendanceEvent",
    "AttendanceStore",
    "Embedding",
    "EmbeddingStore",
    "FaceDistance",
    "FaceEncoder",
    "Frame",
    "FrameSource",
    "LivenessChecker",
    "LivenessPolicyProtocol",
    "RecognitionService",
    "UserRecord",
    "UserRecordProtocol",
]


@runtime_checkable
class UserRecordProtocol(Protocol):
    """Structural view of :class:`~face_attendance.types.UserRecord`.

    Consumers that only need to read a record can depend on this instead of the
    concrete dataclass, which keeps storage adapters free to return their own
    row types.
    """

    @property
    def name(self) -> str:
        """Return the normalized display name."""

    @property
    def embeddings(self) -> tuple[Embedding, ...]:
        """Return every stored sample for the identity."""


@runtime_checkable
class EmbeddingStore(Protocol):
    """Durable storage for face embeddings and their owning identities.

    Implementations are responsible for durability, cross-process
    serializability, and validation. Callers rely on :meth:`register` to reject
    malformed input rather than persisting it.
    """

    @property
    def max_embeddings_per_user(self) -> int:
        """Return the maximum number of samples retained per user."""

    def ensure_exists(self) -> None:
        """Create the backing storage if it does not already exist."""

    def register(self, name: str, embedding: Embedding | Sequence[float]) -> bool:
        """Store a face sample for a user.

        Args:
            name: Human-readable identity. Normalized by the implementation.
            embedding: Face embedding to persist.

        Returns:
            ``True`` when a new sample was written. ``False`` when the sample
            was a duplicate or the per-user limit was already reached.

        Raises:
            RegistryError: If the name or embedding is invalid, or if the
                sample is already registered to a different user.
        """

    def remove(self, name: str) -> bool:
        """Delete every stored sample for a user.

        Args:
            name: Human-readable identity. Normalized by the implementation.

        Returns:
            ``True`` when a user was removed, ``False`` when none existed.
        """

    def get(self, name: str) -> UserRecord | None:
        """Return the record for a user, or ``None`` when absent."""

    def names(self) -> tuple[str, ...]:
        """Return every registered name, sorted case-insensitively."""

    def records(self) -> tuple[UserRecord, ...]:
        """Return every user record in a deterministic order."""

    def iter_embeddings(self) -> Iterator[tuple[str, Embedding]]:
        """Yield each ``(name, embedding)`` pair exactly once."""

    def __len__(self) -> int:
        """Return the number of registered users."""


@runtime_checkable
class AttendanceStore(Protocol):
    """Append-only storage for attendance transitions.

    Implementations enforce the sign-in/sign-out state machine: the first event
    for a user must be ``in``, a repeated ``in`` is rejected, and ``out`` is
    accepted only while the user is signed in.
    """

    def ensure_exists(self) -> None:
        """Create the backing storage and any required header or schema."""

    def record(
        self,
        name: str,
        action: str,
        timestamp: datetime | None = None,
    ) -> AttendanceEvent:
        """Append an attendance event.

        Args:
            name: Human-readable identity. Normalized by the implementation.
            action: Either ``"in"`` or ``"out"``.
            timestamp: Optional event time. Defaults to the current UTC time.
                Naive values are interpreted as UTC.

        Returns:
            The persisted :class:`~face_attendance.types.AttendanceEvent`.

        Raises:
            AttendanceError: If the action is invalid or the transition is not
                permitted for the user's current state.
        """

    def events(self) -> tuple[AttendanceEvent, ...]:
        """Return every recorded event in append order."""

    def latest_action(self, name: str) -> AttendanceAction | None:
        """Return the user's most recent action, or ``None`` when unseen."""


@runtime_checkable
class FrameSource(Protocol):
    """A source of camera frames.

    Implementations wrap a physical capture device, a video file, or a synthetic
    generator. Consumers must call :meth:`release` when finished.
    """

    def read(self) -> tuple[bool, Frame | None]:
        """Return the next frame.

        Returns:
            A ``(success, frame)`` pair. ``success`` is ``False`` and ``frame``
            is ``None`` when no frame is currently available, which is a
            transient condition rather than an error.

        Raises:
            CameraError: If the underlying device fails irrecoverably.
        """

    def release(self) -> None:
        """Release the underlying device. Must be idempotent."""


@runtime_checkable
class FaceEncoder(Protocol):
    """Converts camera frames into face embeddings."""

    def encode(self, frame: Frame) -> list[Embedding]:
        """Return one embedding per face detected in the frame.

        Args:
            frame: RGB frame to analyze.

        Returns:
            Zero or more embeddings. An empty list means no face was found.

        Raises:
            RecognitionError: If the frame could not be processed.
        """


@runtime_checkable
class FaceDistance(Protocol):
    """Computes the distance between two face embeddings."""

    def __call__(self, known: Embedding, candidate: Embedding) -> float:
        """Return a non-negative distance, where smaller means more similar.

        Raises:
            RecognitionError: If the embeddings cannot be compared.
        """


@runtime_checkable
class LivenessChecker(Protocol):
    """Decides whether a frame originates from a live subject.

    A checker must return a real :class:`bool`. Returning a NumPy scalar or a
    truthy object is rejected, and the surrounding
    :class:`~face_attendance.liveness.LivenessPolicy` fails closed.
    """

    def is_live(self, frame: Frame) -> bool:
        """Return ``True`` when the frame is judged to be live.

        Raises:
            LivenessError: If the frame could not be evaluated. The policy
                converts this into a rejection.
        """


@runtime_checkable
class LivenessPolicyProtocol(Protocol):
    """Decides whether a frame may proceed to face matching."""

    def evaluate(self, frame: Frame) -> LivenessResult:
        """Evaluate a frame and return its liveness verdict."""


@runtime_checkable
class RecognitionService(Protocol):
    """Matches a frame against the enrolled identities."""

    def recognize(self, frame: Frame) -> RecognitionResult:
        """Return the best match for a frame without checking liveness.

        Args:
            frame: RGB frame to analyze.

        Returns:
            A :class:`~face_attendance.types.RecognitionResult`.

        Raises:
            RecognitionError: If the frame or embeddings are invalid.
        """

    def authenticate(
        self,
        frame: Frame,
        liveness_policy: LivenessPolicyProtocol,
    ) -> RecognitionResult:
        """Check liveness, then recognize.

        Args:
            frame: RGB frame to analyze.
            liveness_policy: Policy governing the liveness requirement.

        Returns:
            A :class:`~face_attendance.types.RecognitionResult`, or a result
            whose status reports the failure.

        Raises:
            LivenessError: If liveness is required and not satisfied.
            RecognitionError: If the frame or embeddings are invalid.
        """

    def embedding_for(self, frame: Frame) -> Embedding:
        """Return the single embedding for a frame, for enrollment.

        Raises:
            RegistrationError: If zero or more than one face was detected.
        """
