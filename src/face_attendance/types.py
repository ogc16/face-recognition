"""Shared value types used across storage backends and interface contracts.

These are deliberately dependency-light so that both the concrete storage
implementations and the :mod:`~face_attendance.protocols` contracts can import
them without creating a cycle. The owning modules re-export these names, so
``face_attendance.registry.UserRecord`` and
``face_attendance.attendance.AttendanceEvent`` remain valid import paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    #: A captured camera frame as an 8-bit, three-channel RGB image array.
    #:
    #: Frames cross several third-party boundaries (OpenCV, Pillow, dlib).
    #: Those boundaries are the only place where a value changes
    #: representation, and they are annotated with :func:`typing.cast` rather
    #: than widened to :data:`typing.Any`.
    Frame: TypeAlias = NDArray[np.uint8]
else:
    Frame: TypeAlias = object

__all__ = [
    "VALID_ACTIONS",
    "AttendanceAction",
    "AttendanceEvent",
    "Embedding",
    "Frame",
    "LivenessResult",
    "LivenessStatus",
    "RecognitionResult",
    "RecognitionStatus",
    "UserRecord",
    "UserRecordDict",
]

#: A face embedding: an immutable tuple of float values.
#:
#: The bundled model produces 128-dimensional vectors, but the storage layer
#: deliberately stays dimension-agnostic so alternative encoders can be
#: substituted.
Embedding: TypeAlias = tuple[float, ...]

#: The two attendance transitions the application permits.
AttendanceAction: TypeAlias = Literal["in", "out"]

#: The set of valid attendance actions.
VALID_ACTIONS: frozenset[str] = frozenset({"in", "out"})


@dataclass(frozen=True, slots=True)
class UserRecord:
    """A registered identity and all of its stored face samples.

    Attributes:
        name: The normalized display name of the identity.
        embeddings: Every sample enrolled for this identity. Implementations
            guarantee a non-empty tuple of equal-length embeddings.
    """

    name: str
    embeddings: tuple[Embedding, ...]

    def to_json(self) -> UserRecordDict:
        """Return a JSON-serializable representation of this record.

        Returns:
            A mapping with a ``name`` string and an ``embeddings`` list of
            float lists.
        """
        return {
            "name": self.name,
            "embeddings": [list(embedding) for embedding in self.embeddings],
        }


#: The JSON object shape produced by :meth:`UserRecord.to_json`.
UserRecordDict: TypeAlias = dict[str, object]


@dataclass(frozen=True, slots=True)
class AttendanceEvent:
    """A single attendance transition.

    Attributes:
        timestamp: ISO 8601 event time, normalized to UTC and timezone-aware.
        name: The normalized display name of the identity.
        action: Either ``"in"`` or ``"out"``.
    """

    timestamp: str
    name: str
    action: AttendanceAction

    def to_row(self) -> dict[str, str]:
        """Return the CSV field mapping for this event.

        Returns:
            A mapping with ``timestamp``, ``name``, and ``action`` keys.
        """
        return {"timestamp": self.timestamp, "name": self.name, "action": self.action}


class LivenessStatus(str, Enum):
    """The outcome of a liveness evaluation."""

    LIVE = "live"
    SPOOF = "spoof"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class LivenessResult:
    """A liveness verdict together with the policy that produced it.

    Attributes:
        status: Whether the frame was live, spoofed, or not evaluable.
        required: Whether liveness was mandatory. A non-required
            :attr:`LivenessStatus.UNAVAILABLE` verdict still allows the frame
            through, which is how the policy fails open when the operator has
            explicitly opted out of liveness.
    """

    status: LivenessStatus
    required: bool

    @property
    def allowed(self) -> bool:
        """Return whether the frame may proceed to face matching."""
        return self.status is LivenessStatus.LIVE or (
            not self.required and self.status is LivenessStatus.UNAVAILABLE
        )


class RecognitionStatus(str, Enum):
    """The outcome of a recognition attempt."""

    MATCH = "match"
    UNKNOWN = "unknown"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """The best match found for a frame.

    Attributes:
        status: Whether a match, no face, multiple faces, or an unknown face.
        face_count: How many faces were detected in the frame.
        name: The matched identity, or ``None`` when there is no match.
        distance: Embedding distance to the closest enrolled sample, or
            ``None`` when no distance was computed.
        confidence: Normalized closeness in ``[0, 1]``, or ``None`` when no
            distance was computed.
    """

    status: RecognitionStatus
    face_count: int
    name: str | None = None
    distance: float | None = None
    confidence: float | None = None

    @property
    def matched(self) -> bool:
        """Return whether an enrolled identity was matched."""
        return self.status is RecognitionStatus.MATCH and self.name is not None

    @property
    def match_quality(self) -> float | None:
        """Return the confidence, or ``None`` when no distance was computed."""
        return self.confidence
