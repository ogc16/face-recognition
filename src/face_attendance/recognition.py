import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .errors import (
    DependencyError,
    LivenessError,
    RecognitionError,
    RegistrationError,
    RegistryError,
)
from .liveness import LivenessPolicy, LivenessStatus
from .registry import Embedding, FaceRegistry
from .validation import validate_embedding


class RecognitionStatus(str, Enum):
    MATCH = "match"
    UNKNOWN = "unknown"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    status: RecognitionStatus
    face_count: int
    name: str | None = None
    distance: float | None = None
    confidence: float | None = None

    @property
    def matched(self) -> bool:
        return self.status is RecognitionStatus.MATCH and self.name is not None

    @property
    def match_quality(self) -> float | None:
        return self.confidence


class FaceEncoder(Protocol):
    def encode(self, frame: Any) -> list[Embedding]: ...


class FaceDistance(Protocol):
    def __call__(self, known: Embedding, candidate: Embedding) -> float: ...


class FaceRecognitionBackend:
    def encode(self, frame: Any) -> list[Embedding]:
        raise NotImplementedError

    def distance(self, known: Embedding, candidate: Embedding) -> float:
        raise NotImplementedError


class DefaultFaceRecognitionBackend:
    def __init__(self) -> None:
        try:
            importlib.import_module("face_recognition_models")
        except (Exception, SystemExit) as exc:
            raise DependencyError(
                "Face recognition model data is unavailable. Install the "
                "`face_recognition_models` package and its `setuptools` runtime before "
                "using recognition."
            ) from exc
        try:
            self._face_recognition = importlib.import_module("face_recognition")
            self._numpy = importlib.import_module("numpy")
        except (Exception, SystemExit) as exc:
            raise DependencyError(
                "Face recognition dependencies are unavailable. Install the project with "
                "`python -m pip install -e .`."
            ) from exc

    def encode(self, frame: Any) -> list[Embedding]:
        try:
            encodings = self._face_recognition.face_encodings(frame)
            return [validate_embedding(encoding) for encoding in encodings]
        except (RecognitionError, RegistryError, TypeError, ValueError) as exc:
            raise RecognitionError(f"Unable to encode the captured image: {exc}") from exc
        except Exception as exc:
            raise RecognitionError(
                "The face-recognition model could not process the image"
            ) from exc

    def distance(self, known: Embedding, candidate: Embedding) -> float:
        try:
            known_array = self._numpy.asarray([known], dtype=float)
            candidate_array = self._numpy.asarray(candidate, dtype=float)
            value = self._face_recognition.face_distance(known_array, candidate_array)[0]
            return float(value)
        except Exception as exc:
            raise RecognitionError("Unable to compare face embeddings") from exc


class FaceRecognitionService:
    def __init__(
        self,
        registry: FaceRegistry,
        encoder: FaceEncoder,
        distance: FaceDistance | Callable[[Embedding, Embedding], float],
        tolerance: float,
        require_single_face: bool = True,
    ) -> None:
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise RecognitionError("Tolerance must be a number")
        try:
            numeric_tolerance = float(tolerance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RecognitionError("Tolerance must be a finite number") from exc
        if not math.isfinite(numeric_tolerance) or not 0 < numeric_tolerance <= 1:
            raise RecognitionError("Tolerance must be a finite value greater than 0 and at most 1")
        if not isinstance(require_single_face, bool):
            raise RecognitionError("Single-face requirement must be a boolean")
        self.registry = registry
        self.encoder = encoder
        self.distance = distance
        self.tolerance = numeric_tolerance
        self.require_single_face = require_single_face

    def _encode(self, frame: Any) -> list[Embedding]:
        try:
            encodings = self.encoder.encode(frame)
            return [validate_embedding(embedding) for embedding in encodings]
        except (RegistryError, TypeError, ValueError) as exc:
            raise RecognitionError("Face encoder returned an invalid embedding") from exc

    def recognize(self, frame: Any) -> RecognitionResult:
        encodings = self._encode(frame)
        face_count = len(encodings)
        if face_count == 0:
            return RecognitionResult(status=RecognitionStatus.NO_FACE, face_count=0)
        if self.require_single_face and face_count > 1:
            return RecognitionResult(status=RecognitionStatus.MULTIPLE_FACES, face_count=face_count)
        dimensions = {len(encoding) for encoding in encodings}
        if len(dimensions) != 1:
            raise RecognitionError("Face encoder returned inconsistent embedding dimensions")
        records = self.registry.records()
        if not records:
            return RecognitionResult(status=RecognitionStatus.UNKNOWN, face_count=face_count)

        best_name: str | None = None
        best_distance = math.inf
        for record in records:
            for known_embedding in record.embeddings:
                for candidate in encodings:
                    if len(known_embedding) != len(candidate):
                        continue
                    try:
                        distance = float(self.distance(known_embedding, candidate))
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise RecognitionError("Face distance is invalid") from exc
                    if not math.isfinite(distance):
                        raise RecognitionError("Face distance is not finite")
                    if distance < best_distance:
                        best_name = record.name
                        best_distance = distance

        if best_name is None or best_distance > self.tolerance:
            return RecognitionResult(
                status=RecognitionStatus.UNKNOWN,
                face_count=face_count,
                distance=best_distance if math.isfinite(best_distance) else None,
            )

        confidence = max(0.0, min(1.0, 1.0 - best_distance / (self.tolerance * 2)))
        return RecognitionResult(
            status=RecognitionStatus.MATCH,
            face_count=face_count,
            name=best_name,
            distance=best_distance,
            confidence=confidence,
        )

    def check_liveness(self, frame: Any, liveness_policy: LivenessPolicy) -> None:
        if not isinstance(liveness_policy, LivenessPolicy):
            raise LivenessError("A liveness policy is required")
        liveness = liveness_policy.evaluate(frame)
        if not liveness.allowed:
            if liveness.status is LivenessStatus.SPOOF:
                raise LivenessError("Liveness verification rejected this face")
            raise LivenessError("A configured liveness checker is required")

    def authenticate(self, frame: Any, liveness_policy: LivenessPolicy) -> RecognitionResult:
        self.check_liveness(frame, liveness_policy)
        return self.recognize(frame)

    def embedding_for(self, frame: Any) -> Embedding:
        encodings = self._encode(frame)
        if len(encodings) == 0:
            raise RegistrationError("No face was detected in the captured image")
        if len(encodings) > 1:
            raise RegistrationError("Registration requires exactly one face")
        return encodings[0]
