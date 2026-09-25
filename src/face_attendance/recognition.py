import importlib
import math
from collections.abc import Callable

from .errors import (
    DependencyError,
    LivenessError,
    RecognitionError,
    RegistrationError,
    RegistryError,
)
from .liveness import LivenessStatus
from .protocols import (
    EmbeddingStore,
    FaceDistance,
    FaceEncoder,
    Frame,
    LivenessPolicyProtocol,
)
from .registry import Embedding
from .types import RecognitionResult, RecognitionStatus
from .validation import validate_embedding

__all__ = [
    "DefaultFaceRecognitionBackend",
    "Embedding",
    "FaceDistance",
    "FaceEncoder",
    "FaceRecognitionBackend",
    "FaceRecognitionService",
    "RecognitionResult",
    "RecognitionStatus",
]


class FaceRecognitionBackend:
    """Marker base class for pluggable face encoders.

    Subclasses override :meth:`encode` and :meth:`distance`. This class exists
    only for documentation value; consumers should depend on
    :class:`~face_attendance.protocols.FaceEncoder` instead.
    """

    def encode(self, frame: Frame) -> list[Embedding]:
        """Return one embedding per face detected in the frame.

        Args:
            frame: RGB frame to analyze.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError

    def distance(self, known: Embedding, candidate: Embedding) -> float:
        """Return a non-negative distance between two embeddings.

        Args:
            known: The enrolled embedding.
            candidate: The observed embedding.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
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

    def encode(self, frame: Frame) -> list[Embedding]:
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
        registry: EmbeddingStore,
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

    def _encode(self, frame: Frame) -> list[Embedding]:
        try:
            encodings = self.encoder.encode(frame)
            return [validate_embedding(embedding) for embedding in encodings]
        except (RegistryError, TypeError, ValueError) as exc:
            raise RecognitionError("Face encoder returned an invalid embedding") from exc

    def recognize(self, frame: Frame) -> RecognitionResult:
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

    def check_liveness(self, frame: Frame, liveness_policy: LivenessPolicyProtocol) -> None:
        """Enforce the liveness policy for a frame.

        Args:
            frame: Frame about to be matched.
            liveness_policy: Policy governing the liveness requirement. Any
                object exposing ``evaluate(frame)`` is accepted, so callers may
                substitute their own policy.

        Raises:
            LivenessError: If the policy is missing its ``evaluate`` method, if
                the verdict is not allowed, or if it is a spoof.
        """
        evaluate = getattr(liveness_policy, "evaluate", None)
        if not callable(evaluate):
            raise LivenessError("A liveness policy is required")
        liveness = evaluate(frame)
        if not liveness.allowed:
            if liveness.status is LivenessStatus.SPOOF:
                raise LivenessError("Liveness verification rejected this face")
            raise LivenessError("A configured liveness checker is required")

    def authenticate(
        self,
        frame: Frame,
        liveness_policy: LivenessPolicyProtocol,
    ) -> RecognitionResult:
        self.check_liveness(frame, liveness_policy)
        return self.recognize(frame)

    def embedding_for(self, frame: Frame) -> Embedding:
        encodings = self._encode(frame)
        if len(encodings) == 0:
            raise RegistrationError("No face was detected in the captured image")
        if len(encodings) > 1:
            raise RegistrationError("Registration requires exactly one face")
        return encodings[0]
