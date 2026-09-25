from dataclasses import dataclass

from .config import AppConfig
from .liveness import LivenessChecker, LivenessPolicy
from .protocols import AttendanceStore, EmbeddingStore
from .recognition import DefaultFaceRecognitionBackend, FaceRecognitionService
from .storage import create_attendance_store, create_embedding_store


@dataclass(frozen=True, slots=True)
class Runtime:
    """Everything the application needs, assembled from the configuration.

    Attributes:
        config: The configuration the runtime was built from.
        registry: The embedding store, which may be JSON, SQLite, or
            PostgreSQL depending on ``config.storage_backend``.
        attendance: The attendance store, which may be CSV, SQLite, or
            PostgreSQL.
        service: The recognition service bound to the embedding store.
        liveness_policy: The policy applied before any match is accepted.
    """

    config: AppConfig
    registry: EmbeddingStore
    attendance: AttendanceStore
    service: FaceRecognitionService
    liveness_policy: LivenessPolicy


def build_runtime(
    config: AppConfig,
    liveness_checker: LivenessChecker | None = None,
) -> Runtime:
    """Assemble the application from a configuration.

    The storage backends are chosen by
    :func:`~face_attendance.storage.create_embedding_store` and
    :func:`~face_attendance.storage.create_attendance_store`, so a different
    backend requires only a configuration change rather than a code change.

    Args:
        config: The application configuration.
        liveness_checker: Optional liveness implementation. When ``None`` and
            ``config.require_liveness`` is set, the policy fails closed.

    Returns:
        The assembled :class:`Runtime`.
    """
    backend = DefaultFaceRecognitionBackend()
    registry = create_embedding_store(
        config,
        max_embeddings_per_user=config.max_embeddings_per_user,
    )
    registry.ensure_exists()
    attendance = create_attendance_store(config)
    attendance.ensure_exists()
    service = FaceRecognitionService(
        registry=registry,
        encoder=backend,
        distance=backend.distance,
        tolerance=config.tolerance,
    )
    return Runtime(
        config=config,
        registry=registry,
        attendance=attendance,
        service=service,
        liveness_policy=LivenessPolicy(
            checker=liveness_checker,
            required=config.require_liveness,
        ),
    )
