from dataclasses import dataclass

from .attendance import AttendanceLog
from .config import AppConfig
from .liveness import LivenessChecker, LivenessPolicy
from .recognition import DefaultFaceRecognitionBackend, FaceRecognitionService
from .registry import FaceRegistry


@dataclass(frozen=True, slots=True)
class Runtime:
    config: AppConfig
    registry: FaceRegistry
    attendance: AttendanceLog
    service: FaceRecognitionService
    liveness_policy: LivenessPolicy


def build_runtime(
    config: AppConfig,
    liveness_checker: LivenessChecker | None = None,
) -> Runtime:
    backend = DefaultFaceRecognitionBackend()
    registry = FaceRegistry(
        config.registry_path,
        max_embeddings_per_user=config.max_embeddings_per_user,
    )
    registry.ensure_exists()
    attendance = AttendanceLog(config.attendance_path)
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
