import pytest

from face_attendance.attendance import AttendanceLog
from face_attendance.camera import FixedFrameSource
from face_attendance.liveness import CallableLivenessChecker, LivenessPolicy
from face_attendance.protocols import (
    AttendanceStore,
    EmbeddingStore,
    FaceDistance,
    FaceEncoder,
    FrameSource,
    LivenessChecker,
    LivenessPolicyProtocol,
    RecognitionService,
    UserRecordProtocol,
)
from face_attendance.recognition import FaceRecognitionService
from face_attendance.registry import FaceRegistry
from face_attendance.types import UserRecord


class StubEncoder:
    def encode(self, frame):
        return [(0.1, 0.2)]


def encoder_distance(known, candidate):
    return 0.0


@pytest.fixture
def registry(tmp_path):
    store = FaceRegistry(tmp_path / "registry.json")
    store.ensure_exists()
    return store


@pytest.fixture
def attendance_log(tmp_path):
    log = AttendanceLog(tmp_path / "attendance.csv")
    log.ensure_exists()
    return log


def test_face_registry_satisfies_the_embedding_store_contract(registry):
    assert isinstance(registry, EmbeddingStore)


def test_attendance_log_satisfies_the_attendance_store_contract(attendance_log):
    assert isinstance(attendance_log, AttendanceStore)


def test_user_record_satisfies_the_structural_record_contract():
    record = UserRecord(name="Ada", embeddings=((0.1,),))

    assert isinstance(record, UserRecordProtocol)


def test_a_plain_object_does_not_satisfy_the_record_contract():
    assert not isinstance(object(), UserRecordProtocol)


def test_callable_liveness_checker_satisfies_the_contract():
    assert isinstance(CallableLivenessChecker(lambda frame: True), LivenessChecker)


def test_liveness_policy_satisfies_the_policy_contract():
    assert isinstance(LivenessPolicy(required=False), LivenessPolicyProtocol)


def test_recognition_service_satisfies_the_service_contract(registry):
    service = FaceRecognitionService(
        registry=registry,
        encoder=StubEncoder(),
        distance=encoder_distance,
        tolerance=0.5,
    )

    assert isinstance(service, RecognitionService)


def test_default_backend_satisfies_the_encoder_contract():
    from face_attendance.recognition import DefaultFaceRecognitionBackend

    assert isinstance(DefaultFaceRecognitionBackend, FaceEncoder)


def test_backend_distance_method_satisfies_the_distance_contract():
    from face_attendance.recognition import DefaultFaceRecognitionBackend

    backend = DefaultFaceRecognitionBackend.__new__(DefaultFaceRecognitionBackend)

    assert isinstance(backend.distance, FaceDistance)


def test_a_fixed_frame_source_satisfies_the_frame_source_contract():
    assert isinstance(FixedFrameSource(("a",)), FrameSource)


def test_a_plain_object_does_not_satisfy_the_frame_source_contract():
    assert not isinstance(object(), FrameSource)


def test_a_plain_object_does_not_satisfy_the_embedding_store_contract():
    assert not isinstance(object(), EmbeddingStore)


def test_a_plain_object_does_not_satisfy_the_attendance_store_contract():
    assert not isinstance(object(), AttendanceStore)


def test_a_plain_object_does_not_satisfy_the_face_encoder_contract():
    assert not isinstance(object(), FaceEncoder)


def test_a_plain_object_does_not_satisfy_the_liveness_checker_contract():
    assert not isinstance(object(), LivenessChecker)


def test_a_plain_object_does_not_satisfy_the_recognition_service_contract():
    assert not isinstance(object(), RecognitionService)


def test_a_plain_object_does_not_satisfy_the_face_distance_contract():
    assert not isinstance(object(), FaceDistance)
