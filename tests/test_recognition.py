import math

import pytest

from face_attendance.errors import (
    DependencyError,
    LivenessError,
    RecognitionError,
    RegistrationError,
)
from face_attendance.liveness import LivenessPolicy
from face_attendance.recognition import (
    FaceRecognitionService,
    RecognitionStatus,
)
from face_attendance.registry import FaceRegistry


class FakeEncoder:
    def __init__(self, *faces):
        self.faces = list(faces)

    def encode(self, frame):
        return self.faces


def distance(known, candidate):
    return math.dist(known, candidate)


def service_for(tmp_path, faces, tolerance=0.6):
    registry = FaceRegistry(tmp_path / "registry.json")
    registry.register("Ada", [0.0, 0.0])
    return FaceRecognitionService(registry, FakeEncoder(*faces), distance, tolerance)


def test_recognition_matches_the_closest_registered_face(tmp_path):
    service = service_for(tmp_path, [(0.01, 0.01)])

    result = service.recognize(object())

    assert result.status is RecognitionStatus.MATCH
    assert result.name == "Ada"
    assert result.matched is True
    assert result.distance is not None
    assert result.confidence is not None


def test_recognition_reports_unknown_faces(tmp_path):
    service = service_for(tmp_path, [(1.0, 1.0)])

    result = service.recognize(object())

    assert result.status is RecognitionStatus.UNKNOWN
    assert result.matched is False


def test_recognition_reports_no_face(tmp_path):
    service = service_for(tmp_path, [])

    result = service.recognize(object())

    assert result.status is RecognitionStatus.NO_FACE
    assert result.face_count == 0


def test_recognition_rejects_multiple_faces(tmp_path):
    service = service_for(tmp_path, [(0.0, 0.0), (0.01, 0.01)])

    result = service.recognize(object())

    assert result.status is RecognitionStatus.MULTIPLE_FACES
    assert result.face_count == 2


def test_registration_requires_exactly_one_face(tmp_path):
    service = service_for(tmp_path, [(0.0, 0.0), (0.01, 0.01)])

    with pytest.raises(RegistrationError):
        service.embedding_for(object())


def test_empty_registry_is_unknown(tmp_path):
    registry = FaceRegistry(tmp_path / "registry.json")
    service = FaceRecognitionService(
        registry,
        FakeEncoder((0.0, 0.0)),
        distance,
        tolerance=0.6,
    )

    assert service.recognize(object()).status is RecognitionStatus.UNKNOWN


def test_authentication_fails_closed_when_required_liveness_is_unavailable(tmp_path):
    service = service_for(tmp_path, [(0.01, 0.01)])

    with pytest.raises(LivenessError):
        service.authenticate(object(), LivenessPolicy(required=True))


def test_registration_path_fails_closed_without_required_liveness(tmp_path):
    service = service_for(tmp_path, [(0.01, 0.01)])

    with pytest.raises(LivenessError):
        service.check_liveness(object(), LivenessPolicy(required=True))


def test_authentication_allows_optional_liveness_unavailable(tmp_path):
    service = service_for(tmp_path, [(0.01, 0.01)])

    result = service.authenticate(object(), LivenessPolicy(required=False))

    assert result.matched is True


def test_incompatible_known_embedding_is_ignored(tmp_path):
    service = service_for(tmp_path, [(0.0, 0.0, 0.0)])

    result = service.recognize(object())

    assert result.status is RecognitionStatus.UNKNOWN


def test_recognition_loads_registry_once_per_request(tmp_path, monkeypatch):
    registry = FaceRegistry(tmp_path / "registry.json")
    registry.register("Ada", [0.0, 0.0])
    reloaded_registry = FaceRegistry(registry.path)
    service = FaceRecognitionService(
        reloaded_registry,
        FakeEncoder((0.01, 0.01)),
        distance,
        tolerance=0.6,
    )
    original = reloaded_registry._read
    calls = 0

    def counted_read():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(reloaded_registry, "_read", counted_read)
    result = service.recognize(object())

    assert result.matched is True
    assert calls == 1


def test_match_quality_is_exposed_without_changing_distance(tmp_path):
    service = service_for(tmp_path, [(0.01, 0.01)])

    result = service.recognize(object())

    assert result.match_quality == result.confidence
    assert result.distance is not None


def test_service_rejects_invalid_tolerance_type(tmp_path):
    registry = FaceRegistry(tmp_path / "registry.json")
    with pytest.raises(RecognitionError):
        FaceRecognitionService(registry, FakeEncoder(), distance, tolerance="0.6")


def test_service_rejects_non_boolean_single_face_setting(tmp_path):
    registry = FaceRegistry(tmp_path / "registry.json")
    with pytest.raises(RecognitionError):
        FaceRecognitionService(
            registry,
            FakeEncoder(),
            distance,
            tolerance=0.6,
            require_single_face="yes",
        )


def test_service_rejects_invalid_encoder_output(tmp_path):
    registry = FaceRegistry(tmp_path / "registry.json")
    service = FaceRecognitionService(
        registry,
        FakeEncoder((float("nan"),)),
        distance,
        tolerance=0.6,
    )

    with pytest.raises(RecognitionError):
        service.recognize(object())


def test_service_requires_a_liveness_policy_object(tmp_path):
    service = service_for(tmp_path, [(0.01, 0.01)])

    with pytest.raises(LivenessError):
        service.check_liveness(object(), object())


def test_default_backend_reports_missing_model_data(monkeypatch):
    from face_attendance import recognition

    def missing_model(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(recognition.importlib, "import_module", missing_model)

    with pytest.raises(DependencyError, match="model data"):
        recognition.DefaultFaceRecognitionBackend()
