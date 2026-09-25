import math

import pytest

from face_attendance.errors import LivenessError, RegistrationError
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


def test_authentication_allows_optional_liveness_unavailable(tmp_path):
    service = service_for(tmp_path, [(0.01, 0.01)])

    result = service.authenticate(object(), LivenessPolicy(required=False))

    assert result.matched is True


def test_incompatible_known_embedding_is_ignored(tmp_path):
    service = service_for(tmp_path, [(0.0, 0.0, 0.0)])

    result = service.recognize(object())

    assert result.status is RecognitionStatus.UNKNOWN
