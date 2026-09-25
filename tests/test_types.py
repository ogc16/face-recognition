import numpy as np
import pytest

from face_attendance.types import (
    VALID_ACTIONS,
    AttendanceAction,
    AttendanceEvent,
    Embedding,
    Frame,
    LivenessResult,
    LivenessStatus,
    RecognitionResult,
    RecognitionStatus,
    UserRecord,
    UserRecordDict,
)


def test_frame_alias_describes_an_rgb_array():
    frame: Frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert frame.shape == (4, 4, 3)
    assert frame.dtype == np.uint8


def test_embedding_alias_is_a_float_tuple():
    embedding: Embedding = (0.1, 0.2)

    assert embedding == (0.1, 0.2)


def test_valid_actions_contains_both_transitions():
    assert frozenset({"in", "out"}) == VALID_ACTIONS
    assert set(VALID_ACTIONS) == {"in", "out"}


def test_attendance_action_literal_is_importable():
    action: AttendanceAction = "in"

    assert action == "in"


def test_user_record_to_json_round_trips():
    record = UserRecord(name="Ada", embeddings=((0.1, 0.2), (0.3, 0.4)))

    payload = record.to_json()

    assert payload == {
        "name": "Ada",
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
    }
    assert isinstance(payload, dict)


def test_user_record_is_immutable_and_hashable():
    record = UserRecord(name="Ada", embeddings=((0.1,),))

    with pytest.raises(AttributeError):
        record.name = "Grace"

    assert hash(record) == hash(UserRecord(name="Ada", embeddings=((0.1,),)))


def test_user_record_dict_alias_is_a_mapping():
    payload: UserRecordDict = {"name": "Ada", "embeddings": [[0.1]]}

    assert payload["name"] == "Ada"


def test_attendance_event_to_row_exposes_csv_fields():
    event = AttendanceEvent(timestamp="2026-09-25T09:00:00+00:00", name="Ada", action="in")

    assert event.to_row() == {
        "timestamp": "2026-09-25T09:00:00+00:00",
        "name": "Ada",
        "action": "in",
    }


def test_attendance_event_does_not_validate_the_action_at_runtime():
    event = AttendanceEvent(
        timestamp="2026-09-25T09:00:00+00:00",
        name="Ada",
        action="sideways",
    )

    assert event.action == "sideways"


def test_the_store_rejects_an_invalid_action(tmp_path):
    from face_attendance.attendance import AttendanceLog
    from face_attendance.errors import AttendanceError

    log = AttendanceLog(tmp_path / "attendance.csv")
    log.ensure_exists()

    with pytest.raises(AttendanceError):
        log.record("Ada", "sideways")


def test_attendance_event_is_immutable():
    event = AttendanceEvent(timestamp="2026-09-25T09:00:00+00:00", name="Ada", action="in")

    with pytest.raises(AttributeError):
        event.name = "Grace"


def test_liveness_result_allows_live_regardless_of_requirement():
    assert LivenessResult(LivenessStatus.LIVE, True).allowed is True
    assert LivenessResult(LivenessStatus.LIVE, False).allowed is True


def test_liveness_result_rejects_spoof():
    assert LivenessResult(LivenessStatus.SPOOF, False).allowed is False


def test_liveness_result_rejects_unavailable_when_required():
    assert LivenessResult(LivenessStatus.UNAVAILABLE, True).allowed is False


def test_liveness_result_allows_unavailable_when_not_required():
    assert LivenessResult(LivenessStatus.UNAVAILABLE, False).allowed is True


def test_liveness_result_is_immutable():
    result = LivenessResult(LivenessStatus.LIVE, True)

    with pytest.raises(AttributeError):
        result.required = False


def test_liveness_status_compares_as_a_string():
    assert LivenessStatus.LIVE == "live"
    assert RecognitionStatus.MATCH == "match"


def test_recognition_result_reports_a_match():
    result = RecognitionResult(
        status=RecognitionStatus.MATCH,
        face_count=1,
        name="Ada",
        distance=0.1,
        confidence=0.9,
    )

    assert result.matched is True
    assert result.match_quality == 0.9


def test_recognition_result_without_a_name_is_not_a_match():
    result = RecognitionResult(status=RecognitionStatus.MATCH, face_count=1)

    assert result.matched is False


@pytest.mark.parametrize(
    ("status", "face_count"),
    [
        (RecognitionStatus.UNKNOWN, 1),
        (RecognitionStatus.NO_FACE, 0),
        (RecognitionStatus.MULTIPLE_FACES, 3),
    ],
)
def test_non_match_statuses_report_matched_false(status, face_count):
    result = RecognitionResult(status=status, face_count=face_count)

    assert result.matched is False
    assert result.match_quality is None
