import pytest

from face_attendance.errors import LivenessError
from face_attendance.liveness import LivenessPolicy, LivenessStatus


class Checker:
    def __init__(self, result):
        self.result = result

    def is_live(self, frame):
        return self.result


def test_missing_checker_is_allowed_when_optional():
    result = LivenessPolicy(required=False).evaluate(object())

    assert result.status is LivenessStatus.UNAVAILABLE
    assert result.allowed is True


def test_missing_checker_fails_closed_when_required():
    result = LivenessPolicy(required=True).evaluate(object())

    assert result.allowed is False


def test_spoof_is_rejected():
    result = LivenessPolicy(Checker(False), required=False).evaluate(object())

    assert result.status is LivenessStatus.SPOOF
    assert result.allowed is False


def test_checker_errors_are_wrapped():
    class BrokenChecker:
        def is_live(self, frame):
            raise RuntimeError("model unavailable")

    with pytest.raises(LivenessError):
        LivenessPolicy(BrokenChecker(), required=False).evaluate(object())
