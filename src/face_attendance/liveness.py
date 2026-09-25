from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .errors import LivenessError


class LivenessStatus(str, Enum):
    LIVE = "live"
    SPOOF = "spoof"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class LivenessResult:
    status: LivenessStatus
    required: bool

    @property
    def allowed(self) -> bool:
        return self.status is LivenessStatus.LIVE or (
            not self.required and self.status is LivenessStatus.UNAVAILABLE
        )


class LivenessChecker(Protocol):
    def is_live(self, frame: Any) -> bool: ...


class CallableLivenessChecker:
    def __init__(self, callback: Callable[[Any], bool]) -> None:
        self._callback = callback

    def is_live(self, frame: Any) -> bool:
        return self._callback(frame)


class LivenessPolicy:
    def __init__(self, checker: LivenessChecker | None = None, required: bool = True) -> None:
        if not isinstance(required, bool):
            raise LivenessError("Liveness requirement must be a boolean")
        self.checker = checker
        self.required = required

    def evaluate(self, frame: Any) -> LivenessResult:
        if self.checker is None:
            return LivenessResult(LivenessStatus.UNAVAILABLE, self.required)
        try:
            is_live = self.checker.is_live(frame)
        except Exception as exc:
            raise LivenessError("Liveness verification failed") from exc
        if not isinstance(is_live, bool):
            raise LivenessError("Liveness checker returned an invalid result")
        status = LivenessStatus.LIVE if is_live else LivenessStatus.SPOOF
        return LivenessResult(status=status, required=self.required)
