"""Liveness verification policy.

The policy is the single place where a liveness verdict is turned into a
yes/no decision. It is deliberately fail-closed: any unexpected value, thrown
error, or missing checker either rejects the frame or, when the operator has
explicitly opted out, is reported as :attr:`LivenessStatus.UNAVAILABLE` so the
decision is visible in logs rather than silently assumed.
"""

from __future__ import annotations

from collections.abc import Callable

from .errors import LivenessError
from .protocols import Frame, LivenessChecker
from .types import LivenessResult, LivenessStatus

__all__ = [
    "CallableLivenessChecker",
    "LivenessChecker",
    "LivenessPolicy",
    "LivenessResult",
    "LivenessStatus",
]


class CallableLivenessChecker:
    """Adapts a plain callable to the :class:`LivenessChecker` contract.

    Args:
        callback: Callable receiving a frame and returning a real :class:`bool`.

    Raises:
        LivenessError: If the callback is not callable.
    """

    def __init__(self, callback: Callable[[Frame], bool]) -> None:
        if not callable(callback):
            raise LivenessError("Liveness callback must be callable")
        self._callback = callback

    def is_live(self, frame: Frame) -> bool:
        """Delegate to the wrapped callable.

        Args:
            frame: Frame to evaluate.

        Returns:
            The callable's verdict, passed through unchanged.
        """
        return self._callback(frame)


class LivenessPolicy:
    """Decides whether a frame may proceed to face matching.

    Args:
        checker: Verdict provider, or ``None`` when no checker is configured.
        required: When ``True``, an :attr:`LivenessStatus.UNAVAILABLE` verdict
            rejects the frame. When ``False``, it is allowed.

    Raises:
        LivenessError: If ``required`` is not a boolean.
    """

    def __init__(
        self,
        checker: LivenessChecker | None = None,
        required: bool = True,
    ) -> None:
        if not isinstance(required, bool):
            raise LivenessError("Liveness requirement must be a boolean")
        self.checker = checker
        self.required = required

    def evaluate(self, frame: Frame) -> LivenessResult:
        """Evaluate a frame and return the resulting verdict.

        Args:
            frame: Frame to evaluate.

        Returns:
            A :class:`~face_attendance.types.LivenessResult`. When no checker is
            configured the status is
            :attr:`~face_attendance.types.LivenessStatus.UNAVAILABLE`.

        Raises:
            LivenessError: If the checker raises, or returns anything other
                than a real :class:`bool`.
        """
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
