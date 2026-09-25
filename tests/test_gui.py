import queue
from types import SimpleNamespace

import pytest

FaceAttendanceApp = pytest.importorskip("face_attendance.gui").FaceAttendanceApp


class Status:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


def make_app():
    app = FaceAttendanceApp.__new__(FaceAttendanceApp)
    app._closed = False
    app._after_id = None
    app._task_poll_id = None
    app._task_results = queue.Queue()
    app.status_var = Status()
    app._latest_frame = None
    app._scheduled_frame_delays = []
    app._schedule_frame = app._scheduled_frame_delays.append
    return app


def test_webcam_loop_reschedules_after_unexpected_display_error():
    app = make_app()
    app.camera = SimpleNamespace(read=lambda: (True, object()))
    app._show_frame = lambda label, frame: (_ for _ in ()).throw(RuntimeError("display failed"))

    app._process_webcam()

    assert app.status_var.value == "Camera processing failed"
    assert app._scheduled_frame_delays == [500]


def test_task_loop_reschedules_after_completion_error():
    app = make_app()
    failures = []
    app._complete_task = lambda future, callback: (_ for _ in ()).throw(
        RuntimeError("callback failed")
    )
    app._task_failed = failures.append
    future = SimpleNamespace()
    app._task_results.put((future, lambda value: None))
    scheduled = []
    app._schedule_task_poll = lambda: scheduled.append(True)

    app._drain_tasks()

    assert len(failures) == 1
    assert str(failures[0]) == "callback failed"
    assert scheduled == [True]
