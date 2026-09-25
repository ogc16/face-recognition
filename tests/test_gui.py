import pytest

from face_attendance import gui
from face_attendance.config import AppConfig
from face_attendance.errors import (
    CameraError,
    DependencyError,
    RecognitionError,
    RegistryError,
)
from face_attendance.liveness import CallableLivenessChecker
from face_attendance.runtime import Runtime
from face_attendance.types import (
    AttendanceEvent,
    LivenessResult,
    LivenessStatus,
    RecognitionResult,
    RecognitionStatus,
)
from tk_fakes import FakeImageModule, FakeImageTkModule, GuiHarness, TclError

_FRAME = object()


class StubService:
    def __init__(self, result=None, embedding=(0.1, 0.2)):
        self.result = result or RecognitionResult(status=RecognitionStatus.UNKNOWN, face_count=1)
        self.embedding = embedding
        self.authenticate_calls = 0
        self.embedding_calls = 0
        self.error: Exception | None = None

    def authenticate(self, frame, liveness_policy):
        self.authenticate_calls += 1
        if self.error is not None:
            raise self.error
        return self.result

    def recognize(self, frame):
        return self.result

    def embedding_for(self, frame):
        self.embedding_calls += 1
        return self.embedding


class StubRegistry:
    def __init__(self, names=("Ada",)):
        self._names = tuple(names)
        self.registered: list[tuple[str, tuple[float, ...]]] = []
        self.removals: list[str] = []
        self.removed_result = True
        self.error: Exception | None = None

    def names(self):
        if self.error is not None:
            raise self.error
        return self._names

    def register(self, name, embedding):
        self.registered.append((name, embedding))
        return True

    def remove(self, name):
        self.removals.append(name)
        return self.removed_result


class StubAttendance:
    def __init__(self):
        self.recorded: list[tuple[str, str]] = []
        self.action: str | None = None

    def record(self, name, action, timestamp=None):
        self.recorded.append((name, action))
        return AttendanceEvent(
            timestamp="2026-09-25T09:00:00+00:00", name=name, action=self.action or action
        )


class StubLivenessPolicy:
    def __init__(self, checker=None, required=True):
        self.checker = checker
        self.required = required

    def evaluate(self, frame):
        return LivenessResult(LivenessStatus.LIVE, self.required)


class StubRuntime:
    def __init__(self, config, service, registry, attendance, liveness_policy):
        self.config = config
        self.service = service
        self.registry = registry
        self.attendance = attendance
        self.liveness_policy = liveness_policy


class StubCamera:
    def __init__(self, frames=None, error=None):
        self.frames = list(frames or [])
        self.error = error
        self.released = False

    def read(self):
        if self.error is not None:
            raise self.error
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class StubCameraFactory:
    def __init__(self, camera):
        self.camera = camera
        self.requested_index = None

    def create(self, camera_index):
        self.requested_index = camera_index
        return self.camera


@pytest.fixture
def harness(monkeypatch):
    return GuiHarness().install(monkeypatch)


def build_app(
    tmp_path,
    harness,
    *,
    service=None,
    registry=None,
    attendance=None,
    liveness_policy=None,
    camera=None,
    require_liveness=True,
):
    config = AppConfig(
        registry_path=tmp_path / "registry.json",
        attendance_path=tmp_path / "attendance.csv",
        camera_index=2,
        require_liveness=require_liveness,
    )
    runtime = Runtime(
        config=config,
        service=service or StubService(),
        registry=registry or StubRegistry(),
        attendance=attendance or StubAttendance(),
        liveness_policy=liveness_policy or StubLivenessPolicy(),
    )
    camera = camera or StubCamera(frames=[_FRAME])
    factory = StubCameraFactory(camera)
    app = gui.FaceAttendanceApp(runtime, camera_factory=factory)
    return app, camera, factory


def drain(app):
    """Run every task the app has completed and reschedule the poll."""
    for _ in range(20):
        app._drain_tasks()
        if app._task_results.empty():
            return
    raise AssertionError("task queue did not settle")


def test_app_configures_the_window_from_config(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)

    assert harness.root.title_text == "Face Attendance"
    expected = f"{app.config.window_width}x{app.config.window_height}+100+100"
    assert harness.root.geometry_text == expected
    assert harness.root.minsize_args == (800, 600)


def test_app_registers_a_close_handler(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)

    assert harness.root.protocols["WM_DELETE_WINDOW"] == app.close


def test_app_asks_the_factory_for_the_configured_camera(tmp_path, harness):
    _, camera, factory = build_app(tmp_path, harness)

    assert factory.requested_index == 2
    assert isinstance(camera, StubCamera)


def test_app_destroys_the_root_when_the_camera_fails(tmp_path, harness):
    def exploding_create(camera_index):
        raise CameraError("no device")

    config = AppConfig(registry_path=tmp_path / "r.json", attendance_path=tmp_path / "a.csv")
    runtime = Runtime(
        config=config,
        service=StubService(),
        registry=StubRegistry(),
        attendance=StubAttendance(),
        liveness_policy=StubLivenessPolicy(),
    )
    factory = StubCameraFactory(StubCamera())
    factory.create = exploding_create

    with pytest.raises(CameraError):
        gui.FaceAttendanceApp(runtime, camera_factory=factory)

    assert harness.root.destroyed is True


def test_app_schedules_the_frame_loop_and_task_poll(tmp_path, harness):
    build_app(tmp_path, harness)

    delays = [item[1] for item in harness.root.scheduled]

    assert 0 in delays
    assert 50 in delays


def test_app_lists_registered_users(tmp_path, harness):
    build_app(tmp_path, harness, registry=StubRegistry(names=("Ada", "Grace")))

    assert harness.tk.listboxes[0].items == ["Ada", "Grace"]


def test_app_reports_a_failure_to_load_users(tmp_path, harness):
    registry = StubRegistry()
    registry.error = RegistryError("corrupt")

    app, _, _ = build_app(tmp_path, harness, registry=registry)

    assert app.status_var.get() == "Unable to load registered users"


def test_refresh_users_ignores_a_missing_listbox(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._users_listbox = None

    app._refresh_users()


def test_frame_loop_displays_a_frame(tmp_path, harness):
    app, camera, _ = build_app(tmp_path, harness)

    app._process_webcam()

    assert app._latest_frame is _FRAME
    assert app.video_label.options["image"] is not None
    assert camera.frames == []


def test_frame_loop_reports_a_missing_frame(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness, camera=StubCamera(frames=[]))

    app._process_webcam()

    assert app.status_var.get() == "Waiting for camera frame…"
    assert app._latest_frame is None


def test_frame_loop_reports_a_camera_error(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness, camera=StubCamera(error=CameraError("gone")))

    app._process_webcam()

    assert app.status_var.get() == "gone"


def test_frame_loop_hides_an_unexpected_camera_error(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness, camera=StubCamera(error=ZeroDivisionError()))

    app._process_webcam()

    assert app.status_var.get() == "Camera processing failed"


def test_frame_loop_stops_once_closed(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._closed = True

    app._process_webcam()

    assert app._latest_frame is None


def test_scheduling_stops_once_closed(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._closed = True
    harness.root.scheduled.clear()

    app._schedule_frame(0)
    app._schedule_task_poll()

    assert harness.root.scheduled == []


def test_scheduling_survives_a_destroyed_root(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)

    def destroyed_after(delay, callback=None, *args):
        raise TclError("main window gone")

    harness.root.after = destroyed_after
    app._schedule_frame(0)
    app._schedule_task_poll()

    assert app._after_id is None
    assert app._task_poll_id is None


def test_show_frame_requires_the_image_dependencies(tmp_path, harness, monkeypatch):
    app, _, _ = build_app(tmp_path, harness)
    harness.set_image_dependencies(monkeypatch, image=None)

    with pytest.raises(DependencyError):
        app._show_frame(app.video_label, _FRAME)


def test_show_frame_thumbnails_to_the_label_size(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)

    app._show_frame(app.video_label, _FRAME)

    image = FakeImageModule.created[-1]
    assert image.thumbnail_sizes == [(628, 468)]
    assert FakeImageTkModule.photos[-1][0] == "photo"
    assert app.video_label.options["image"] is FakeImageTkModule.photos[-1]


def test_snapshot_returns_none_before_the_first_frame(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._latest_frame = None

    assert app._snapshot() is None


def test_snapshot_copies_the_frame_when_supported(tmp_path, harness):
    class CopyableFrame:
        def __init__(self):
            self.copies = 0

        def copy(self):
            self.copies += 1
            return "copied"

    frame = CopyableFrame()
    app, _, _ = build_app(tmp_path, harness, camera=StubCamera(frames=[frame]))

    app._process_webcam()

    assert app._snapshot() == "copied"
    assert frame.copies == 1


def test_login_without_a_frame_warns(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness, camera=StubCamera(frames=[]))
    app._process_webcam()

    app._on_login()

    assert harness.messagebox.of_kind("warning")[0][1] == "No camera frame"
    assert app._busy is False


def test_login_records_attendance_for_a_match(tmp_path, harness):
    result = RecognitionResult(
        status=RecognitionStatus.MATCH,
        face_count=1,
        name="Ada",
        distance=0.1,
        confidence=0.93,
    )
    attendance = StubAttendance()
    app, _, _ = build_app(
        tmp_path, harness, service=StubService(result=result), attendance=attendance
    )
    app._process_webcam()

    app._on_login()
    drain(app)

    assert attendance.recorded == [("Ada", "in")]
    assert app.status_var.get() == "Ada signed in"
    assert harness.messagebox.of_kind("info")[0][2].endswith("93% match quality.")


def test_logout_records_an_out_event(tmp_path, harness):
    result = RecognitionResult(status=RecognitionStatus.MATCH, face_count=1, name="Ada")
    attendance = StubAttendance()
    app, _, _ = build_app(
        tmp_path, harness, service=StubService(result=result), attendance=attendance
    )
    app._process_webcam()

    app._on_logout()
    drain(app)

    assert attendance.recorded == [("Ada", "out")]
    assert app.status_var.get() == "Ada signed out"


def test_login_reports_a_confirmed_match_without_quality(tmp_path, harness):
    result = RecognitionResult(status=RecognitionStatus.MATCH, face_count=1, name="Ada")
    app, _, _ = build_app(tmp_path, harness, service=StubService(result=result))
    app._process_webcam()

    app._on_login()
    drain(app)

    assert harness.messagebox.of_kind("info")[0][2].endswith("confirmed.")


def test_login_is_ignored_while_busy(tmp_path, harness):
    service = StubService()
    app, _, _ = build_app(tmp_path, harness, service=service)
    app._process_webcam()
    app._set_busy(True)

    app._on_login()

    assert service.authenticate_calls == 0
    assert app._busy is True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RecognitionStatus.NO_FACE, "No face was detected."),
        (RecognitionStatus.MULTIPLE_FACES, "Multiple faces were detected."),
        (RecognitionStatus.UNKNOWN, "This face is not registered."),
    ],
)
def test_unmatched_logins_explain_the_failure(tmp_path, harness, status, expected):
    result = RecognitionResult(status=status, face_count=1)
    app, _, _ = build_app(tmp_path, harness, service=StubService(result=result))
    app._process_webcam()

    app._on_login()
    drain(app)

    assert app.status_var.get() == "Recognition failed"
    assert harness.messagebox.of_kind("warning")[0][2].startswith(expected)


def test_an_unlisted_status_falls_back_to_a_generic_message(tmp_path, harness):
    result = RecognitionResult(status="mystery", face_count=1)
    app, _, _ = build_app(tmp_path, harness, service=StubService(result=result))
    app._process_webcam()

    app._on_login()
    drain(app)

    assert harness.messagebox.of_kind("warning")[0][2] == "Face recognition failed. Try again."


def test_a_failing_service_surfaces_an_error(tmp_path, harness):
    service = StubService()
    service.error = RecognitionError("model exploded")
    app, _, _ = build_app(tmp_path, harness, service=service)
    app._process_webcam()

    app._on_login()
    drain(app)

    assert app.status_var.get() == "Action failed"
    assert app._busy is False
    assert harness.messagebox.of_kind("error")[0][1] == "Action failed"


def test_buttons_are_disabled_while_busy(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)

    app._set_busy(True)

    assert app.login_button.options["state"] == "disabled"
    assert app.logout_button.options["state"] == "disabled"
    assert app.register_button.options["state"] == "disabled"
    assert app.remove_button.options["state"] == "disabled"

    app._set_busy(False)

    assert app.login_button.options["state"] == "normal"


def test_removal_without_a_selection_warns(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)

    app._remove_selected_user()

    assert harness.messagebox.of_kind("warning")[0][1] == "No user selected"


def test_removal_is_ignored_while_busy(tmp_path, harness):
    registry = StubRegistry()
    app, _, _ = build_app(tmp_path, harness, registry=registry)
    app._set_busy(True)
    harness.tk.listboxes[0].select(0)

    app._remove_selected_user()

    assert registry.removals == []


def test_removal_is_skipped_without_a_listbox(tmp_path, harness):
    registry = StubRegistry()
    app, _, _ = build_app(tmp_path, harness, registry=registry)
    app._users_listbox = None

    app._remove_selected_user()

    assert registry.removals == []


def test_removal_can_be_declined(tmp_path, harness):
    registry = StubRegistry()
    app, _, _ = build_app(tmp_path, harness, registry=registry)
    harness.tk.listboxes[0].select(0)
    harness.messagebox.confirm = False

    app._remove_selected_user()

    assert registry.removals == []
    assert app._busy is False


def test_confirmed_removal_deletes_the_user(tmp_path, harness):
    registry = StubRegistry()
    app, _, _ = build_app(tmp_path, harness, registry=registry)
    harness.tk.listboxes[0].select(0)

    app._remove_selected_user()
    drain(app)

    assert registry.removals == ["Ada"]
    assert app.status_var.get() == "Removed Ada"


def test_removal_reports_an_unregistered_user(tmp_path, harness):
    registry = StubRegistry()
    registry.removed_result = False
    app, _, _ = build_app(tmp_path, harness, registry=registry)
    harness.tk.listboxes[0].select(0)

    app._remove_selected_user()
    drain(app)

    assert app.status_var.get() == "Ada was not registered"


def test_registration_without_a_frame_warns(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness, camera=StubCamera(frames=[]))
    app._process_webcam()

    app._open_registration()

    assert harness.messagebox.of_kind("warning")[0][1] == "No camera frame"
    assert app._registration_window is None


def test_registration_is_ignored_while_busy(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._process_webcam()
    app._set_busy(True)

    app._open_registration()

    assert app._registration_window is None


def test_opening_registration_builds_a_modal_window(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._process_webcam()

    app._open_registration()

    window = harness.tk.toplevels[-1]
    assert window.title_text == "Register user"
    assert window.geometry_text == "760x560+180+140"
    assert window.grabbed is True
    assert window.transient_for is harness.root
    assert app._registration_frame is _FRAME
    assert app.registration_accept_button is not None
    assert window.protocols["WM_DELETE_WINDOW"] == app._close_registration


def test_closing_registration_clears_state(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._process_webcam()
    app._open_registration()
    window = harness.tk.toplevels[-1]

    app._close_registration()

    assert window.destroyed is True
    assert app._registration_window is None
    assert app._registration_image_label is None
    assert app.registration_accept_button is None
    assert app._registration_frame is None
    assert app._registration_name is None


def test_closing_registration_tolerates_a_dead_window(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._process_webcam()
    app._open_registration()
    harness.tk.toplevels[-1].destroyed = True

    app._close_registration()

    assert app._registration_window is None


def test_closing_registration_survives_a_configure_failure(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._process_webcam()
    app._open_registration()

    def broken_configure(**kwargs):
        raise TclError("label is gone")

    app._registration_image_label.configure = broken_configure

    app._close_registration()

    assert app._registration_image_label is None


def test_accepting_registration_requires_an_open_window(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._process_webcam()
    app._open_registration()
    app._registration_window = None

    app._accept_registration()

    assert app._busy is False


def test_accepting_registration_requires_a_name_and_frame(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)

    app._accept_registration()

    assert app._busy is False


def test_accepting_registration_is_ignored_while_busy(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._process_webcam()
    app._open_registration()
    app._set_busy(True)

    app._accept_registration()

    assert app._busy is True


def test_registration_rejects_a_blank_name(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._process_webcam()
    app._open_registration()
    app._registration_name.set("   ")

    app._accept_registration()

    assert harness.messagebox.of_kind("error")[0][1] == "Invalid name"
    assert app._busy is False


def test_registration_stores_the_face(tmp_path, harness):
    service = StubService(embedding=(0.5, 0.6))
    registry = StubRegistry()
    app, _, _ = build_app(tmp_path, harness, service=service, registry=registry)
    app._process_webcam()
    app._open_registration()
    app._registration_name.set("  grace  ")

    app._accept_registration()
    drain(app)

    assert registry.registered == [("grace", (0.5, 0.6))]
    assert app.status_var.get() == "Registered grace"
    assert harness.messagebox.of_kind("info")[0][1] == "Registration complete"
    assert app._registration_window is None


def test_registration_reports_a_duplicate_sample(tmp_path, harness):
    service = StubService()
    registry = StubRegistry()
    registry.register = lambda name, embedding: False
    app, _, _ = build_app(tmp_path, harness, service=service, registry=registry)
    app._process_webcam()
    app._open_registration()
    app._registration_name.set("Ada")

    app._accept_registration()
    drain(app)

    assert harness.messagebox.of_kind("info")[0][1] == "Already registered"


def test_a_failing_registration_task_shows_the_dialog_parent(tmp_path, harness):
    service = StubService()
    service.embedding_for = lambda frame: (_ for _ in ()).throw(RecognitionError("no face"))
    app, _, _ = build_app(tmp_path, harness, service=service)
    app._process_webcam()
    app._open_registration()
    app._registration_name.set("Ada")

    app._accept_registration()
    drain(app)

    error = harness.messagebox.of_kind("error")[0]
    assert error[3]["parent"] is app._registration_window or error[1] == "Action failed"


def test_a_failing_submission_is_reported(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)

    def refusing_submit(task):
        raise RuntimeError("executor stopped")

    app._executor.submit = refusing_submit

    app._submit(lambda: None, lambda value: None)

    assert app.status_var.get() == "Action failed"
    assert app._busy is False


def test_task_reporting_survives_a_toplevel_failure(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._process_webcam()
    app._open_registration()

    def broken_showerror(title, message, **kwargs):
        raise TclError("dialog unavailable")

    harness.messagebox.showerror = broken_showerror

    app._task_failed(RuntimeError("boom"))

    assert app.status_var.get() == "Action failed"


def test_completing_a_task_after_close_does_nothing(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._closed = True
    called = []
    app._set_busy(False)

    class Done:
        def result(self):
            return "value"

    app._complete_task(Done(), called.append)

    assert called == []


def test_draining_tasks_after_close_does_nothing(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)
    app._closed = True

    app._drain_tasks()

    assert app._task_results.empty()


def test_run_enters_the_mainloop(tmp_path, harness):
    app, _, _ = build_app(tmp_path, harness)

    app.run()

    assert harness.root.mainloop_calls == 1


def test_close_releases_every_resource(tmp_path, harness):
    app, camera, _ = build_app(tmp_path, harness)
    app._process_webcam()
    app._open_registration()

    app.close()

    assert camera.released is True
    assert harness.root.destroyed is True
    assert app._closed is True
    assert app._registration_window is None


def test_close_is_idempotent(tmp_path, harness):
    app, camera, _ = build_app(tmp_path, harness)
    app.close()
    releases = []
    camera.release = lambda: releases.append(1)

    app.close()

    assert releases == []


def test_close_survives_a_camera_release_failure(tmp_path, harness):
    app, camera, _ = build_app(tmp_path, harness)

    def exploding_release():
        raise CameraError("device stuck")

    camera.release = exploding_release

    with pytest.raises(CameraError):
        app.close()

    assert harness.root.destroyed is True


def test_liveness_banner_when_no_checker_is_configured(tmp_path, harness):
    policy = StubLivenessPolicy(checker=None, required=True)
    app, _, _ = build_app(tmp_path, harness, liveness_policy=policy)

    assert app.liveness_var.get() == "Liveness required but unavailable"


def test_liveness_banner_when_liveness_is_optional(tmp_path, harness):
    policy = StubLivenessPolicy(checker=None, required=False)
    app, _, _ = build_app(tmp_path, harness, liveness_policy=policy)

    assert app.liveness_var.get() == "Liveness optional and unavailable"


def test_liveness_banner_when_a_checker_is_active(tmp_path, harness):
    policy = StubLivenessPolicy(checker=CallableLivenessChecker(lambda frame: True), required=True)
    app, _, _ = build_app(tmp_path, harness, liveness_policy=policy)

    assert app.liveness_var.get() == "Liveness active and required"


def test_liveness_banner_when_a_checker_is_optional(tmp_path, harness):
    policy = StubLivenessPolicy(checker=CallableLivenessChecker(lambda frame: True), required=False)
    app, _, _ = build_app(tmp_path, harness, liveness_policy=policy)

    assert app.liveness_var.get() == "Liveness active and optional"


def test_style_uses_the_clam_theme_when_available(tmp_path, harness):
    build_app(tmp_path, harness)

    assert harness.ttk.styles[0].used == "clam"
    assert "Title.TLabel" in harness.ttk.styles[0].configured


def test_style_skips_the_theme_when_unavailable(monkeypatch, tmp_path):
    harness = GuiHarness(themes=()).install(monkeypatch)

    build_app(tmp_path, harness)

    assert harness.ttk.styles[0].used is None


def test_run_gui_reports_a_missing_display(monkeypatch, tmp_path):
    def no_display(runtime):
        raise TclError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr("face_attendance.gui.build_runtime", lambda config: None)
    monkeypatch.setattr("face_attendance.gui.FaceAttendanceApp", no_display)
    config = AppConfig(registry_path=tmp_path / "r", attendance_path=tmp_path / "a")

    with pytest.raises(DependencyError, match="usable Tkinter display"):
        gui.run_gui(config)


def test_run_gui_returns_zero_on_success(monkeypatch, tmp_path):
    class StubApp:
        def run(self):
            return None

    monkeypatch.setattr("face_attendance.gui.build_runtime", lambda config: None)
    monkeypatch.setattr("face_attendance.gui.FaceAttendanceApp", lambda runtime: StubApp())

    result = gui.run_gui(AppConfig(registry_path=tmp_path / "r", attendance_path=tmp_path / "a"))

    assert result == 0


def test_fire_after_reports_a_missing_callback(harness):
    with pytest.raises(LookupError):
        harness.root.fire_after(30)
