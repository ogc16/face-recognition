import pytest

from face_attendance.camera import (
    CameraFactory,
    FixedFrameSource,
    FrameSequenceFactory,
    OpenCVCamera,
    OpenCVCameraFactory,
    VideoFileCameraFactory,
    VideoFileFrameSource,
    require_vision_dependencies,
    video_file_frame_source,
)
from face_attendance.errors import CameraError, DependencyError
from face_attendance.protocols import FrameSource

_BGR_TO_RGB = 4


class _FakeCapture:
    """Stands in for ``cv2.VideoCapture``."""

    def __init__(self, opened=True, frames=None, read_error=None):
        self.opened = opened
        self.frames = list(frames or [])
        self.read_error = read_error
        self.released = False
        self.requested_index = None

    def isOpened(self):
        return self.opened

    def read(self):
        if self.read_error is not None:
            raise self.read_error
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class _FakeCv2:
    """Minimal ``cv2`` stand-in recording calls made against it."""

    COLOR_BGR2RGB = _BGR_TO_RGB

    def __init__(self, capture):
        self.capture = capture
        self.converted = []

    def VideoCapture(self, index):
        self.capture.requested_index = index
        return self.capture

    def cvtColor(self, frame, code):
        assert code == self.COLOR_BGR2RGB
        self.converted.append(frame)
        return ("rgb", frame)


@pytest.fixture
def fake_cv2(monkeypatch):
    def install(capture):
        module = _FakeCv2(capture)
        monkeypatch.setattr(
            "face_attendance.camera.optional_module",
            lambda name: module if name == "cv2" else None,
        )
        return module

    return install


@pytest.fixture
def no_cv2(monkeypatch):
    monkeypatch.setattr(
        "face_attendance.camera.optional_module",
        lambda name: None,
    )


def frame_source_stub():
    """Return a minimal object satisfying the FrameSource contract."""

    class Stub:
        def read(self):
            return False, None

        def release(self):
            return None

    return Stub()


def test_require_vision_dependencies_passes_when_present():
    require_vision_dependencies("json")


def test_require_vision_dependencies_reports_the_missing_package(no_cv2):
    with pytest.raises(DependencyError, match="cv2"):
        require_vision_dependencies("cv2")


def test_opencv_camera_rejects_a_boolean_index(fake_cv2):
    fake_cv2(_FakeCapture())

    with pytest.raises(CameraError, match="integer"):
        OpenCVCamera(True)


def test_opencv_camera_rejects_a_non_integer_index(fake_cv2):
    fake_cv2(_FakeCapture())

    with pytest.raises(CameraError, match="integer"):
        OpenCVCamera("0")


def test_opencv_camera_rejects_a_negative_index(fake_cv2):
    fake_cv2(_FakeCapture())

    with pytest.raises(CameraError, match="zero or greater"):
        OpenCVCamera(-1)


def test_opencv_camera_requires_open_cv(no_cv2):
    with pytest.raises(DependencyError):
        OpenCVCamera(0)


def test_opencv_camera_reports_a_device_that_will_not_open(fake_cv2):
    capture = _FakeCapture(opened=False)
    fake_cv2(capture)

    with pytest.raises(CameraError, match="Unable to open camera 2"):
        OpenCVCamera(2)

    assert capture.released is True


def test_opencv_camera_converts_frames_to_rgb(fake_cv2):
    capture = _FakeCapture(frames=[object()])
    module = fake_cv2(capture)
    camera = OpenCVCamera(0)

    success, frame = camera.read()

    assert success is True
    assert frame == ("rgb", module.converted[0])
    assert capture.requested_index == 0


def test_opencv_camera_reports_an_empty_read_as_transient(fake_cv2):
    fake_cv2(_FakeCapture(frames=[]))

    success, frame = OpenCVCamera(0).read()

    assert success is False
    assert frame is None


def test_opencv_camera_wraps_a_device_read_failure(fake_cv2):
    fake_cv2(_FakeCapture(read_error=OSError("device gone")))

    with pytest.raises(CameraError, match="Unable to read from the camera"):
        OpenCVCamera(0).read()


def test_opencv_camera_wraps_a_conversion_failure(fake_cv2, monkeypatch):
    module = fake_cv2(_FakeCapture(frames=[object()]))

    def explode(frame, code):
        raise ValueError("bad pixel format")

    monkeypatch.setattr(module, "cvtColor", explode)

    with pytest.raises(CameraError, match="Unable to read from the camera"):
        OpenCVCamera(0).read()


def test_opencv_camera_refuses_to_read_after_release(fake_cv2):
    fake_cv2(_FakeCapture(frames=[object()]))
    camera = OpenCVCamera(0)
    camera.release()

    with pytest.raises(CameraError, match="released"):
        camera.read()


def test_opencv_camera_release_is_idempotent(fake_cv2):
    capture = _FakeCapture()
    fake_cv2(capture)
    camera = OpenCVCamera(0)

    camera.release()
    camera.release()

    assert capture.released is True


def test_opencv_camera_factory_passes_the_requested_index(fake_cv2):
    capture = _FakeCapture()
    module = fake_cv2(capture)

    source = OpenCVCameraFactory().create(3)

    assert isinstance(source, OpenCVCamera)
    assert capture.requested_index == 3
    assert module is not None


def test_opencv_camera_factory_override_wins(fake_cv2):
    capture = _FakeCapture()
    fake_cv2(capture)

    OpenCVCameraFactory(camera_index_override=7).create(1)

    assert capture.requested_index == 7


def test_video_file_source_replays_then_reports_end_of_stream(fake_cv2):
    fake_cv2(_FakeCapture(frames=[object(), object()]))
    source = video_file_frame_source("clip.mp4")

    assert source.read()[0] is True
    assert source.read()[0] is True
    assert source.read() == (False, None)


def test_video_file_source_reports_an_unreadable_file(fake_cv2):
    capture = _FakeCapture(opened=False)
    fake_cv2(capture)

    with pytest.raises(CameraError, match="Unable to open video file"):
        VideoFileFrameSource("missing.mp4")

    assert capture.released is True


def test_video_file_source_requires_open_cv(no_cv2):
    with pytest.raises(DependencyError):
        VideoFileFrameSource("clip.mp4")


def test_video_file_source_refuses_to_read_after_release(fake_cv2):
    fake_cv2(_FakeCapture(frames=[object()]))
    source = video_file_frame_source("clip.mp4")
    source.release()

    with pytest.raises(CameraError, match="released"):
        source.read()


def test_video_file_source_release_is_idempotent(fake_cv2):
    capture = _FakeCapture()
    fake_cv2(capture)
    source = video_file_frame_source("clip.mp4")

    source.release()
    source.release()

    assert capture.released is True


def test_video_file_factory_ignores_the_camera_index(fake_cv2):
    capture = _FakeCapture()
    module = fake_cv2(capture)

    source = VideoFileCameraFactory("clip.mp4").create(9)

    assert isinstance(source, VideoFileFrameSource)
    assert capture.requested_index == "clip.mp4"
    assert module is not None


def test_fixed_frame_source_yields_frames_in_order():
    source = FixedFrameSource(("a", "b"))

    assert source.read() == (True, "a")
    assert source.read() == (True, "b")
    assert source.read() == (False, None)


def test_fixed_frame_source_can_loop():
    source = FixedFrameSource(("a",), loop_when_exhausted=True)

    assert source.read() == (True, "a")
    assert source.read() == (True, "a")


def test_fixed_frame_source_can_simulate_disconnection():
    source = FixedFrameSource(("a",), raise_after_frames=True)

    assert source.read() == (True, "a")

    with pytest.raises(CameraError, match="disconnected"):
        source.read()


def test_fixed_frame_source_rejects_an_empty_sequence():
    with pytest.raises(CameraError, match="empty"):
        FixedFrameSource(()).read()


def test_fixed_frame_source_refuses_to_read_after_release():
    source = FixedFrameSource(("a",))
    source.release()

    with pytest.raises(CameraError, match="released"):
        source.read()


def test_frame_sequence_factory_builds_independent_sources():
    factory = FrameSequenceFactory(("a", "b"))

    first = factory.create(0)
    second = factory.create(0)

    assert first.read() == (True, "a")
    assert second.read() == (True, "a")


def test_frame_sequence_factory_forwards_its_behaviour_flags():
    factory = FrameSequenceFactory(("a",), loop_when_exhausted=True)

    source = factory.create(0)

    assert source.read() == (True, "a")
    assert source.read() == (True, "a")


def test_factories_satisfy_the_camera_factory_contract(fake_cv2):
    fake_cv2(_FakeCapture())

    factories = [
        OpenCVCameraFactory(),
        VideoFileCameraFactory("clip.mp4"),
        FrameSequenceFactory(("a",)),
    ]

    for factory in factories:
        assert isinstance(factory, CameraFactory)


def test_frame_sources_satisfy_the_frame_source_contract():
    sources = [
        FixedFrameSource(("a",)),
        FrameSequenceFactory(("a",)).create(0),
        frame_source_stub(),
    ]

    for source in sources:
        assert isinstance(source, FrameSource)
