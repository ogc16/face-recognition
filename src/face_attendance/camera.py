"""Camera capture abstractions and factories.

The desktop app never constructs a capture device directly. It asks a
:class:`CameraFactory` for a :class:`~face_attendance.protocols.FrameSource`,
which makes the capture backend substitutable: a physical webcam, a video file
for replay and testing, or a synthetic generator.

The factory is an *abstract factory* in the GoF sense. It produces a family of
related products (:class:`FrameSource` implementations) chosen by a single
decision (the configured camera index and backend), so callers depend on the
abstract product type only.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

from .errors import CameraError, DependencyError
from .protocols import Frame, FrameSource

__all__ = [
    "CameraFactory",
    "FixedFrameSource",
    "FrameSequenceFactory",
    "OpenCVCamera",
    "OpenCVCameraFactory",
    "VideoFileCameraFactory",
    "VideoFileFrameSource",
    "require_vision_dependencies",
    "video_file_frame_source",
]


def optional_module(name: str) -> Any:
    """Import a module, returning ``None`` when it is unavailable.

    Args:
        name: Fully qualified module name, e.g. ``"cv2"``.

    Returns:
        The imported module, or ``None`` when the import failed.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def require_vision_dependencies(*names: str) -> None:
    """Ensure every named module is importable.

    Args:
        *names: Module names that must be available, e.g. ``"cv2"``.

    Raises:
        DependencyError: If any module cannot be imported.
    """
    for name in names:
        if optional_module(name) is None:
            raise DependencyError(
                f"The '{name}' package is unavailable. Install the project with "
                "`python -m pip install -e .`."
            )


@runtime_checkable
class CameraFactory(Protocol):
    """Abstract factory producing :class:`FrameSource` instances.

    Implementations decide *how* a device is opened; callers only see the
    abstract product. This is the seam that keeps the GUI free of OpenCV.
    """

    def create(self, camera_index: int) -> FrameSource:
        """Return a frame source for the requested device.

        Args:
            camera_index: Zero-based device index, as exposed by OpenCV.

        Returns:
            A ready-to-read :class:`~face_attendance.protocols.FrameSource`.

        Raises:
            CameraError: If the device cannot be opened.
            DependencyError: If required capture dependencies are missing.
        """


class OpenCVCamera:
    """A :class:`FrameSource` backed by an OpenCV capture device.

    Args:
        camera_index: Zero-based device index.

    Raises:
        DependencyError: If OpenCV is not installed.
        CameraError: If the device cannot be opened, or the index is negative.
    """

    def __init__(self, camera_index: int) -> None:
        if isinstance(camera_index, bool) or not isinstance(camera_index, int):
            raise CameraError("Camera index must be an integer")
        if camera_index < 0:
            raise CameraError("Camera index must be zero or greater")
        require_vision_dependencies("cv2")
        cv2 = optional_module("cv2")
        if cv2 is None:  # pragma: no cover - guarded by require_vision_dependencies
            raise CameraError("OpenCV is unavailable")
        self.camera_index = camera_index
        self._capture: Any = cv2.VideoCapture(camera_index)
        if not bool(self._capture.isOpened()):
            self._capture.release()
            raise CameraError(f"Unable to open camera {camera_index}")
        self._released = False

    def read(self) -> tuple[bool, Frame | None]:
        """Return the next RGB frame from the device.

        Returns:
            A ``(success, frame)`` pair. ``(False, None)`` means no frame was
            available right now, which is normal for a busy or absent device.

        Raises:
            CameraError: If the device fails irrecoverably or was released.
        """
        if self._released:
            raise CameraError("Camera has been released")
        cv2 = optional_module("cv2")
        if cv2 is None:  # pragma: no cover - OpenCV cannot vanish at runtime
            raise CameraError("OpenCV is unavailable")
        try:
            success, frame = self._capture.read()
        except Exception as exc:
            raise CameraError("Unable to read from the camera") from exc
        if not success or frame is None:
            return False, None
        try:
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            raise CameraError("Unable to read from the camera") from exc
        return True, converted

    def release(self) -> None:
        """Release the underlying device. Idempotent."""
        if self._released:
            return
        self._released = True
        self._capture.release()


class OpenCVCameraFactory:
    """Abstract factory producing :class:`OpenCVCamera` devices.

    Args:
        camera_index_override: When set, ignore the requested index and always
            open this device. Useful for forcing a specific camera in tests.
    """

    def __init__(self, camera_index_override: int | None = None) -> None:
        self.camera_index_override = camera_index_override

    def create(self, camera_index: int) -> FrameSource:
        """Open an OpenCV capture device.

        Args:
            camera_index: Requested device index, ignored when the factory was
                constructed with an override.

        Returns:
            A live :class:`OpenCVCamera`.

        Raises:
            CameraError: If the device cannot be opened.
            DependencyError: If OpenCV is not installed.
        """
        index = (
            self.camera_index_override
            if self.camera_index_override is not None
            else camera_index
        )
        return OpenCVCamera(index)


class VideoFileFrameSource:
    """A :class:`FrameSource` that replays frames from a video file.

    This makes the capture path exercisable on a machine with no webcam, which
    is what the offline and failure-mode tests rely on.

    Args:
        path: Path to a video file readable by OpenCV.

    Raises:
        DependencyError: If OpenCV is not installed.
        CameraError: If the file cannot be opened.
    """

    def __init__(self, path: str) -> None:
        require_vision_dependencies("cv2")
        cv2 = optional_module("cv2")
        if cv2 is None:  # pragma: no cover - guarded by require_vision_dependencies
            raise CameraError("OpenCV is unavailable")
        self._capture: Any = cv2.VideoCapture(path)
        if not bool(self._capture.isOpened()):
            self._capture.release()
            raise CameraError(f"Unable to open video file: {path}")
        self._released = False

    def read(self) -> tuple[bool, Frame | None]:
        """Return the next frame from the file, then the end of stream.

        Returns:
            A ``(success, frame)`` pair. ``(False, None)`` marks end of file.

        Raises:
            CameraError: If the source was released.
        """
        if self._released:
            raise CameraError("Video source has been released")
        cv2 = optional_module("cv2")
        if cv2 is None:  # pragma: no cover - OpenCV cannot vanish at runtime
            raise CameraError("OpenCV is unavailable")
        success, frame = self._capture.read()
        if not success or frame is None:
            return False, None
        return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        """Release the underlying file handle. Idempotent."""
        if self._released:
            return
        self._released = True
        self._capture.release()


class VideoFileCameraFactory:
    """Abstract factory that always replays a fixed video file.

    Args:
        path: Path to the video file to replay.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def create(self, camera_index: int) -> FrameSource:
        """Return a replay source, ignoring the camera index.

        Args:
            camera_index: Ignored.

        Returns:
            A live :class:`VideoFileFrameSource`.

        Raises:
            CameraError: If the file cannot be opened.
        """
        return VideoFileFrameSource(self.path)


def video_file_frame_source(path: str) -> FrameSource:
    """Return a :class:`FrameSource` replaying ``path``.

    Args:
        path: Path to a video file readable by OpenCV.

    Returns:
        A live :class:`VideoFileFrameSource`.

    Raises:
        CameraError: If the file cannot be opened.
    """
    return VideoFileFrameSource(path)


class FixedFrameSource:
    """A :class:`FrameSource` that yields a fixed sequence of frames.

    Useful as a deterministic stand-in for a camera in tests and in the
    synthetic benchmark, where no video decoding is wanted.

    Args:
        frames: Frames to yield, in order.
        loop_when_exhausted: When ``True``, restart from the first frame after
            the last one instead of reporting end of stream.
        raise_after_frames: When ``True``, raise :class:`CameraError` once the
            frames are exhausted, simulating abrupt device loss.
    """

    def __init__(
        self,
        frames: tuple[Frame, ...],
        *,
        loop_when_exhausted: bool = False,
        raise_after_frames: bool = False,
    ) -> None:
        self._frames = frames
        self._loop = loop_when_exhausted
        self._raise_after = raise_after_frames
        self._index = 0
        self.released = False

    def read(self) -> tuple[bool, Frame | None]:
        """Return the next frame in the sequence.

        Returns:
            A ``(success, frame)`` pair. ``(False, None)`` marks exhaustion
            unless the source was created to loop.

        Raises:
            CameraError: If the source was released, or if it was created with
                ``raise_after_frames`` and the frames are exhausted.
        """
        if self.released:
            raise CameraError("Frame source has been released")
        if not self._frames:
            raise CameraError("Frame source is empty")
        if self._index >= len(self._frames):
            if self._raise_after:
                raise CameraError("Camera disconnected")
            if not self._loop:
                return False, None
            self._index = 0
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        """Mark the source released. Idempotent."""
        self.released = True


class FrameSequenceFactory:
    """Abstract factory producing :class:`FixedFrameSource` objects.

    Args:
        frames: Frames to hand to each created source.
        loop_when_exhausted: See :class:`FixedFrameSource`.
        raise_after_frames: See :class:`FixedFrameSource`.
    """

    def __init__(
        self,
        frames: tuple[Frame, ...],
        *,
        loop_when_exhausted: bool = False,
        raise_after_frames: bool = False,
    ) -> None:
        self._frames = frames
        self._loop = loop_when_exhausted
        self._raise_after = raise_after_frames

    def create(self, camera_index: int) -> FrameSource:
        """Return a new fixed-frame source, ignoring the camera index.

        Args:
            camera_index: Ignored.

        Returns:
            A live :class:`FixedFrameSource`.
        """
        return FixedFrameSource(
            self._frames,
            loop_when_exhausted=self._loop,
            raise_after_frames=self._raise_after,
        )
