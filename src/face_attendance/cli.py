import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .attendance import AttendanceEvent, AttendanceLog
from .config import AppConfig
from .errors import FaceAttendanceError
from .recognition import RecognitionResult, RecognitionStatus
from .registry import FaceRegistry
from .runtime import Runtime, build_runtime


def _load_image(path: Path) -> Any:
    try:
        cv2 = importlib.import_module("cv2")
        numpy = importlib.import_module("numpy")
    except Exception as exc:
        raise FaceAttendanceError("Vision dependencies are unavailable") from exc
    try:
        encoded = numpy.fromfile(str(path), dtype=numpy.uint8)
        if encoded.size == 0:
            raise FaceAttendanceError(f"Unable to read image: {path}")
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise FaceAttendanceError(f"Unable to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except FaceAttendanceError:
        raise
    except Exception as exc:
        raise FaceAttendanceError(f"Unable to read image: {path}") from exc


def _print_result(result: RecognitionResult) -> None:
    payload = {
        "status": result.status.value,
        "name": result.name,
        "distance": result.distance,
        "match_quality": result.match_quality,
        "confidence": result.confidence,
        "face_count": result.face_count,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _print_attendance(events: tuple[AttendanceEvent, ...]) -> None:
    if not events:
        print("No attendance events recorded")
        return
    for event in events:
        print(f"{event.timestamp}\t{event.name}\t{event.action}")


def _register(runtime: Runtime, name: str, image_path: Path) -> int:
    embedding = runtime.service.embedding_for(_load_image(image_path))
    added = runtime.registry.register(name, embedding)
    record = runtime.registry.get(name)
    display_name = record.name if record is not None else name
    if added:
        print(f"Registered {display_name}")
    else:
        print(f"No sample added for {display_name}: duplicate or maximum samples reached")
    return 0


def _recognize(runtime: Runtime, image_path: Path) -> int:
    result = runtime.service.authenticate(_load_image(image_path), runtime.liveness_policy)
    _print_result(result)
    return 0 if result.status is RecognitionStatus.MATCH else 2


def _list_users(config: AppConfig) -> int:
    registry = FaceRegistry(
        config.registry_path,
        max_embeddings_per_user=config.max_embeddings_per_user,
    )
    names = registry.names()
    if not names:
        print("No users registered")
        return 0
    for name in names:
        print(name)
    return 0


def _remove_user(config: AppConfig, name: str) -> int:
    registry = FaceRegistry(
        config.registry_path,
        max_embeddings_per_user=config.max_embeddings_per_user,
    )
    record = registry.get(name)
    if registry.remove(name):
        print(f"Removed {record.name if record is not None else name}")
    else:
        print(f"No registered user named {name}")
    return 0


def _init(config: AppConfig) -> int:
    registry = FaceRegistry(
        config.registry_path,
        max_embeddings_per_user=config.max_embeddings_per_user,
    )
    registry.ensure_exists()
    AttendanceLog(config.attendance_path).ensure_exists()
    print(f"Registry: {registry.path}")
    print(f"Attendance log: {config.attendance_path}")
    return 0


def _doctor(config: AppConfig) -> int:
    registry = FaceRegistry(
        config.registry_path,
        max_embeddings_per_user=config.max_embeddings_per_user,
    )
    registry.ensure_exists()
    attendance = AttendanceLog(config.attendance_path)
    attendance.ensure_exists()
    print(f"Registry OK: {registry.path} ({len(registry)} users)")
    print(f"Attendance log OK: {attendance.path} ({len(attendance.events())} events)")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face recognition attendance system")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help="Path to a JSON configuration file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create the registry and attendance files")
    subparsers.add_parser("doctor", help="Validate data files without loading vision models")
    register = subparsers.add_parser("register", help="Register a face from an image")
    register.add_argument("name")
    register.add_argument("image", type=Path)
    recognize = subparsers.add_parser("recognize", help="Recognize a face in an image")
    recognize.add_argument("image", type=Path)
    subparsers.add_parser("list", help="List registered users")
    remove = subparsers.add_parser("remove", help="Remove a registered user")
    remove.add_argument("name")
    subparsers.add_parser("attendance", help="Print attendance events")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        config = AppConfig.from_file(arguments.config)
        if arguments.command == "init":
            return _init(config)
        if arguments.command == "doctor":
            return _doctor(config)
        if arguments.command == "list":
            return _list_users(config)
        if arguments.command == "remove":
            return _remove_user(config, arguments.name)
        if arguments.command == "attendance":
            _print_attendance(AttendanceLog(config.attendance_path).events())
            return 0
        runtime = build_runtime(config)
        if arguments.command == "register":
            return _register(runtime, arguments.name, arguments.image)
        return _recognize(runtime, arguments.image)
    except (FaceAttendanceError, ImportError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
