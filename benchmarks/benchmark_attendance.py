import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EMBEDDING_DIMENSIONS = 128


def make_embedding(seed: int) -> tuple[float, ...]:
    generator = random.Random(seed)
    return tuple(generator.random() for _ in range(EMBEDDING_DIMENSIONS))


def run_benchmark(events: int, users: int, samples_per_user: int) -> dict[str, object]:
    from face_attendance.attendance import AttendanceLog
    from face_attendance.registry import FaceRegistry

    with tempfile.TemporaryDirectory(prefix="face-attendance-benchmark-") as directory:
        root = Path(directory)
        registry = FaceRegistry(
            root / "registry.json",
            max_embeddings_per_user=samples_per_user,
        )
        registry.ensure_exists()
        registry_started = time.perf_counter()
        for user_index in range(users):
            name = f"user-{user_index:04d}"
            for sample_index in range(samples_per_user):
                registry.register(name, make_embedding(user_index * 100 + sample_index))
        registry_seconds = time.perf_counter() - registry_started

        attendance = AttendanceLog(root / "attendance.csv")
        attendance.ensure_exists()
        signed_in: set[str] = set()
        attendance_started = time.perf_counter()
        for event_index in range(events):
            name = f"user-{event_index % users:04d}"
            if name in signed_in:
                attendance.record(name, "out")
                signed_in.remove(name)
            else:
                attendance.record(name, "in")
                signed_in.add(name)
        attendance_seconds = time.perf_counter() - attendance_started

    return {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "events": events,
        "users": users,
        "samples_per_user": samples_per_user,
        "registry_seconds": round(registry_seconds, 6),
        "attendance_seconds": round(attendance_seconds, 6),
        "events_per_second": round(events / attendance_seconds, 3),
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure durable registry and attendance operations with synthetic data"
    )
    parser.add_argument("--events", type=positive_int, default=100)
    parser.add_argument("--users", type=positive_int, default=8)
    parser.add_argument("--samples-per-user", type=positive_int, default=5)
    arguments = parser.parse_args()
    result = run_benchmark(arguments.events, arguments.users, arguments.samples_per_user)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
