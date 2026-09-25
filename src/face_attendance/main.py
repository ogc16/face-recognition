import argparse
import sys
from pathlib import Path

from .config import AppConfig
from .errors import FaceAttendanceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the face attendance desktop app")
    parser.add_argument("--config", type=Path, help="Path to a JSON configuration file")
    arguments = parser.parse_args(argv)
    try:
        from .gui import run_gui

        run_gui(AppConfig.from_file(arguments.config))
    except (FaceAttendanceError, ImportError, OSError, ValueError) as exc:
        print(f"Unable to start face attendance: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
