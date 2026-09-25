# Face Attendance

A local face recognition attendance system with a Tkinter desktop app and a scriptable CLI. It stores face embeddings rather than camera images, keeps the core logic testable without a display, and supports configurable liveness integration.

## What is included

- Live webcam capture with a configurable camera index
- Face registration with normalized names, validation, and multiple samples per user
- Single-face recognition with a configurable distance tolerance
- Sign-in and sign-out events in a local CSV log
- Atomic, versioned JSON embedding storage without `pickle` deserialization
- Interprocess file locking for registry and attendance updates
- Optional pluggable liveness checking with a fail-closed required mode
- CLI commands for initialization, registration, recognition, user listing, and attendance output
- Dependency-injected core services covered by unit tests

## Requirements

- Python 3.10 or newer
- A webcam and a working camera driver for the desktop app
- Tkinter from the Python installation
- A supported compiler/toolchain for `dlib` when installing `face-recognition` on Windows

The optional vision model is not bundled. Camera and model behavior must be verified on the deployment machine.

## Setup

Create an environment and install the project with its development tools:

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS and Linux, activate the environment with `source .venv/bin/activate`.

Launch the desktop app from the checkout with:

```text
python main.py
```

Installed environments can use `python -m face_attendance` or `face-attendance-gui`.

On Windows, `face-recognition` depends on `dlib`. If pip attempts to build `dlib` from source, install Visual Studio Build Tools and CMake, or use a Python/platform combination with a compatible prebuilt wheel.

## Configuration

Copy `config.example.json` to `config.json` and pass it with `--config`:

```text
python main.py --config config.json
python -m face_attendance.cli --config config.json list
```

Environment variables override values from the configuration file:

| Setting | Environment variable |
| --- | --- |
| Camera index | `FACE_ATTENDANCE_CAMERA_INDEX` |
| Registry path | `FACE_ATTENDANCE_REGISTRY_PATH` |
| Attendance path | `FACE_ATTENDANCE_ATTENDANCE_PATH` |
| Match tolerance | `FACE_ATTENDANCE_TOLERANCE` |
| Require liveness | `FACE_ATTENDANCE_REQUIRE_LIVENESS` |
| Samples per user | `FACE_ATTENDANCE_MAX_EMBEDDINGS` |
| Window size | `FACE_ATTENDANCE_WINDOW_WIDTH`, `FACE_ATTENDANCE_WINDOW_HEIGHT` |

Unknown `FACE_ATTENDANCE_*` variables and unknown JSON keys are rejected so misspelled security settings cannot silently fall back to defaults. The default tolerance is `0.6`. Lower values are stricter and can reject valid matches; higher values increase false accepts. Register several samples per user when lighting or angles vary.

The default data paths are relative to the current working directory:

- `data/registry.json`
- `data/attendance.csv`

For a sensitive deployment, configure absolute paths on an access-controlled local disk rather than a repository, synchronized folder, or removable drive.

## Desktop workflow

1. Start the app and wait for a camera frame.
2. Select **Register user**, enter a name, and capture a face sample.
3. Add several samples for the user when practical.
4. Use **Sign in** after recognition; use **Sign out** to record the matching departure.
5. Review the CSV file for the append-only attendance history.

Attendance transitions are enforced: the first event for a user must be `in`, repeated `in` events are rejected, and `out` is accepted only while the user is signed in. Recognition and attendance processing run in a bounded worker thread; results are delivered back to Tkinter on the UI thread.

Enrollment is intentionally operator-trusted in this lightweight application. Anyone with access to the desktop or CLI can register a new identity, so deploy the app only on a controlled workstation. Add administrator authorization before using it as a high-assurance enrollment system.

## CLI

Run the installed `face-attendance` command or the module directly:

```text
python -m face_attendance.cli init
python -m face_attendance.cli register "Ada Lovelace" photo.jpg
python -m face_attendance.cli recognize photo.jpg
python -m face_attendance.cli list
python -m face_attendance.cli attendance
```

`init` creates the versioned registry and an attendance CSV header. `register` accepts an image containing exactly one face. `recognize` exits with status `0` for a match, `2` for no match/no face/multiple faces, and `1` for configuration, dependency, liveness, or file errors. The `attendance` command prints events as tab-separated text.

The CLI has no bundled liveness model. When `require_liveness` is `true`, recognition fails unless a checker is supplied through the programmatic `build_runtime` API.

## Liveness integration

Liveness checking is pluggable because a robust anti-spoof model is deployment-specific. Pass a `LivenessChecker` to `build_runtime`:

- `false` (default) permits recognition when no checker is configured, while the desktop app displays that liveness is unavailable.
- `true` requires a checker and rejects authentication when it is missing, errors, or reports a spoof.
- A checker receives the same RGB frame used by the recognition backend.

No high-assurance anti-spoof guarantee is provided by the default deployment.

## Data, privacy, and migration

- Embeddings are stored as JSON in the configured registry path.
- Camera frames are not written to disk.
- The old pickle-based database format is intentionally not loaded; re-register users once after migration.
- Names and attendance events are personal data; obtain consent and follow applicable biometric-privacy requirements.
- Restrict access to the registry, attendance CSV, backups, and logs. `.gitignore` prevents the default `data/` directory from being committed, but it does not prevent cloud synchronization.
- Deleting a file is not a guaranteed secure erasure on SSDs, snapshots, or cloud backups; use an approved data-retention and disposal process.

## Development

The core modules use dependency injection, so recognition, persistence, attendance, configuration, liveness, and file-transition behavior can be tested without a webcam or the optional vision libraries.

```text
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Live camera capture, `dlib` installation, model accuracy, and anti-spoof integration still require deployment-machine testing.

## Architecture

```text
main.py
└── face_attendance
    ├── config.py          # JSON/environment configuration
    ├── registry.py        # Versioned JSON embedding store
    ├── file_lock.py       # Cross-process data-file locking
    ├── recognition.py     # Face matching and policy enforcement
    ├── attendance.py      # CSV event log and state transitions
    ├── liveness.py        # Pluggable liveness policy
    ├── gui.py             # Tkinter desktop workflow
    ├── cli.py             # Scriptable commands
    ├── runtime.py         # Dependency assembly
    └── validation.py      # Names and embedding validation
```

The GUI performs recognition and attendance work in a worker, but only the Tk main thread updates widgets. The registry and attendance writer serialize access across processes, write durable temporary or appended data, and use atomic replacement where applicable.

## Troubleshooting

- **Missing dependencies:** install the project with `python -m pip install -e .` and confirm `python -c "import cv2, face_recognition, PIL"`.
- **No camera frame:** check the camera index, close other applications using the camera, and confirm camera permissions.
- **Unknown face:** add more samples, improve lighting, and tune tolerance only with local testing; do not treat a larger tolerance as a security control.
- **Headless launch error:** use the CLI for non-interactive workflows or run the GUI on a host with a desktop display.
