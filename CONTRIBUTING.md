# Contributing

Thank you for helping improve Face Attendance. Contributions should be focused,
reproducible, and safe for biometric data.

## Before you start

- Read the [Code of Conduct](CODE_OF_CONDUCT.md).
- For security vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of
  opening a public issue.
- Search existing issues and pull requests before starting substantial work.
- Keep changes scoped to one problem so they can be reviewed and reverted
  independently.

## Development setup

Use Python 3.10 or newer.

```text
git clone https://github.com/ogc16/face-recognition.git
cd face-recognition
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on macOS and Linux, then install the project with
development tools:

```text
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`dlib`, which is pulled in by `face-recognition`, may require a compiler and
CMake on platforms without a prebuilt wheel. A camera is needed only for live
GUI testing; the core tests use injected fakes.

## Checks

Run the same checks used by CI before opening a pull request:

```text
python -m pytest --cov=face_attendance --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m compileall -q src main.py util.py
```

Run the formatter before re-running the format check:

```text
python -m ruff format .
```

Add or update tests for behavior changes. Tests should use temporary directories,
synthetic frames, and fake encoders rather than a webcam, real faces, or
production data.

## Pull requests

Keep pull requests focused and include:

1. A clear description of the problem and the chosen solution.
2. Tests for new behavior and regression coverage for fixes.
3. Updated documentation when configuration, commands, or security behavior
   changes.
4. A note about any deployment, camera, model, or privacy impact.

Use descriptive commit messages, for example `fix: reject duplicate registry
names`. Do not rewrite unrelated history in a pull request.

## Data safety

Never commit:

- face images or videos;
- face embeddings, registries, or attendance records;
- `config.json` containing local paths or deployment settings;
- camera credentials, tokens, or private logs.

Use the files under `data/` only as local runtime state. Report suspected
exposure through the private channel described in [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contribution is licensed under the
[MIT License](LICENSE).
