import importlib
import sys
import types

import pytest

from face_attendance import validation
from face_attendance.errors import DependencyError, RecognitionError, RegistryError
from face_attendance.recognition import (
    DefaultFaceRecognitionBackend,
    FaceRecognitionBackend,
)
from face_attendance.validation import (
    MAX_EMBEDDING_LENGTH,
    MAX_NAME_LENGTH,
    name_key,
    normalize_name,
    validate_embedding,
)


def install_stub(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)
    return module


def stub_recognition_module(monkeypatch, encodings=None, distance=None, encode_error=None):
    module = types.ModuleType("face_recognition")
    module.face_encodings = lambda frame: (
        (_ for _ in ()).throw(encode_error) if encode_error else (encodings or [])
    )
    module.face_distance = lambda known, candidate: [distance if distance is not None else 0.1]
    install_stub(monkeypatch, "face_recognition", module)
    return module


def test_marker_backend_rejects_direct_use():
    backend = FaceRecognitionBackend()

    with pytest.raises(NotImplementedError):
        backend.encode(object())

    with pytest.raises(NotImplementedError):
        backend.distance((0.1,), (0.2,))


def test_default_backend_reports_missing_model_data(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(DependencyError, match="model data is unavailable"):
        DefaultFaceRecognitionBackend()


def test_default_backend_reports_missing_dependencies(monkeypatch):
    install_stub(monkeypatch, "face_recognition_models", types.ModuleType("stub"))
    real_import = importlib.import_module

    def only_models(name, *args, **kwargs):
        if name == "face_recognition_models":
            return real_import(name, *args, **kwargs)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", only_models)

    with pytest.raises(DependencyError, match="dependencies are unavailable"):
        DefaultFaceRecognitionBackend()


def requires_numpy():
    """Skip a test that needs NumPy, which the core job does not install."""
    pytest.importorskip("numpy")


def test_default_backend_encodes_faces(monkeypatch):
    requires_numpy()
    install_stub(monkeypatch, "face_recognition_models", types.ModuleType("stub"))
    stub_recognition_module(monkeypatch, encodings=[[0.1, 0.2], [0.3, 0.4]])
    backend = DefaultFaceRecognitionBackend()

    result = backend.encode(object())

    assert result == [(0.1, 0.2), (0.3, 0.4)]


def test_default_backend_wraps_an_encode_failure(monkeypatch):
    requires_numpy()
    install_stub(monkeypatch, "face_recognition_models", types.ModuleType("stub"))
    stub_recognition_module(monkeypatch, encode_error=ZeroDivisionError())
    backend = DefaultFaceRecognitionBackend()

    with pytest.raises(RecognitionError, match="could not process the image"):
        backend.encode(object())


def test_default_backend_reports_an_invalid_encoding(monkeypatch):
    requires_numpy()
    install_stub(monkeypatch, "face_recognition_models", types.ModuleType("stub"))
    stub_recognition_module(monkeypatch, encode_error=ValueError("bad pixels"))
    backend = DefaultFaceRecognitionBackend()

    with pytest.raises(RecognitionError, match="Unable to encode the captured image"):
        backend.encode(object())


def test_default_backend_rejects_invalid_encodings(monkeypatch):
    requires_numpy()
    install_stub(monkeypatch, "face_recognition_models", types.ModuleType("stub"))
    stub_recognition_module(monkeypatch, encodings=[[0.1, float("nan")]])
    backend = DefaultFaceRecognitionBackend()

    with pytest.raises(RecognitionError, match="Unable to encode"):
        backend.encode(object())


def test_default_backend_computes_a_distance(monkeypatch):
    requires_numpy()
    install_stub(monkeypatch, "face_recognition_models", types.ModuleType("stub"))
    stub_recognition_module(monkeypatch, distance=0.42)
    backend = DefaultFaceRecognitionBackend()

    assert backend.distance((0.1, 0.2), (0.3, 0.4)) == pytest.approx(0.42)


def test_default_backend_wraps_a_distance_failure(monkeypatch):
    requires_numpy()
    install_stub(monkeypatch, "face_recognition_models", types.ModuleType("stub"))
    module = stub_recognition_module(monkeypatch)

    def exploding(known, candidate):
        raise RuntimeError("mismatch")

    module.face_distance = exploding
    backend = DefaultFaceRecognitionBackend()

    with pytest.raises(RecognitionError, match="Unable to compare"):
        backend.distance((0.1,), (0.2,))


def test_normalize_name_strips_and_normalizes():
    assert normalize_name("  Ada  ") == "Ada"


def test_normalize_name_applies_nfkc():
    fullwidth = "".join(chr(code) for code in (0xFF21, 0xFF44, 0xFF41))

    assert normalize_name(fullwidth) == "Ada"
    assert fullwidth != "Ada"


def test_normalize_name_rejects_non_text():
    with pytest.raises(RegistryError, match="must be text"):
        normalize_name(object())


def test_normalize_name_rejects_empty_input():
    with pytest.raises(RegistryError, match="cannot be empty"):
        normalize_name("   ")


def test_normalize_name_enforces_the_length_limit():
    with pytest.raises(RegistryError, match="cannot exceed"):
        normalize_name("a" * (MAX_NAME_LENGTH + 1))


def test_normalize_name_accepts_the_length_limit_boundary():
    assert len(normalize_name("a" * MAX_NAME_LENGTH)) == MAX_NAME_LENGTH


@pytest.mark.parametrize("name", [".", ".."])
def test_normalize_name_rejects_reserved_values(name):
    with pytest.raises(RegistryError, match="reserved"):
        normalize_name(name)


def test_normalize_name_rejects_control_characters():
    with pytest.raises(RegistryError, match="control characters"):
        normalize_name("Ada\x00")


def test_normalize_name_rejects_path_separators():
    with pytest.raises(RegistryError, match="path separators"):
        normalize_name("Ada/Bel")


def test_normalize_name_rejects_backslashes():
    with pytest.raises(RegistryError, match="path separators"):
        normalize_name("Ada\\Bel")


def test_name_key_folds_case():
    assert name_key("  ADA  ") == "ada"


def test_validate_embedding_converts_to_a_float_tuple():
    assert validate_embedding([1, 2, 3]) == (1.0, 2.0, 3.0)


def test_validate_embedding_accepts_any_iterable():
    assert validate_embedding(iter([1.0, 2.0])) == (1.0, 2.0)


@pytest.mark.parametrize("value", ["1,2", b"12", bytearray(b"12")])
def test_validate_embedding_rejects_text(value):
    with pytest.raises(RegistryError, match="sequence of numbers"):
        validate_embedding(value)


def test_validate_embedding_rejects_non_numbers():
    with pytest.raises(RegistryError, match="contain numbers"):
        validate_embedding([object()])


def test_validate_embedding_rejects_an_empty_sequence():
    with pytest.raises(RegistryError, match="cannot be empty"):
        validate_embedding([])


def test_validate_embedding_enforces_the_length_limit():
    with pytest.raises(RegistryError, match="too large"):
        validate_embedding([0.0] * (MAX_EMBEDDING_LENGTH + 1))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_embedding_rejects_non_finite_values(value):
    with pytest.raises(RegistryError, match="non-finite"):
        validate_embedding([value])


def test_validation_module_exposes_its_limits():
    assert validation.MAX_NAME_LENGTH == 80
    assert validation.MAX_EMBEDDING_LENGTH == 4096


def test_main_module_delegates_to_main(monkeypatch):
    module = importlib.import_module("face_attendance.__main__")

    assert callable(module.main)


def test_main_module_exits_with_the_main_result(monkeypatch):
    import runpy

    monkeypatch.setattr("face_attendance.main.main", lambda: 7)
    monkeypatch.setattr("sys.argv", ["face-attendance"])

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("face_attendance.__main__", run_name="__main__")

    assert caught.value.code == 7
