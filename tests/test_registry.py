import json

import pytest

from face_attendance.errors import RegistryError
from face_attendance.registry import FaceRegistry


def test_registry_persists_multiple_samples(tmp_path):
    registry = FaceRegistry(tmp_path / "registry.json", max_embeddings_per_user=2)

    assert registry.register("Ada", [0.1, 0.2]) is True
    assert registry.register("Ada", [0.3, 0.4]) is True
    assert registry.register("Ada", [0.5, 0.6]) is False
    assert registry.names() == ("Ada",)
    assert len(registry.get("Ada").embeddings) == 2

    reloaded = FaceRegistry(tmp_path / "registry.json")
    assert reloaded.get("Ada").embeddings == ((0.1, 0.2), (0.3, 0.4))
    payload = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["users"][0]["name"] == "Ada"


def test_registry_deduplicates_an_exact_sample(tmp_path):
    registry = FaceRegistry(tmp_path / "registry.json")

    assert registry.register("Ada", [0.1, 0.2]) is True
    assert registry.register("Ada", [0.1, 0.2]) is False
    assert len(registry.get("Ada").embeddings) == 1


@pytest.mark.parametrize("name", ["", "   ", "../Ada", "Ada\\Lovelace", "Ada\nLovelace"])
def test_registry_rejects_unsafe_names(tmp_path, name):
    registry = FaceRegistry(tmp_path / "registry.json")

    with pytest.raises(RegistryError):
        registry.register(name, [0.1, 0.2])


def test_registry_rejects_non_finite_embeddings(tmp_path):
    registry = FaceRegistry(tmp_path / "registry.json")

    with pytest.raises(RegistryError):
        registry.register("Ada", [0.1, float("nan")])


def test_registry_rejects_unsupported_files(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"version": 99, "users": []}), encoding="utf-8")

    with pytest.raises(RegistryError):
        FaceRegistry(path).names()


def test_registry_can_remove_users(tmp_path):
    registry = FaceRegistry(tmp_path / "registry.json")
    registry.register("Ada", [0.1, 0.2])

    assert registry.remove("Ada") is True
    assert registry.remove("Ada") is False
    assert registry.names() == ()


def test_failed_save_does_not_change_persisted_users(tmp_path, monkeypatch):
    registry = FaceRegistry(tmp_path / "registry.json")
    registry.register("Ada", [0.1, 0.2])

    def fail_save(users):
        raise RegistryError("disk full")

    monkeypatch.setattr(registry, "_save", fail_save)
    with pytest.raises(RegistryError):
        registry.register("Ada", [0.3, 0.4])

    assert registry.get("Ada").embeddings == ((0.1, 0.2),)
