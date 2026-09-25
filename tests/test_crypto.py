"""At-rest registry encryption tests.

Encryption is a security control, so these tests assert properties rather than
just round-trips: that the plaintext is genuinely absent from the file, that a
wrong key is rejected rather than silently accepted, that tampering is
detected, and that a plaintext registry is never silently accepted by an
encrypted deployment.

Only the tests that perform real encryption request the ``crypto`` fixture,
which skips without the optional ``cryptography`` package. The rest of the
module -- cipher selection, key handling, and envelope validation -- is
dependency-free and is therefore covered by the dependency-free CI job.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import ModuleType

import pytest

from face_attendance.config import AppConfig
from face_attendance.crypto import (
    REGISTRY_CIPHERS,
    REGISTRY_KEY_ENV,
    PlaintextCipher,
    build_cipher,
    generate_key,
    read_registry_key,
)
from face_attendance.errors import ConfigurationError, RegistryError
from face_attendance.registry import FaceRegistry
from face_attendance.storage import create_embedding_store

ALPHA = (0.1, 0.2, 0.3)
BETA = (0.4, 0.5, 0.6)
CIPHERS = [cipher for cipher in REGISTRY_CIPHERS if cipher != "none"]


@pytest.fixture
def crypto() -> ModuleType:
    """Require the optional ``cryptography`` package."""
    return pytest.importorskip("cryptography", reason="requires the cryptography package")


@pytest.fixture(params=CIPHERS)
def cipher(request: pytest.FixtureRequest, crypto: ModuleType) -> str:
    """Parametrize a test over every cipher that requires a key."""
    return str(request.param)


@pytest.fixture
def key(cipher: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Generate a key valid for the cipher under test."""
    generated = generate_key(cipher)
    monkeypatch.setenv(REGISTRY_KEY_ENV, generated)
    return generated


def registry_path(tmp_path: Path) -> Path:
    """Return the registry path for a test.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The registry file path.
    """
    return tmp_path / "registry.json"


def build(key_value: str, cipher: str, path: Path) -> FaceRegistry:
    """Return a registry using a specific cipher and key.

    Args:
        key_value: The encryption key.
        cipher: The cipher name.
        path: The registry file path.

    Returns:
        The configured registry.
    """
    return FaceRegistry(path, cipher=build_cipher(cipher, key_value))


def payload_bytes(payload: str) -> bytes:
    """Decode a stored envelope payload to raw bytes.

    Args:
        payload: The stored ciphertext.

    Returns:
        The decoded bytes.
    """
    return base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))


def decode_payload(payload: str) -> str:
    """Best-effort decode of a stored envelope payload to text.

    Fernet tokens are URL-safe base64 and AES-GCM payloads are standard
    base64; URL-safe decoding handles both alphabets.

    Args:
        payload: The stored ciphertext.

    Returns:
        The decoded text, with undecodable bytes replaced.
    """
    return payload_bytes(payload).decode("utf-8", "replace")


def corrupt_last_byte(document: dict[str, object]) -> None:
    """Flip a bit in the final byte of a stored payload.

    Args:
        document: The registry document, modified in place.
    """
    raw = bytearray(payload_bytes(str(document["payload"])))
    raw[-1] ^= 0x01
    document["payload"] = base64.urlsafe_b64encode(bytes(raw)).decode("ascii")


def test_the_default_cipher_is_plaintext() -> None:
    assert AppConfig().registry_cipher == "none"


def test_the_plaintext_cipher_writes_no_envelope() -> None:
    cipher = PlaintextCipher()
    assert cipher.name is None
    assert cipher.encrypt('{"version": 1}') == {}


def test_the_plaintext_cipher_refuses_to_decrypt() -> None:
    with pytest.raises(RegistryError):
        PlaintextCipher().decrypt({"payload": "x"})


def test_a_plaintext_registry_is_unchanged_on_disk(tmp_path: Path) -> None:
    path = registry_path(tmp_path)
    FaceRegistry(path).register("Ada", ALPHA)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {"version": 1, "users": [{"name": "Ada", "embeddings": [list(ALPHA)]}]}


def test_an_encrypted_registry_round_trips(tmp_path: Path, cipher: str, key: str) -> None:
    path = registry_path(tmp_path)
    store = build(key, cipher, path)
    assert store.register("Ada", ALPHA) is True
    assert store.register("Ada", BETA) is True
    reloaded = build(key, cipher, path)
    record = reloaded.get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA, BETA)
    assert reloaded.names() == ("Ada",)


def test_the_plaintext_is_absent_from_the_file(tmp_path: Path, cipher: str, key: str) -> None:
    path = registry_path(tmp_path)
    build(key, cipher, path).register("Ada", ALPHA)
    raw = path.read_text(encoding="utf-8")
    assert "Ada" not in raw
    assert "embeddings" not in raw
    document = json.loads(raw)
    assert document["encrypted"] == cipher
    assert document["version"] == 1
    assert "Ada" not in decode_payload(str(document["payload"]))


def test_a_wrong_key_is_rejected(tmp_path: Path, cipher: str, key: str) -> None:
    path = registry_path(tmp_path)
    build(key, cipher, path).register("Ada", ALPHA)
    other = generate_key(cipher)
    with pytest.raises(RegistryError) as error:
        build(other, cipher, path).get("Ada")
    assert "decrypt" in str(error.value).lower()


def test_removal_and_lookup_still_work_when_encrypted(
    tmp_path: Path, cipher: str, key: str
) -> None:
    path = registry_path(tmp_path)
    store = build(key, cipher, path)
    store.register("Ada", ALPHA)
    store.register("Grace", BETA)
    assert store.names() == ("Ada", "Grace")
    assert store.remove("Ada") is True
    assert store.get("Ada") is None
    assert store.names() == ("Grace",)
    assert store.remove("Ada") is False


def test_an_encrypted_registry_enforces_the_sample_cap(
    tmp_path: Path, cipher: str, key: str
) -> None:
    path = registry_path(tmp_path)
    FaceRegistry(path, max_embeddings_per_user=1, cipher=build_cipher(cipher, key)).register(
        "Ada", ALPHA
    )
    capped = FaceRegistry(path, max_embeddings_per_user=1, cipher=build_cipher(cipher, key))
    assert capped.register("Ada", BETA) is False


def test_an_empty_encrypted_registry_reads_as_empty(tmp_path: Path, cipher: str, key: str) -> None:
    path = registry_path(tmp_path)
    build(key, cipher, path).ensure_exists()
    reloaded = build(key, cipher, path)
    assert reloaded.names() == ()
    assert reloaded.get("Nobody") is None


def test_aes_gcm_stores_a_fresh_nonce_per_write(tmp_path: Path, crypto: ModuleType) -> None:
    key = generate_key("aes-gcm")
    path = registry_path(tmp_path)
    store = build(key, "aes-gcm", path)
    store.register("Ada", ALPHA)
    first = json.loads(path.read_text(encoding="utf-8"))["nonce"]
    store.register("Ada", BETA)
    second = json.loads(path.read_text(encoding="utf-8"))["nonce"]
    assert first != second


def test_tampering_with_the_ciphertext_is_detected(tmp_path: Path, cipher: str, key: str) -> None:
    path = registry_path(tmp_path)
    build(key, cipher, path).register("Ada", ALPHA)
    document = json.loads(path.read_text(encoding="utf-8"))
    corrupt_last_byte(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryError):
        build(key, cipher, path).get("Ada")


def test_tampering_with_the_nonce_is_detected(tmp_path: Path, crypto: ModuleType) -> None:
    key = generate_key("aes-gcm")
    path = registry_path(tmp_path)
    build(key, "aes-gcm", path).register("Ada", ALPHA)
    document = json.loads(path.read_text(encoding="utf-8"))
    nonce = bytearray(base64.b64decode(str(document["nonce"])))
    nonce[0] ^= 0x01
    document["nonce"] = base64.b64encode(bytes(nonce)).decode("ascii")
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryError):
        build(key, "aes-gcm", path).get("Ada")


def test_an_envelope_for_a_different_cipher_is_rejected(tmp_path: Path, crypto: ModuleType) -> None:
    path = registry_path(tmp_path)
    build(generate_key("aes-gcm"), "aes-gcm", path).register("Ada", ALPHA)
    with pytest.raises(RegistryError) as error:
        build(generate_key("fernet"), "fernet", path).get("Ada")
    assert "written with" in str(error.value)


def test_an_encrypted_registry_needs_a_cipher(tmp_path: Path, crypto: ModuleType) -> None:
    path = registry_path(tmp_path)
    build(generate_key("aes-gcm"), "aes-gcm", path).register("Ada", ALPHA)
    with pytest.raises(RegistryError):
        FaceRegistry(path).get("Ada")


def test_a_plaintext_registry_is_rejected_by_a_cipher(tmp_path: Path, crypto: ModuleType) -> None:
    path = registry_path(tmp_path)
    FaceRegistry(path).register("Ada", ALPHA)
    with pytest.raises(RegistryError) as error:
        build(generate_key("aes-gcm"), "aes-gcm", path).get("Ada")
    assert "not encrypted" in str(error.value)


def test_an_unsupported_envelope_field_is_rejected(tmp_path: Path, cipher: str, key: str) -> None:
    path = registry_path(tmp_path)
    build(key, cipher, path).register("Ada", ALPHA)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["surprise"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryError):
        build(key, cipher, path).get("Ada")


def test_a_tampering_rejection_does_not_leak_secrets(tmp_path: Path, cipher: str, key: str) -> None:
    path = registry_path(tmp_path)
    build(key, cipher, path).register("Ada", ALPHA)
    document = json.loads(path.read_text(encoding="utf-8"))
    corrupt_last_byte(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryError) as error:
        build(key, cipher, path).get("Ada")
    assert str(document["payload"]) not in str(error.value)
    assert key not in str(error.value)


def test_a_missing_payload_is_reported(key: str) -> None:
    cipher = build_cipher("aes-gcm", key)
    with pytest.raises(RegistryError) as error:
        cipher.decrypt({"nonce": "aaaa"})
    assert "payload" in str(error.value)


def test_a_missing_nonce_is_reported(key: str) -> None:
    cipher = build_cipher("aes-gcm", key)
    with pytest.raises(RegistryError) as error:
        cipher.decrypt({"payload": "aaaa"})
    assert "nonce" in str(error.value)


def test_a_non_base64_nonce_is_reported(key: str) -> None:
    cipher = build_cipher("aes-gcm", key)
    with pytest.raises(RegistryError):
        cipher.decrypt({"nonce": "!!!", "payload": "aaaa"})


def test_an_empty_key_is_rejected() -> None:
    with pytest.raises(RegistryError) as error:
        build_cipher("aes-gcm", "   ")
    assert REGISTRY_KEY_ENV in str(error.value)


def test_a_missing_environment_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REGISTRY_KEY_ENV, raising=False)
    with pytest.raises(RegistryError) as error:
        build_cipher("aes-gcm")
    assert REGISTRY_KEY_ENV in str(error.value)


def test_a_blank_environment_key_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REGISTRY_KEY_ENV, "   ")
    assert read_registry_key() is None
    with pytest.raises(RegistryError):
        build_cipher("aes-gcm")


def test_a_key_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, crypto: ModuleType
) -> None:
    generated = generate_key("aes-gcm")
    monkeypatch.setenv(REGISTRY_KEY_ENV, generated)
    assert read_registry_key() == generated
    assert build_cipher("aes-gcm").name == "aes-gcm"


def test_an_unknown_cipher_is_reported() -> None:
    with pytest.raises(RegistryError) as error:
        build_cipher("rot13", "secret")
    assert "rot13" in str(error.value)


def test_cipher_names_are_matched_case_insensitively(key: str) -> None:
    assert build_cipher("AES-GCM", key).name == "aes-gcm"
    assert build_cipher("  fernet  ", key).name == "fernet"
    assert build_cipher("NONE").name is None


def test_generated_keys_are_usable_and_distinct(crypto: ModuleType) -> None:
    for cipher in CIPHERS:
        first = generate_key(cipher)
        second = generate_key(cipher)
        assert first != second
        assert build_cipher(cipher, first).name == cipher


def test_the_none_cipher_generates_no_key() -> None:
    with pytest.raises(RegistryError):
        generate_key("none")


def test_key_generation_rejects_an_unknown_cipher() -> None:
    with pytest.raises(RegistryError):
        generate_key("rot13")


def test_a_key_of_exactly_32_bytes_is_used_verbatim(crypto: ModuleType) -> None:
    raw = bytes(range(32))
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    assert build_cipher("aes-gcm", encoded).name == "aes-gcm"


def test_a_weak_short_secret_is_refused(crypto: ModuleType) -> None:
    with pytest.raises(RegistryError) as error:
        build_cipher("aes-gcm", "correct horse battery")
    assert "32 characters" in str(error.value)


def test_a_long_enough_secret_is_accepted(crypto: ModuleType) -> None:
    secret = "correct horse battery staple plus padding"
    assert len(secret) >= 32
    cipher = build_cipher("aes-gcm", secret)
    assert cipher.decrypt(cipher.encrypt('{"version": 1}')) == '{"version": 1}'


def test_generated_secrets_satisfy_the_minimum_length(crypto: ModuleType) -> None:
    for cipher in CIPHERS:
        assert len(generate_key(cipher)) >= 32


def test_missing_cryptography_reports_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import face_attendance.crypto as crypto

    def refuse(name: str) -> object:
        raise ImportError(f"no module named {name}")

    monkeypatch.setattr(crypto.importlib, "import_module", refuse)
    with pytest.raises(RegistryError) as error:
        crypto.FernetCipher("key")
    assert "face-attendance[crypto]" in str(error.value)


def test_a_decrypted_payload_that_is_not_json_is_reported(tmp_path: Path) -> None:
    class NotJson:
        name: str | None = "none"

        def encrypt(self, plaintext: str) -> dict[str, str]:
            return {}

        def decrypt(self, envelope: dict[str, object]) -> str:
            return "not json at all"

    path = registry_path(tmp_path)
    FaceRegistry(path, cipher=NotJson()).register("Ada", ALPHA)
    with pytest.raises(RegistryError) as error:
        FaceRegistry(path, cipher=NotJson()).get("Ada")
    assert "not encrypted" not in str(error.value)


def test_the_config_accepts_a_cipher_name() -> None:
    assert AppConfig(registry_cipher="aes-gcm").registry_cipher == "aes-gcm"
    with pytest.raises(ConfigurationError):
        AppConfig(registry_cipher="rot13")
    with pytest.raises(ConfigurationError):
        AppConfig.from_file(None, environ={"FACE_ATTENDANCE_REGISTRY_CIPHER": "rot13"})


def test_the_config_accepts_a_cipher_from_the_environment() -> None:
    config = AppConfig.from_file(None, environ={"FACE_ATTENDANCE_REGISTRY_CIPHER": "fernet"})
    assert config.registry_cipher == "fernet"


def test_the_factory_applies_the_configured_cipher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crypto: ModuleType
) -> None:
    monkeypatch.setenv(REGISTRY_KEY_ENV, generate_key("aes-gcm"))
    config = AppConfig(storage_backend="json", registry_cipher="aes-gcm")
    path = tmp_path / "registry.json"
    store = create_embedding_store(config, path)
    assert isinstance(store, FaceRegistry)
    store.register("Ada", ALPHA)
    assert "Ada" not in path.read_text(encoding="utf-8")
    assert create_embedding_store(config, path).get("Ada") is not None


def test_the_factory_defaults_to_a_plaintext_registry(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    create_embedding_store(AppConfig(), path).register("Ada", ALPHA)
    assert "Ada" in path.read_text(encoding="utf-8")


def test_the_factory_leaves_sqlite_unencrypted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crypto: ModuleType
) -> None:
    monkeypatch.setenv(REGISTRY_KEY_ENV, generate_key("aes-gcm"))
    config = AppConfig(storage_backend="sqlite", registry_cipher="aes-gcm")
    store = create_embedding_store(config, tmp_path / "faces.db")
    store.register("Ada", ALPHA)
    assert store.get("Ada") is not None


def test_iterating_an_encrypted_registry_yields_every_sample(
    tmp_path: Path, cipher: str, key: str
) -> None:
    path = registry_path(tmp_path)
    store = build(key, cipher, path)
    store.register("Ada", ALPHA)
    store.register("Ada", BETA)
    store.register("Grace", (0.9, 0.8, 0.7))
    assert len(list(store.iter_embeddings())) == 3
    assert len(store.records()) == 2
    assert len(store) == 2
