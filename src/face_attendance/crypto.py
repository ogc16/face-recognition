"""Optional at-rest encryption for the embedding registry.

Face embeddings are biometric data, so the bundled registry format stores them
as plaintext JSON by default for backwards compatibility. Setting a cipher
switches the registry to an encrypted envelope, in which the whole payload is
encrypted and only the algorithm name and ciphertext are visible.

Supported ciphers:

``none``
    Plaintext. The default, and the only option that needs no third-party
    package.
``fernet``
    Fernet from :mod:`cryptography`. Authenticated, and includes a creation
    timestamp so old ciphertexts can be rejected.
``aes-gcm``
    AES-256-GCM from :mod:`cryptography`. Authenticated, with an explicit
    random nonce per write.

:class:`cryptography` is an optional dependency, imported lazily so the
dependency-free core keeps working.

Security notes
--------------
The key is read from the ``FACE_ATTENDANCE_REGISTRY_KEY`` environment variable
and is deliberately *not* a configuration-file field: configuration files are
routinely shared, committed, and copied between machines, and a key in such a
file provides no protection. Use a secret manager or a per-user environment
variable in production.

Encryption is authenticated, so a modified registry is rejected rather than
silently decrypted to garbage. The error messages never include the key or any
ciphertext.
"""

from __future__ import annotations

import base64
import importlib
import os
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from .errors import RegistryError

if TYPE_CHECKING:
    from types import ModuleType

__all__ = [
    "REGISTRY_CIPHERS",
    "REGISTRY_KEY_ENV",
    "AesGcmCipher",
    "FernetCipher",
    "PlaintextCipher",
    "RegistryCipher",
    "build_cipher",
    "generate_key",
    "read_registry_key",
]

#: The environment variable holding the registry key.
REGISTRY_KEY_ENV = "FACE_ATTENDANCE_REGISTRY_KEY"

#: The supported ciphers, in the order they are documented.
REGISTRY_CIPHERS = ("none", "fernet", "aes-gcm")

_AES_GCM_NONCE_BYTES = 12
_AES_GCM_KEY_BYTES = 32
_FERNET_KEY_BYTES = 32
_AES_GCM_MIN_KEY_CHARS = 32


class RegistryCipher(Protocol):
    """Encrypts and decrypts a registry payload.

    Attributes:
        name: The envelope's ``encrypted`` value, or ``None`` for plaintext.
    """

    name: str | None

    def encrypt(self, plaintext: str) -> dict[str, str]:
        """Wrap a serialized registry.

        Args:
            plaintext: The serialized JSON registry.

        Returns:
            The envelope fields to store alongside ``version``. Empty for a
            plaintext cipher.
        """

    def decrypt(self, envelope: Mapping[str, object]) -> str:
        """Recover a serialized registry from its envelope.

        Args:
            envelope: The stored envelope fields.

        Returns:
            The original serialized JSON registry.

        Raises:
            RegistryError: If the envelope is malformed, the cipher is
                unavailable, or authentication fails.
        """


class PlaintextCipher:
    """Stores the registry unencrypted, as previous versions did."""

    name: str | None = None

    def encrypt(self, plaintext: str) -> dict[str, str]:
        """Return no envelope fields.

        Args:
            plaintext: Ignored.

        Returns:
            An empty mapping.
        """
        return {}

    def decrypt(self, envelope: Mapping[str, object]) -> str:
        """Return no plaintext, because there is no envelope.

        Args:
            envelope: Ignored.

        Raises:
            RegistryError: Always; a plaintext registry has no envelope.
        """
        raise RegistryError("This registry is not encrypted")


def _load_cryptography() -> ModuleType:
    """Import :mod:`cryptography.fernet` or its AES primitives on demand.

    Returns:
        The imported module.

    Raises:
        RegistryError: If ``cryptography`` is not installed.
    """
    try:
        module: ModuleType = importlib.import_module("cryptography.fernet")
    except ImportError as exc:
        raise RegistryError(
            "Registry encryption requires the 'cryptography' package. "
            'Install it with: pip install "face-attendance[crypto]"'
        ) from exc
    return module


class FernetCipher:
    """Fernet-encrypted registry payloads.

    Attributes:
        name: Always ``"fernet"``.
    """

    name: str | None = "fernet"

    def __init__(self, key: str) -> None:
        """Validate and wrap a Fernet key.

        Args:
            key: A url-safe base64-encoded 32-byte Fernet key.

        Raises:
            RegistryError: If the key is not a valid Fernet key.
        """
        self._fernet = _load_cryptography().Fernet(_require_key(key))

    def encrypt(self, plaintext: str) -> dict[str, str]:
        """Encrypt the registry.

        Args:
            plaintext: The serialized JSON registry.

        Returns:
            The envelope, holding the Fernet ciphertext as ASCII text.
        """
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return {"payload": cast(bytes, token).decode("ascii")}

    def decrypt(self, envelope: Mapping[str, object]) -> str:
        """Decrypt the registry.

        Args:
            envelope: The stored envelope fields.

        Returns:
            The original serialized JSON registry.

        Raises:
            RegistryError: If the envelope is malformed or authentication
                fails, which includes a wrong key.
        """
        payload = _require_payload(envelope)
        try:
            return cast(bytes, self._fernet.decrypt(payload.encode("ascii"))).decode("utf-8")
        except Exception as exc:
            raise RegistryError(
                "Unable to decrypt registry: wrong key, or the file was modified"
            ) from exc


class AesGcmCipher:
    """AES-256-GCM encrypted registry payloads.

    Attributes:
        name: Always ``"aes-gcm"``.
    """

    name: str | None = "aes-gcm"

    def __init__(self, key: str) -> None:
        """Derive an AES-256 key from the configured secret.

        The secret must be long enough to carry real entropy, because a
        shorter one is stretched with a single hash pass, which is fast enough
        to brute force offline. Generate a suitable secret with
        :func:`generate_key`.

        Args:
            key: The registry secret.

        Raises:
            RegistryError: If the key is empty, too short, or unusable.
        """
        secret = _require_key(key)
        if len(secret) < _AES_GCM_MIN_KEY_CHARS:
            raise RegistryError(
                f"{REGISTRY_KEY_ENV} must be at least {_AES_GCM_MIN_KEY_CHARS} characters "
                f"for aes-gcm. Generate one with face_attendance.crypto.generate_key()"
            )
        material = secret.encode("utf-8")
        self._key = material if len(material) == _AES_GCM_KEY_BYTES else _sha256(material)
        self._aead = _load_aesgcm()(self._key)

    def encrypt(self, plaintext: str) -> dict[str, str]:
        """Encrypt the registry with a fresh random nonce.

        Args:
            plaintext: The serialized JSON registry.

        Returns:
            The envelope, holding a base64 nonce and ciphertext.

        Raises:
            RegistryError: If encryption fails.
        """
        try:
            nonce = secrets.token_bytes(_AES_GCM_NONCE_BYTES)
            ciphertext = self._aead.encrypt(nonce, plaintext.encode("utf-8"), None)
        except Exception as exc:
            raise RegistryError("Unable to encrypt registry") from exc
        return {
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "payload": base64.b64encode(ciphertext).decode("ascii"),
        }

    def decrypt(self, envelope: Mapping[str, object]) -> str:
        """Decrypt the registry.

        Args:
            envelope: The stored envelope fields.

        Returns:
            The original serialized JSON registry.

        Raises:
            RegistryError: If the envelope is malformed or authentication
                fails, which includes a wrong key.
        """
        nonce_field = envelope.get("nonce")
        if not isinstance(nonce_field, str) or not nonce_field:
            raise RegistryError("Encrypted registry is missing its nonce")
        payload = _require_payload(envelope)
        try:
            nonce = base64.b64decode(nonce_field, validate=True)
            plaintext = self._aead.decrypt(nonce, base64.b64decode(payload, validate=True), None)
        except Exception as exc:
            raise RegistryError(
                "Unable to decrypt registry: wrong key, or the file was modified"
            ) from exc
        return plaintext.decode("utf-8")


def _sha256(material: bytes) -> bytes:
    """Hash a secret to a fixed-length AES key.

    Args:
        material: The secret bytes.

    Returns:
        The 32-byte digest.
    """
    import hashlib

    return hashlib.sha256(material).digest()


class _Aead(Protocol):
    """The subset of an AES-GCM implementation this module uses."""

    def encrypt(self, nonce: bytes, data: bytes, associated_data: bytes | None) -> bytes: ...

    def decrypt(self, nonce: bytes, data: bytes, associated_data: bytes | None) -> bytes: ...


#: The AES-GCM constructor: it takes a key and returns an AEAD.
_AeadFactory: TypeAlias = Callable[[bytes], _Aead]


def _load_aesgcm() -> _AeadFactory:
    """Load the ``AESGCM`` class from :mod:`cryptography`.

    Returns:
        The ``AESGCM`` class, which is called with a key to construct an AEAD.

    Raises:
        RegistryError: If ``cryptography`` is not installed.
    """
    _load_cryptography()
    try:
        ciphers = importlib.import_module("cryptography.hazmat.primitives.ciphers.aead")
    except ImportError as exc:
        raise RegistryError(
            "Registry encryption requires the 'cryptography' package. "
            'Install it with: pip install "face-attendance[crypto]"'
        ) from exc
    return cast(_AeadFactory, ciphers.AESGCM)


def _require_key(key: str) -> str:
    """Return a non-empty key.

    Args:
        key: The configured key.

    Returns:
        The key.

    Raises:
        RegistryError: If the key is empty or not text.
    """
    if not isinstance(key, str) or not key.strip():
        raise RegistryError(
            f"Registry encryption requires a key in the {REGISTRY_KEY_ENV} environment variable"
        )
    return key.strip()


def _require_payload(envelope: Mapping[str, object]) -> str:
    """Return the ciphertext from an envelope.

    Args:
        envelope: The stored envelope fields.

    Returns:
        The ciphertext.

    Raises:
        RegistryError: If the ciphertext is missing or not text.
    """
    payload = envelope.get("payload")
    if not isinstance(payload, str) or not payload:
        raise RegistryError("Encrypted registry is missing its payload")
    return payload


def generate_key(cipher: str) -> str:
    """Generate a new key suitable for a cipher.

    Args:
        cipher: One of :data:`REGISTRY_CIPHERS`, excluding ``none``.

    Returns:
        A new key, as a string suitable for the environment variable.

    Raises:
        RegistryError: If the cipher is ``none`` or unknown, or the optional
            ``cryptography`` package is unavailable.
    """
    normalized = cipher.strip().lower()
    if normalized == "none":
        raise RegistryError("The 'none' cipher does not use a key")
    if normalized == "fernet":
        _load_cryptography()
        return base64.urlsafe_b64encode(secrets.token_bytes(_FERNET_KEY_BYTES)).decode("ascii")
    if normalized == "aes-gcm":
        return base64.urlsafe_b64encode(secrets.token_bytes(_AES_GCM_KEY_BYTES)).decode("ascii")
    supported = ", ".join(REGISTRY_CIPHERS)
    raise RegistryError(f"Unknown registry cipher {cipher!r}. Expected one of: {supported}")


def read_registry_key(environ: Mapping[str, str] | None = None) -> str | None:
    """Read the registry key from the environment.

    Args:
        environ: The environment mapping. Defaults to :data:`os.environ`.

    Returns:
        The key, or ``None`` when unset or empty.
    """
    environment = os.environ if environ is None else environ
    value = environment.get(REGISTRY_KEY_ENV)
    if value is None or not value.strip():
        return None
    return value.strip()


class _CipherFactory(ABC):
    """Builds a cipher from a key. Exists to keep the switch exhaustive."""

    @abstractmethod
    def build(self, key: str) -> RegistryCipher:
        """Return the cipher for a key."""


class _FernetFactory(_CipherFactory):
    """Builds :class:`FernetCipher` instances."""

    def build(self, key: str) -> RegistryCipher:
        """Return a Fernet cipher.

        Args:
            key: The Fernet key.

        Returns:
            The cipher.
        """
        return FernetCipher(key)


class _AesGcmFactory(_CipherFactory):
    """Builds :class:`AesGcmCipher` instances."""

    def build(self, key: str) -> RegistryCipher:
        """Return an AES-GCM cipher.

        Args:
            key: The registry secret.

        Returns:
            The cipher.
        """
        return AesGcmCipher(key)


_FACTORIES: dict[str, _CipherFactory] = {
    "fernet": _FernetFactory(),
    "aes-gcm": _AesGcmFactory(),
}


def build_cipher(
    cipher: str,
    key: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> RegistryCipher:
    """Return the cipher named by ``cipher``.

    Args:
        cipher: One of :data:`REGISTRY_CIPHERS`, matched case-insensitively.
        key: The key. When ``None``, it is read from ``environ`` (defaulting
            to :data:`os.environ`) under :data:`REGISTRY_KEY_ENV`.
        environ: An explicit environment mapping, for testing.

    Returns:
        The requested :class:`RegistryCipher`.

    Raises:
        RegistryError: If the cipher is unknown, or a key is required but
            absent.
    """
    normalized = cipher.strip().lower()
    if normalized == "none":
        return PlaintextCipher()
    factory = _FACTORIES.get(normalized)
    if factory is None:
        supported = ", ".join(REGISTRY_CIPHERS)
        raise RegistryError(f"Unknown registry cipher {cipher!r}. Expected one of: {supported}")
    resolved = key if key is not None else read_registry_key(environ)
    if resolved is None:
        raise RegistryError(
            f"Registry cipher {normalized!r} requires a key in the "
            f"{REGISTRY_KEY_ENV} environment variable"
        )
    return factory.build(resolved)
