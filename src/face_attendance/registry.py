import contextlib
import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from .crypto import PlaintextCipher, RegistryCipher
from .errors import RegistryError
from .file_lock import exclusive_file_lock
from .types import Embedding, UserRecord
from .validation import name_key, normalize_name, validate_embedding

__all__ = ["REGISTRY_VERSION", "Embedding", "FaceRegistry", "UserRecord"]

REGISTRY_VERSION = 1
FileSignature = tuple[int, int, int]


class FaceRegistry:
    """The bundled JSON registry, optionally encrypted at rest.

    Args:
        path: Location of the registry file.
        max_embeddings_per_user: Maximum samples retained per identity.
        cipher: The at-rest cipher. Defaults to plaintext, which keeps the
            on-disk format identical to previous versions.

    Raises:
        RegistryError: If ``max_embeddings_per_user`` is not a positive
            integer, or the cipher cannot be constructed.
    """

    def __init__(
        self,
        path: str | Path,
        max_embeddings_per_user: int = 5,
        cipher: RegistryCipher | None = None,
    ) -> None:
        if isinstance(max_embeddings_per_user, bool) or not isinstance(
            max_embeddings_per_user, int
        ):
            raise RegistryError("Maximum embeddings per user must be an integer")
        if max_embeddings_per_user < 1:
            raise RegistryError("At least one embedding per user is required")
        self.path = Path(path)
        self.max_embeddings_per_user = max_embeddings_per_user
        self._cipher = PlaintextCipher() if cipher is None else cipher
        self._lock = RLock()
        self._cached_users: dict[str, UserRecord] | None = None
        self._cached_signature: FileSignature | None = None

    @contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            with self._lock, exclusive_file_lock(self.path):
                yield
        except OSError as exc:
            raise RegistryError(f"Unable to access registry: {self.path}") from exc

    def register(self, name: str, embedding: Embedding | Sequence[float]) -> bool:
        normalized_name = normalize_name(name)
        normalized_embedding = validate_embedding(embedding)
        with self._locked():
            users = self._load()
            key = name_key(normalized_name)
            existing = users.get(key)
            if existing is not None and normalized_embedding in existing.embeddings:
                return False
            if any(
                normalized_embedding in record.embeddings
                for other_key, record in users.items()
                if other_key != key
            ):
                raise RegistryError("Face sample is already registered to another user")
            if existing is None:
                users[key] = UserRecord(name=normalized_name, embeddings=(normalized_embedding,))
            elif len(existing.embeddings) < self.max_embeddings_per_user:
                users[key] = UserRecord(
                    name=existing.name,
                    embeddings=(*existing.embeddings, normalized_embedding),
                )
            else:
                return False
            self._save(users)
            return True

    def remove(self, name: str) -> bool:
        normalized_name = normalize_name(name)
        with self._locked():
            users = self._load()
            key = name_key(normalized_name)
            if key not in users:
                return False
            del users[key]
            self._save(users)
            return True

    def get(self, name: str) -> UserRecord | None:
        normalized_name = normalize_name(name)
        with self._locked():
            return self._load().get(name_key(normalized_name))

    def names(self) -> tuple[str, ...]:
        with self._locked():
            users = self._load()
            return tuple(sorted((record.name for record in users.values()), key=name_key))

    def records(self) -> tuple[UserRecord, ...]:
        with self._locked():
            users = self._load()
            return tuple(users[key] for key in sorted(users))

    def iter_embeddings(self) -> Iterator[tuple[str, Embedding]]:
        with self._locked():
            users = self._load()
            snapshots = tuple(
                (users[key].name, embedding)
                for key in sorted(users)
                for embedding in users[key].embeddings
            )
        yield from snapshots

    def ensure_exists(self) -> None:
        with self._locked():
            if not self.path.exists():
                self._save({})
            else:
                self._load()

    def __len__(self) -> int:
        with self._locked():
            return len(self._load())

    def _load(self) -> dict[str, UserRecord]:
        signature = self._file_signature()
        if self._cached_users is not None and signature == self._cached_signature:
            return dict(self._cached_users)
        users = self._read()
        self._cached_users = users
        self._cached_signature = signature
        return dict(users)

    def _read(self) -> dict[str, UserRecord]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RegistryError(f"Unable to read registry: {self.path}") from exc
        if not isinstance(payload, dict):
            raise RegistryError("Unsupported registry format")
        version = payload.get("version")
        if type(version) is not int or version != REGISTRY_VERSION:
            raise RegistryError("Unsupported registry format")
        if "encrypted" in payload:
            users = self._read_encrypted(payload)
        else:
            if self._cipher.name is not None:
                raise RegistryError(
                    f"Registry is not encrypted, but cipher {self._cipher.name!r} is "
                    "configured. Set registry_cipher to 'none' to use this file, or "
                    "re-enrol the identities to create an encrypted registry."
                )
            users = self._decode_users(payload)
        return users

    def _read_encrypted(self, payload: dict[str, object]) -> dict[str, UserRecord]:
        """Decrypt an encrypted registry envelope and decode its payload.

        Args:
            payload: The full stored document.

        Returns:
            The decoded user records.

        Raises:
            RegistryError: If the envelope names an unexpected cipher, the
                cipher is unavailable, or decryption fails.
        """
        algorithm = payload.get("encrypted")
        if algorithm != self._cipher.name:
            expected = self._cipher.name or "an unencrypted registry"
            raise RegistryError(
                f"Registry was written with {algorithm!r} but this deployment expects {expected}"
            )
        allowed = {"version", "encrypted", "nonce", "payload"}
        if not set(payload) <= allowed:
            raise RegistryError("Registry contains unsupported fields")
        if self._cipher.name is None:
            raise RegistryError("Registry is encrypted but no cipher is configured")
        envelope = {
            key: value for key, value in payload.items() if key not in {"version", "encrypted"}
        }
        plaintext = self._cipher.decrypt(envelope)
        try:
            inner = json.loads(plaintext)
        except json.JSONDecodeError as exc:
            raise RegistryError("Decrypted registry is not valid JSON") from exc
        if not isinstance(inner, dict):
            raise RegistryError("Unsupported registry format")
        return self._decode_users(inner)

    def _decode_users(self, payload: dict[str, object]) -> dict[str, UserRecord]:
        """Validate a plaintext registry document and return its records.

        Args:
            payload: The document, carrying ``version`` and ``users``.

        Returns:
            The decoded user records, keyed by case-folded name.

        Raises:
            RegistryError: If the document is structurally invalid or violates
                a registry invariant.
        """
        if set(payload) != {"version", "users"}:
            raise RegistryError("Registry contains unsupported fields")
        raw_users = payload.get("users")
        if not isinstance(raw_users, list):
            raise RegistryError("Registry users must be a list")
        users: dict[str, UserRecord] = {}
        seen_embeddings: dict[Embedding, str] = {}
        for raw_user in raw_users:
            if not isinstance(raw_user, dict) or set(raw_user) != {"name", "embeddings"}:
                raise RegistryError("Registry user data is invalid")
            raw_name = raw_user["name"]
            raw_embeddings = raw_user["embeddings"]
            if not isinstance(raw_name, str) or not isinstance(raw_embeddings, list):
                raise RegistryError("Registry user data is invalid")
            name = normalize_name(raw_name)
            key = name_key(name)
            if key in users:
                raise RegistryError(f"Duplicate registry user: {name}")
            embeddings = tuple(validate_embedding(embedding) for embedding in raw_embeddings)
            if not embeddings:
                raise RegistryError(f"Invalid embeddings for {name}")
            dimensions = {len(embedding) for embedding in embeddings}
            if len(dimensions) != 1:
                raise RegistryError(f"Inconsistent embeddings for {name}")
            for embedding in embeddings:
                previous_name = seen_embeddings.get(embedding)
                if previous_name is not None:
                    raise RegistryError(
                        f"Embedding is registered for both {previous_name} and {name}"
                    )
                seen_embeddings[embedding] = name
            users[key] = UserRecord(name=name, embeddings=embeddings)
        return users

    def _file_signature(self) -> FileSignature | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RegistryError(f"Unable to inspect registry: {self.path}") from exc
        return (stat.st_mtime_ns, stat.st_size, getattr(stat, "st_ino", 0))

    def _save(self, users: dict[str, UserRecord]) -> None:
        document = {
            "version": REGISTRY_VERSION,
            "users": [users[name].to_json() for name in sorted(users)],
        }
        if self._cipher.name is None:
            text = json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        else:
            plaintext = json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            envelope = self._cipher.encrypt(plaintext)
            text = json.dumps(
                {"version": REGISTRY_VERSION, "encrypted": self._cipher.name, **envelope},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        temporary_path: str | None = None
        descriptor: int | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            file_handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
            descriptor = None
            with file_handle:
                file_handle.write(text)
                file_handle.write("\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())
            with contextlib.suppress(OSError):
                os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.path)
            temporary_path = None
            self._cached_users = dict(users)
            self._cached_signature = self._file_signature()
        except OSError as exc:
            raise RegistryError(f"Unable to write registry: {self.path}") from exc
        finally:
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            if temporary_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temporary_path)
