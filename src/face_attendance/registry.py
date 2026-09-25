import contextlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from .errors import RegistryError
from .file_lock import exclusive_file_lock
from .validation import normalize_name, validate_embedding

REGISTRY_VERSION = 1
Embedding = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class UserRecord:
    name: str
    embeddings: tuple[Embedding, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "embeddings": [list(embedding) for embedding in self.embeddings],
        }


class FaceRegistry:
    def __init__(self, path: str | Path, max_embeddings_per_user: int = 5) -> None:
        if isinstance(max_embeddings_per_user, bool) or not isinstance(
            max_embeddings_per_user, int
        ):
            raise RegistryError("Maximum embeddings per user must be an integer")
        if max_embeddings_per_user < 1:
            raise RegistryError("At least one embedding per user is required")
        self.path = Path(path)
        self.max_embeddings_per_user = max_embeddings_per_user
        self._lock = RLock()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            with self._lock, exclusive_file_lock(self.path):
                yield
        except OSError as exc:
            raise RegistryError(f"Unable to access registry: {self.path}") from exc

    def register(self, name: str, embedding: tuple[float, ...] | list[float]) -> bool:
        normalized_name = normalize_name(name)
        normalized_embedding = validate_embedding(embedding)
        with self._locked():
            users = self._load()
            existing = users.get(normalized_name)
            if existing is not None and normalized_embedding in existing.embeddings:
                return False
            if existing is None:
                users[normalized_name] = UserRecord(
                    name=normalized_name, embeddings=(normalized_embedding,)
                )
            elif len(existing.embeddings) < self.max_embeddings_per_user:
                users[normalized_name] = UserRecord(
                    name=normalized_name,
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
            if normalized_name not in users:
                return False
            del users[normalized_name]
            self._save(users)
            return True

    def get(self, name: str) -> UserRecord | None:
        normalized_name = normalize_name(name)
        with self._locked():
            return self._load().get(normalized_name)

    def names(self) -> tuple[str, ...]:
        with self._locked():
            return tuple(sorted(self._load()))

    def records(self) -> tuple[UserRecord, ...]:
        with self._locked():
            users = self._load()
            return tuple(users[name] for name in sorted(users))

    def iter_embeddings(self) -> Iterator[tuple[str, Embedding]]:
        with self._locked():
            users = self._load()
            snapshots = tuple(
                (name, embedding) for name in sorted(users) for embedding in users[name].embeddings
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
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"Unable to read registry: {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("version") != REGISTRY_VERSION:
            raise RegistryError("Unsupported registry format")
        if set(payload) != {"version", "users"}:
            raise RegistryError("Registry contains unsupported fields")
        raw_users = payload.get("users")
        if not isinstance(raw_users, list):
            raise RegistryError("Registry users must be a list")
        users: dict[str, UserRecord] = {}
        for raw_user in raw_users:
            if not isinstance(raw_user, dict) or set(raw_user) != {"name", "embeddings"}:
                raise RegistryError("Registry user data is invalid")
            raw_name = raw_user["name"]
            raw_embeddings = raw_user["embeddings"]
            if not isinstance(raw_name, str) or not isinstance(raw_embeddings, list):
                raise RegistryError("Registry user data is invalid")
            name = normalize_name(raw_name)
            if name in users:
                raise RegistryError(f"Duplicate registry user: {name}")
            embeddings = tuple(validate_embedding(embedding) for embedding in raw_embeddings)
            if not embeddings:
                raise RegistryError(f"Invalid embeddings for {name}")
            dimensions = {len(embedding) for embedding in embeddings}
            if len(dimensions) != 1:
                raise RegistryError(f"Inconsistent embeddings for {name}")
            users[name] = UserRecord(name=name, embeddings=embeddings)
        return users

    def _save(self, users: dict[str, UserRecord]) -> None:
        payload = {
            "version": REGISTRY_VERSION,
            "users": [users[name].to_json() for name in sorted(users)],
        }
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
                json.dump(
                    payload,
                    file_handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                file_handle.write("\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        except OSError as exc:
            raise RegistryError(f"Unable to write registry: {self.path}") from exc
        finally:
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            if temporary_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temporary_path)
