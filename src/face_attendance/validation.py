import math
import unicodedata
from collections.abc import Iterable
from typing import Any

from .errors import RegistryError

MAX_NAME_LENGTH = 80
MAX_EMBEDDING_LENGTH = 4096


def normalize_name(value: str) -> str:
    if not isinstance(value, str):
        raise RegistryError("Name must be text")

    name = unicodedata.normalize("NFKC", value).strip()
    if not name:
        raise RegistryError("Name cannot be empty")
    if len(name) > MAX_NAME_LENGTH:
        raise RegistryError(f"Name cannot exceed {MAX_NAME_LENGTH} characters")
    if name in {".", ".."}:
        raise RegistryError("Name is reserved")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise RegistryError("Name contains control characters")
    if "/" in name or "\\" in name:
        raise RegistryError("Name cannot contain path separators")
    return name


def validate_embedding(values: Iterable[Any]) -> tuple[float, ...]:
    try:
        embedding = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise RegistryError("Embedding must contain numbers") from exc

    if not embedding:
        raise RegistryError("Embedding cannot be empty")
    if len(embedding) > MAX_EMBEDDING_LENGTH:
        raise RegistryError("Embedding is too large")
    if any(not math.isfinite(value) for value in embedding):
        raise RegistryError("Embedding contains a non-finite number")
    return embedding
