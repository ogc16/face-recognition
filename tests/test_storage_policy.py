"""Shared storage policy contract.

Every embedding store and every attendance store must enforce identical rules.
These tests parametrize over the file-backed and SQLite implementations and
assert the *behaviour* only, so a backend cannot pass by agreeing with itself:
the JSON registry and the SQLite registry are checked against the same
expectations.

The abstract bases in :mod:`face_attendance.storage.base` own the policy, so a
new backend only needs to satisfy this contract to be correct.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

from face_attendance.attendance import AttendanceLog
from face_attendance.errors import AttendanceError, RegistryError
from face_attendance.protocols import AttendanceStore, EmbeddingStore
from face_attendance.registry import FaceRegistry
from face_attendance.storage import SqliteAttendanceStore, SqliteEmbeddingStore
from face_attendance.validation import name_key

Factory = Callable[[Path, int], EmbeddingStore]
AttendanceFactory = Callable[[Path], AttendanceStore]

ALPHA = (0.1, 0.2, 0.3)
BETA = (0.4, 0.5, 0.6)
GAMMA = (0.7, 0.8, 0.9)


def json_embeddings(path: Path, max_embeddings: int = 5) -> EmbeddingStore:
    return FaceRegistry(path, max_embeddings_per_user=max_embeddings)


def sqlite_embeddings(path: Path, max_embeddings: int = 5) -> EmbeddingStore:
    return SqliteEmbeddingStore(path, max_embeddings_per_user=max_embeddings)


def csv_attendance(path: Path) -> AttendanceStore:
    return AttendanceLog(path)


def sqlite_attendance(path: Path) -> AttendanceStore:
    return SqliteAttendanceStore(path)


EMBEDDING_STORES: dict[str, Factory] = {
    "json": json_embeddings,
    "sqlite": sqlite_embeddings,
}

ATTENDANCE_STORES: dict[str, AttendanceFactory] = {
    "csv": csv_attendance,
    "sqlite": sqlite_attendance,
}


@pytest.fixture(params=sorted(EMBEDDING_STORES))
def embedding_store(request: pytest.FixtureRequest, tmp_path: Path) -> EmbeddingStore:
    return EMBEDDING_STORES[request.param](tmp_path / "registry", 5)


@pytest.fixture(params=sorted(ATTENDANCE_STORES))
def attendance_store(request: pytest.FixtureRequest, tmp_path: Path) -> AttendanceStore:
    return ATTENDANCE_STORES[request.param](tmp_path / "attendance")


def test_registration_stores_and_reads_back(embedding_store: EmbeddingStore) -> None:
    assert embedding_store.register("Ada", ALPHA) is True
    record = embedding_store.get("Ada")
    assert record is not None
    assert record.name == "Ada"
    assert record.embeddings == (ALPHA,)


def test_registration_is_idempotent_for_the_same_sample(
    embedding_store: EmbeddingStore,
) -> None:
    assert embedding_store.register("Ada", ALPHA) is True
    assert embedding_store.register("Ada", ALPHA) is False
    record = embedding_store.get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA,)


def test_registration_appends_distinct_samples(embedding_store: EmbeddingStore) -> None:
    embedding_store.register("Ada", ALPHA)
    assert embedding_store.register("Ada", BETA) is True
    record = embedding_store.get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA, BETA)


def test_registration_normalizes_the_stored_name(embedding_store: EmbeddingStore) -> None:
    embedding_store.register("  Ada Lovelace  ", ALPHA)
    record = embedding_store.get("Ada Lovelace")
    assert record is not None
    assert record.name == "Ada Lovelace"


def test_registration_is_case_insensitive_on_lookup(
    embedding_store: EmbeddingStore,
) -> None:
    embedding_store.register("Ada", ALPHA)
    assert embedding_store.get("ADA") is not None


def test_a_sample_cannot_belong_to_two_users(embedding_store: EmbeddingStore) -> None:
    embedding_store.register("Ada", ALPHA)
    with pytest.raises(RegistryError):
        embedding_store.register("Grace", ALPHA)
    assert embedding_store.get("Grace") is None


def test_duplicate_detection_ignores_the_user_name_case(
    embedding_store: EmbeddingStore,
) -> None:
    embedding_store.register("Ada", ALPHA)
    assert embedding_store.register("ADA", ALPHA) is False
    record = embedding_store.get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA,)


def test_registration_enforces_the_per_user_sample_limit(tmp_path: Path) -> None:
    for name, factory in EMBEDDING_STORES.items():
        store = factory(tmp_path / f"capped-{name}", 2)
        assert store.register("Ada", ALPHA) is True
        assert store.register("Ada", BETA) is True
        assert store.register("Ada", GAMMA) is False
        record = store.get("Ada")
        assert record is not None
        assert record.embeddings == (ALPHA, BETA)


def test_raising_the_cap_allows_a_further_sample(tmp_path: Path) -> None:
    for name, factory in EMBEDDING_STORES.items():
        store = factory(tmp_path / f"raised-{name}", 1)
        assert store.register("Ada", ALPHA) is True
        assert store.register("Ada", BETA) is False
        roomier = factory(tmp_path / f"raised-{name}", 3)
        assert roomier.register("Ada", BETA) is True
        record = roomier.get("Ada")
        assert record is not None
        assert record.embeddings == (ALPHA, BETA)


def test_remove_deletes_the_user(embedding_store: EmbeddingStore) -> None:
    embedding_store.register("Ada", ALPHA)
    assert embedding_store.remove("Ada") is True
    assert embedding_store.get("Ada") is None


def test_remove_reports_a_missing_user(embedding_store: EmbeddingStore) -> None:
    assert embedding_store.remove("Nobody") is False


def test_remove_is_case_insensitive(embedding_store: EmbeddingStore) -> None:
    embedding_store.register("Ada", ALPHA)
    assert embedding_store.remove("aDa") is True


def test_names_are_sorted_case_insensitively(embedding_store: EmbeddingStore) -> None:
    embedding_store.register("grace", BETA)
    embedding_store.register("Ada", ALPHA)
    embedding_store.register("bob", GAMMA)
    assert embedding_store.names() == ("Ada", "bob", "grace")


def test_registration_rejects_an_invalid_name(embedding_store: EmbeddingStore) -> None:
    with pytest.raises(RegistryError):
        embedding_store.register("   ", ALPHA)


def test_registration_rejects_a_non_finite_embedding(
    embedding_store: EmbeddingStore,
) -> None:
    with pytest.raises(RegistryError):
        embedding_store.register("Ada", (0.1, float("nan")))


def test_registration_rejects_an_empty_embedding(embedding_store: EmbeddingStore) -> None:
    with pytest.raises(RegistryError):
        embedding_store.register("Ada", ())


def test_records_and_iter_embeddings_agree(embedding_store: EmbeddingStore) -> None:
    embedding_store.register("Ada", ALPHA)
    embedding_store.register("Ada", BETA)
    embedding_store.register("Grace", GAMMA)
    assert len(embedding_store.records()) == 2
    assert len(list(embedding_store.iter_embeddings())) == 3
    assert dict(embedding_store.iter_embeddings()) == {
        "Ada": BETA,
        "Grace": GAMMA,
    }
    assert len(embedding_store) == 2


def test_embedding_store_satisfies_the_protocol() -> None:
    assert isinstance(FaceRegistry(Path("unused.json")), EmbeddingStore)
    assert isinstance(SqliteEmbeddingStore(Path("unused.db")), EmbeddingStore)


def test_attendance_store_satisfies_the_protocol() -> None:
    assert isinstance(AttendanceLog(Path("unused.csv")), AttendanceStore)
    assert isinstance(SqliteAttendanceStore(Path("unused.db")), AttendanceStore)


def test_recording_a_sign_in_stores_the_event(attendance_store: AttendanceStore) -> None:
    when = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
    event = attendance_store.record("Ada", "in", when)
    assert event.name == "Ada"
    assert event.action == "in"
    assert event.timestamp == when.isoformat()
    assert attendance_store.events() == (event,)


def test_a_naive_timestamp_is_interpreted_as_utc(
    attendance_store: AttendanceStore,
) -> None:
    naive = datetime(2026, 9, 25, 9, 0)
    event = attendance_store.record("Ada", "in", naive)
    assert event.timestamp == naive.replace(tzinfo=timezone.utc).isoformat()


def test_a_timestamp_is_normalized_to_utc(attendance_store: AttendanceStore) -> None:
    from datetime import timedelta

    offset = timezone(timedelta(hours=5))
    event = attendance_store.record("Ada", "in", datetime(2026, 9, 25, 14, 0, tzinfo=offset))
    assert event.timestamp.endswith("+00:00")
    assert event.timestamp.startswith("2026-09-25T09:00")


def test_events_preserve_append_order(attendance_store: AttendanceStore) -> None:
    first = attendance_store.record("Ada", "in", datetime(2026, 9, 25, 9, tzinfo=timezone.utc))
    second = attendance_store.record("Grace", "in", datetime(2026, 9, 25, 10, tzinfo=timezone.utc))
    third = attendance_store.record("Ada", "out", datetime(2026, 9, 25, 11, tzinfo=timezone.utc))
    assert attendance_store.events() == (first, second, third)


def test_a_double_sign_in_is_rejected(attendance_store: AttendanceStore) -> None:
    attendance_store.record("Ada", "in")
    with pytest.raises(AttendanceError):
        attendance_store.record("Ada", "in")


def test_a_sign_out_without_a_sign_in_is_rejected(
    attendance_store: AttendanceStore,
) -> None:
    with pytest.raises(AttendanceError):
        attendance_store.record("Ada", "out")


def test_a_failed_transition_is_not_persisted(attendance_store: AttendanceStore) -> None:
    with pytest.raises(AttendanceError):
        attendance_store.record("Ada", "out")
    assert attendance_store.events() == ()
    assert attendance_store.latest_action("Ada") is None


def test_a_rejected_duplicate_does_not_block_the_next_sign_in(
    attendance_store: AttendanceStore,
) -> None:
    attendance_store.record("Ada", "in")
    with pytest.raises(AttendanceError):
        attendance_store.record("Ada", "in")
    attendance_store.record("Ada", "out")
    assert attendance_store.latest_action("Ada") == "out"


def test_an_invalid_action_is_rejected(attendance_store: AttendanceStore) -> None:
    with pytest.raises(AttendanceError):
        attendance_store.record("Ada", "sideways")


def test_a_non_datetime_timestamp_is_rejected(attendance_store: AttendanceStore) -> None:
    with pytest.raises(AttendanceError):
        attendance_store.record("Ada", "in", "2026-09-25T09:00:00")  # type: ignore[arg-type]


def test_latest_action_is_case_insensitive(attendance_store: AttendanceStore) -> None:
    attendance_store.record("Ada", "in")
    assert attendance_store.latest_action("ADA") == "in"
    assert attendance_store.latest_action("nobody") is None


def test_two_identities_track_state_independently(attendance_store: AttendanceStore) -> None:
    attendance_store.record("Ada", "in")
    attendance_store.record("Grace", "in")
    attendance_store.record("Ada", "out")
    assert attendance_store.latest_action("Ada") == "out"
    assert attendance_store.latest_action("Grace") == "in"
    with pytest.raises(AttendanceError):
        attendance_store.record("Grace", "in")


def test_the_name_is_normalized_before_storage(attendance_store: AttendanceStore) -> None:
    event = attendance_store.record("  Ada  ", "in")
    assert event.name == "Ada"
    assert attendance_store.latest_action("ada") == "in"
    assert name_key(event.name) == "ada"
