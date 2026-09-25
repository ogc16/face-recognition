"""PostgreSQL backend tests.

``psycopg`` is an optional dependency and no PostgreSQL server is available in
CI, so these tests inject a fake DB-API driver. The fake is a real transaction
simulator rather than a passthrough: writes are staged per connection and only
become visible on ``commit()``, and ``rollback()`` discards them.

That means the tests genuinely exercise what the backend is responsible for --
the SQL it issues, the parameters it binds, the JSONB encoding round-trip, the
domain error mapping, and the commit/rollback discipline -- while the parts
owned by the server (query planning, durability) are explicitly out of scope.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from typing import Any, cast

import pytest

from face_attendance.errors import AttendanceError, RegistryError
from face_attendance.storage import PostgresAttendanceStore, PostgresEmbeddingStore

ALPHA = (0.1, 0.2, 0.3)
BETA = (0.4, 0.5, 0.6)


class FakeOperationalError(RuntimeError):
    """Stands in for a psycopg error raised by the fake server."""


class FakeDatabase:
    """Committed state plus the SQL the backend issued.

    Attributes:
        users: Committed ``name_key -> (name, embeddings JSON)`` rows.
        events: Committed ``(timestamp, name, action)`` rows, in order.
        statements: Every statement the backend executed.
        fail_on: Substring of a statement that should raise, for fault tests.
    """

    def __init__(self) -> None:
        self.users: dict[str, tuple[str, str]] = {}
        self.events: list[tuple[str, str, str]] = []
        self.statements: list[str] = []
        self.fail_on: str | None = None
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def execute(self, statement: str, parameters: object, staged: _Staged) -> list[tuple[Any, ...]]:
        """Record and apply one statement, returning any selected rows.

        Args:
            statement: The SQL text.
            parameters: The bound parameters, or ``None``.
            staged: The connection's uncommitted state.

        Returns:
            Rows for a ``SELECT``, otherwise an empty list.

        Raises:
            FakeOperationalError: If this statement matches :attr:`fail_on`.
        """
        self.statements.append(statement)
        if self.fail_on is not None and self.fail_on in statement:
            raise FakeOperationalError(f"simulated failure for {statement!r}")
        collapsed = " ".join(statement.split())
        if collapsed.startswith("CREATE"):
            return []
        if collapsed == "DELETE FROM face_users":
            staged.users.clear()
            return []
        if collapsed.startswith("SELECT name_key, name, embeddings FROM face_users"):
            return [(key, name, payload) for key, (name, payload) in sorted(staged.users.items())]
        if collapsed.startswith("INSERT INTO face_users"):
            key, name, payload = cast(tuple[str, str, str], tuple(parameters))  # type: ignore[arg-type]
            staged.users[key] = (name, payload)
            return []
        if collapsed.startswith("SELECT timestamp, name, action FROM face_attendance_log"):
            return list(staged.events)
        if collapsed.startswith("INSERT INTO face_attendance_log"):
            timestamp, name, _name_key, action = cast(
                tuple[str, str, str, str],
                tuple(parameters),  # type: ignore[arg-type]
            )
            staged.events.append((timestamp, name, action))
            return []
        raise AssertionError(f"Unexpected statement: {statement!r}")


class _Staged:
    """A connection's uncommitted view of the database."""

    def __init__(self, database: FakeDatabase) -> None:
        self._database = database
        self.users = dict(database.users)
        self.events = list(database.events)


class FakeCursor:
    """A cursor over a single :class:`FakeDatabase`."""

    def __init__(self, database: FakeDatabase, staged: _Staged) -> None:
        self._database = database
        self._staged = staged
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, operation: str, parameters: object = None) -> None:
        self._rows = self._database.execute(operation, parameters, self._staged)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class FakeConnection:
    """A connection whose writes only land on :meth:`commit`."""

    def __init__(self, database: FakeDatabase) -> None:
        self._database = database
        self._staged = _Staged(database)
        self._finished = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._database, self._staged)

    def commit(self) -> None:
        self._database.users = dict(self._staged.users)
        self._database.events = list(self._staged.events)
        self._database.commits += 1
        self._finished = True

    def rollback(self) -> None:
        self._database.rollbacks += 1
        self._finished = True

    def close(self) -> None:
        self._database.closed += 1


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDatabase]:
    """Install the fake driver as ``psycopg`` for the duration of a test."""
    fake = FakeDatabase()
    module = types.ModuleType("psycopg")
    module.connect = lambda dsn: FakeConnection(fake)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", module)
    yield fake
    monkeypatch.delitem(sys.modules, "psycopg", raising=False)


def test_a_missing_driver_reports_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "psycopg", raising=False)
    monkeypatch.setattr(
        "face_attendance.storage.postgresql.importlib.import_module",
        _raise_import_error,
    )
    with pytest.raises(RegistryError) as error:
        PostgresEmbeddingStore("postgresql://localhost/faces")._load_driver()
    assert "pip install" in str(error.value)
    assert "face-attendance[postgres]" in str(error.value)


def _raise_import_error(name: str) -> types.ModuleType:
    raise ImportError(f"no module named {name}")


def test_registration_round_trips_through_jsonb(database: FakeDatabase) -> None:
    store = PostgresEmbeddingStore("postgresql://localhost/faces")
    assert store.register("Ada", ALPHA) is True
    record = store.get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA,)
    stored = database.users["ada"]
    assert stored[0] == "Ada"
    assert json.loads(stored[1]) == [list(ALPHA)]


def test_jsonb_columns_are_casted_explicitly(database: FakeDatabase) -> None:
    PostgresEmbeddingStore("postgresql://localhost/faces").register("Ada", ALPHA)
    inserts = [s for s in database.statements if s.startswith("INSERT INTO face_users")]
    assert inserts and all("%s::jsonb" in s for s in inserts)


def test_the_schema_is_created_before_writing(database: FakeDatabase) -> None:
    PostgresEmbeddingStore("postgresql://localhost/faces").register("Ada", ALPHA)
    assert any("CREATE TABLE IF NOT EXISTS face_users" in s for s in database.statements)


def test_multiple_samples_are_preserved(database: FakeDatabase) -> None:
    store = PostgresEmbeddingStore("postgresql://localhost/faces")
    store.register("Ada", ALPHA)
    assert store.register("Ada", BETA) is True
    record = store.get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA, BETA)


def test_a_second_user_is_stored(database: FakeDatabase) -> None:
    store = PostgresEmbeddingStore("postgresql://localhost/faces")
    store.register("Ada", ALPHA)
    store.register("Grace", BETA)
    assert store.names() == ("Ada", "Grace")
    assert len(store) == 2


def test_remove_deletes_the_row(database: FakeDatabase) -> None:
    store = PostgresEmbeddingStore("postgresql://localhost/faces")
    store.register("Ada", ALPHA)
    assert store.remove("Ada") is True
    assert store.get("Ada") is None
    assert database.users == {}


def test_a_failed_write_is_rolled_back(database: FakeDatabase) -> None:
    store = PostgresEmbeddingStore("postgresql://localhost/faces")
    store.register("Ada", ALPHA)
    database.fail_on = "INSERT INTO face_users"
    with pytest.raises(RegistryError):
        store.register("Ada", BETA)
    database.fail_on = None
    assert database.rollbacks == 1
    record = store.get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA,)


def test_a_corrupt_jsonb_payload_is_reported(database: FakeDatabase) -> None:
    database.users["ada"] = ("Ada", "{not json")
    with pytest.raises(RegistryError):
        PostgresEmbeddingStore("postgresql://localhost/faces").get("Ada")


def test_a_non_list_payload_is_reported(database: FakeDatabase) -> None:
    database.users["ada"] = ("Ada", json.dumps({"not": "a list"}))
    with pytest.raises(RegistryError):
        PostgresEmbeddingStore("postgresql://localhost/faces").get("Ada")


def test_a_string_payload_is_decoded(database: FakeDatabase) -> None:
    database.users["ada"] = ("Ada", json.dumps([list(ALPHA)]))
    record = PostgresEmbeddingStore("postgresql://localhost/faces").get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA,)


def test_a_bytes_payload_is_decoded(database: FakeDatabase) -> None:
    database.users["ada"] = ("Ada", json.dumps([list(ALPHA)]).encode("utf-8"))
    record = PostgresEmbeddingStore("postgresql://localhost/faces").get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA,)


def test_an_already_decoded_payload_is_accepted(database: FakeDatabase) -> None:
    database.users["ada"] = ("Ada", [list(ALPHA)])
    record = PostgresEmbeddingStore("postgresql://localhost/faces").get("Ada")
    assert record is not None
    assert record.embeddings == (ALPHA,)


def test_the_sample_cap_is_enforced(database: FakeDatabase) -> None:
    store = PostgresEmbeddingStore("postgresql://localhost/faces", max_embeddings_per_user=1)
    assert store.register("Ada", ALPHA) is True
    assert store.register("Ada", BETA) is False


def test_an_invalid_cap_is_rejected() -> None:
    with pytest.raises(RegistryError):
        PostgresEmbeddingStore("postgresql://localhost/faces", max_embeddings_per_user=0)
    with pytest.raises(RegistryError):
        PostgresEmbeddingStore("postgresql://localhost/faces", max_embeddings_per_user=True)


def test_attendance_events_round_trip(database: FakeDatabase) -> None:
    store = PostgresAttendanceStore("postgresql://localhost/faces")
    event = store.record("Ada", "in")
    assert event.name == "Ada"
    assert store.events() == (event,)
    assert store.latest_action("Ada") == "in"
    assert database.events == [(event.timestamp, "Ada", "in")]


def test_attendance_preserves_append_order(database: FakeDatabase) -> None:
    store = PostgresAttendanceStore("postgresql://localhost/faces")
    store.record("Ada", "in")
    store.record("Grace", "in")
    store.record("Ada", "out")
    assert [event.name for event in store.events()] == ["Ada", "Grace", "Ada"]
    assert store.latest_action("Ada") == "out"


def test_a_double_sign_in_is_rejected(database: FakeDatabase) -> None:
    store = PostgresAttendanceStore("postgresql://localhost/faces")
    store.record("Ada", "in")
    with pytest.raises(AttendanceError):
        store.record("Ada", "in")


def test_a_failed_insert_does_not_append(database: FakeDatabase) -> None:
    store = PostgresAttendanceStore("postgresql://localhost/faces")
    store.record("Ada", "in")
    database.fail_on = "INSERT INTO face_attendance_log"
    with pytest.raises(AttendanceError):
        store.record("Ada", "out")
    database.fail_on = None
    assert store.latest_action("Ada") == "in"


def test_the_attendance_schema_is_created(database: FakeDatabase) -> None:
    PostgresAttendanceStore("postgresql://localhost/faces").ensure_exists()
    assert any("CREATE TABLE IF NOT EXISTS face_attendance_log" in s for s in database.statements)
    assert any("CREATE INDEX IF NOT EXISTS" in s for s in database.statements)


def test_connections_are_always_closed(database: FakeDatabase) -> None:
    store = PostgresEmbeddingStore("postgresql://localhost/faces")
    store.register("Ada", ALPHA)
    store.get("Ada")
    assert database.closed == database.commits


def test_the_name_key_is_indexed_in_the_attendance_table(
    database: FakeDatabase,
) -> None:
    PostgresAttendanceStore("postgresql://localhost/faces").record("Ada", "in")
    inserts = [s for s in database.statements if s.startswith("INSERT INTO face_attendance_log")]
    assert inserts and all("name_key" in s for s in inserts)


def test_the_driver_loader_returns_the_module(database: FakeDatabase) -> None:
    store = PostgresEmbeddingStore("postgresql://localhost/faces")
    driver = store._load_driver()
    assert driver.connect("postgresql://localhost/faces") is not None


def test_a_driver_fault_becomes_a_domain_error(database: FakeDatabase) -> None:
    database.fail_on = "SELECT"
    with pytest.raises(RegistryError):
        PostgresEmbeddingStore("postgresql://localhost/faces").get("Ada")
    database.fail_on = None


def test_an_attendance_driver_fault_becomes_a_domain_error(
    database: FakeDatabase,
) -> None:
    database.fail_on = "SELECT"
    with pytest.raises(AttendanceError):
        PostgresAttendanceStore("postgresql://localhost/faces").events()
    database.fail_on = None


def test_a_corrupt_payload_error_is_not_double_wrapped(database: FakeDatabase) -> None:
    database.users["ada"] = ("Ada", "{not json")
    with pytest.raises(RegistryError) as error:
        PostgresEmbeddingStore("postgresql://localhost/faces").get("Ada")
    assert "corrupt" in str(error.value)


def test_the_dsn_never_leaks_into_an_error_message(
    database: FakeDatabase,
) -> None:
    secret = "hunter2"
    dsn = f"postgresql://appuser:{secret}@db.internal/faces"
    database.fail_on = "SELECT"
    with pytest.raises(RegistryError) as error:
        PostgresEmbeddingStore(dsn).get("Ada")
    database.fail_on = None
    message = str(error.value)
    assert secret not in message
    assert "appuser" not in message
    assert dsn not in message


def test_a_connection_failure_is_reported_as_a_domain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "psycopg", raising=False)
    module = types.ModuleType("psycopg")

    def refuse(dsn: str) -> FakeConnection:
        raise FakeOperationalError("could not connect to server")

    module.connect = refuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", module)
    with pytest.raises(RegistryError):
        PostgresEmbeddingStore("postgresql://localhost/faces").get("Ada")
    with pytest.raises(AttendanceError):
        PostgresAttendanceStore("postgresql://localhost/faces").events()
