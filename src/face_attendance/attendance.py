import contextlib
import csv
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Literal, cast

from .errors import AttendanceError
from .file_lock import exclusive_file_lock
from .validation import name_key, normalize_name

AttendanceAction = Literal["in", "out"]
VALID_ACTIONS = {"in", "out"}
FileSignature = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class AttendanceEvent:
    timestamp: str
    name: str
    action: AttendanceAction

    def to_row(self) -> dict[str, str]:
        return {"timestamp": self.timestamp, "name": self.name, "action": self.action}


class AttendanceLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._cached_events: list[AttendanceEvent] | None = None
        self._cached_actions: dict[str, AttendanceAction] | None = None
        self._cached_signature: FileSignature | None = None

    @contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            with self._lock, exclusive_file_lock(self.path):
                yield
        except OSError as exc:
            raise AttendanceError(f"Unable to access attendance log: {self.path}") from exc

    def ensure_exists(self) -> None:
        with self._locked():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self._events_snapshot_unlocked()
                return
            with self.path.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "name", "action"])
                writer.writeheader()
                handle.flush()
                os.fsync(handle.fileno())
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)
            self._cached_events = []
            self._cached_actions = {}
            self._cached_signature = self._file_signature_unlocked()

    def record(
        self,
        name: str,
        action: str,
        timestamp: datetime | None = None,
    ) -> AttendanceEvent:
        normalized_name = normalize_name(name)
        if not isinstance(action, str) or action not in VALID_ACTIONS:
            raise AttendanceError("Attendance action must be 'in' or 'out'")
        if timestamp is not None and not isinstance(timestamp, datetime):
            raise AttendanceError("Attendance timestamp must be a datetime")
        event_time = timestamp or datetime.now(timezone.utc)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        event = AttendanceEvent(
            timestamp=event_time.astimezone(timezone.utc).isoformat(),
            name=normalized_name,
            action=cast(AttendanceAction, action),
        )
        with self._locked():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            events = self._events_snapshot_unlocked()
            assert self._cached_actions is not None
            previous_action = self._cached_actions.get(name_key(normalized_name))
            if action == "in" and previous_action == "in":
                raise AttendanceError(f"{normalized_name} is already signed in")
            if action == "out" and previous_action != "in":
                raise AttendanceError(f"{normalized_name} is not signed in")
            with self.path.open("a+", encoding="utf-8", newline="") as handle:
                handle.seek(0, 2)
                if handle.tell() == 0:
                    writer = csv.DictWriter(handle, fieldnames=["timestamp", "name", "action"])
                    writer.writeheader()
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "name", "action"])
                writer.writerow(event.to_row())
                handle.flush()
                os.fsync(handle.fileno())
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)
            events.append(event)
            self._cached_actions[name_key(event.name)] = event.action
            self._cached_signature = self._file_signature_unlocked()
        return event

    def events(self) -> tuple[AttendanceEvent, ...]:
        with self._locked():
            return tuple(self._events_snapshot_unlocked())

    def _events_snapshot_unlocked(self) -> list[AttendanceEvent]:
        signature = self._file_signature_unlocked()
        if self._cached_events is not None and signature == self._cached_signature:
            return self._cached_events
        events = list(self._read_events_unlocked())
        self._cached_events = events
        self._cached_actions = self._build_actions(events)
        self._cached_signature = signature
        return events

    def _read_events_unlocked(self) -> tuple[AttendanceEvent, ...]:
        if not self.path.exists():
            return ()
        try:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                expected_fields = ["timestamp", "name", "action"]
                if reader.fieldnames != expected_fields:
                    raise AttendanceError("Attendance log has an invalid header")
                events: list[AttendanceEvent] = []
                for row in reader:
                    if set(row) != set(expected_fields) or any(
                        value is None or not value.strip() for value in row.values()
                    ):
                        raise AttendanceError("Attendance log contains an invalid row")
                    timestamp = self._parse_timestamp(row["timestamp"])
                    name = normalize_name(row["name"])
                    action = row["action"]
                    if action not in VALID_ACTIONS:
                        raise AttendanceError("Attendance log contains an invalid action")
                    events.append(
                        AttendanceEvent(
                            timestamp=timestamp,
                            name=name,
                            action=cast(AttendanceAction, action),
                        )
                    )
                return tuple(events)
        except AttendanceError:
            raise
        except (OSError, UnicodeError, csv.Error) as exc:
            raise AttendanceError(f"Unable to read attendance log: {self.path}") from exc

    @staticmethod
    def _parse_timestamp(value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise AttendanceError("Attendance log contains an invalid timestamp") from exc
        if parsed.tzinfo is None:
            raise AttendanceError("Attendance log timestamp must include a timezone")
        return parsed.astimezone(timezone.utc).isoformat()

    def _file_signature_unlocked(self) -> FileSignature | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise AttendanceError(f"Unable to inspect attendance log: {self.path}") from exc
        return (stat.st_mtime_ns, stat.st_size, getattr(stat, "st_ino", 0))

    @staticmethod
    def _build_actions(events: list[AttendanceEvent]) -> dict[str, AttendanceAction]:
        actions: dict[str, AttendanceAction] = {}
        for event in events:
            key = name_key(event.name)
            previous = actions.get(key)
            if event.action == "in" and previous == "in":
                raise AttendanceError(f"Invalid attendance transition for {event.name}")
            if event.action == "out" and previous != "in":
                raise AttendanceError(f"Invalid attendance transition for {event.name}")
            actions[key] = event.action
        return actions

    def latest_action(self, name: str) -> AttendanceAction | None:
        normalized_name = normalize_name(name)
        with self._locked():
            self._events_snapshot_unlocked()
            assert self._cached_actions is not None
            return self._cached_actions.get(name_key(normalized_name))
