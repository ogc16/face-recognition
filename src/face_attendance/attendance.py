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
from .validation import normalize_name

AttendanceAction = Literal["in", "out"]
VALID_ACTIONS = {"in", "out"}


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
                self._read_events_unlocked()
                return
            with self.path.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "name", "action"])
                writer.writeheader()
                handle.flush()
                os.fsync(handle.fileno())

    def record(
        self,
        name: str,
        action: str,
        timestamp: datetime | None = None,
    ) -> AttendanceEvent:
        normalized_name = normalize_name(name)
        if action not in VALID_ACTIONS:
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
            previous_action = self._latest_action_unlocked(normalized_name)
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
        return event

    def events(self) -> tuple[AttendanceEvent, ...]:
        with self._locked():
            return self._read_events_unlocked()

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

    def _latest_action_unlocked(self, name: str) -> AttendanceAction | None:
        for event in reversed(self._read_events_unlocked()):
            if event.name == name:
                return event.action
        return None

    def latest_action(self, name: str) -> AttendanceAction | None:
        normalized_name = normalize_name(name)
        matching = [event for event in self.events() if event.name == normalized_name]
        return matching[-1].action if matching else None
