import contextlib
import importlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    lock_api: Any = None
    handle = None
    acquired = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        if os.name == "nt":
            lock_api = importlib.import_module("msvcrt")
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            lock_api.locking(handle.fileno(), lock_api.LK_LOCK, 1)
        else:
            lock_api = importlib.import_module("fcntl")
            lock_api.flock(handle.fileno(), lock_api.LOCK_EX)
        acquired = True
        yield
    finally:
        if handle is not None:
            try:
                if acquired and lock_api is not None:
                    if os.name == "nt":
                        lock_api.locking(handle.fileno(), lock_api.LK_UNLCK, 1)
                    else:
                        lock_api.flock(handle.fileno(), lock_api.LOCK_UN)
            finally:
                handle.close()
