from __future__ import annotations

import importlib
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from face_attendance.liveness import LivenessPolicy
from face_attendance.recognition import (
    DefaultFaceRecognitionBackend,
    FaceRecognitionService,
    RecognitionStatus,
)
from face_attendance.registry import FaceRegistry


def get_button(
    window: tk.Misc,
    text: str,
    color: str,
    command: Any,
    fg: str = "white",
) -> tk.Button:
    return tk.Button(
        window,
        text=text,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        bg=color,
        command=command,
        height=2,
        width=20,
        font=("Helvetica bold", 20),
    )


def get_img_label(window: tk.Misc) -> tk.Label:
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window: tk.Misc, text: str) -> tk.Label:
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window: tk.Misc) -> tk.Text:
    return tk.Text(window, height=2, width=15, font=("Arial", 32))


def msg_box(title: str, description: str) -> None:
    messagebox.showinfo(title, description)


def _to_rgb(image: Any) -> Any:
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        return image
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) == 3 and shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def recognize(image: Any, db_path: str | Path) -> str:
    path = Path(db_path)
    registry_path = path if path.suffix.lower() == ".json" else path / "registry.json"
    registry = FaceRegistry(registry_path)
    backend = DefaultFaceRecognitionBackend()
    service = FaceRecognitionService(registry, backend, backend.distance, tolerance=0.6)
    result = service.authenticate(_to_rgb(image), LivenessPolicy())
    if result.status is RecognitionStatus.MATCH and result.name is not None:
        return result.name
    if result.status is RecognitionStatus.NO_FACE:
        return "no_persons_found"
    return "unknown_person"
