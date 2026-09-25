from __future__ import annotations

import contextlib
import queue
import tkinter as tk
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from tkinter import messagebox, ttk
from typing import Any

from .attendance import AttendanceAction, AttendanceEvent
from .camera import (
    CameraFactory,
    OpenCVCameraFactory,
    optional_module,
    require_vision_dependencies,
)
from .config import AppConfig
from .errors import DependencyError, FaceAttendanceError
from .protocols import FrameSource
from .recognition import RecognitionResult, RecognitionStatus
from .runtime import Runtime, build_runtime
from .validation import normalize_name

_CV2: Any = None
_IMAGE: Any = None
_IMAGE_TK: Any = None


def _load_gui_dependencies() -> None:
    """Populate the module-level handles used for image conversion.

    Raises:
        DependencyError: If OpenCV or Pillow cannot be imported.
    """
    global _CV2, _IMAGE, _IMAGE_TK
    _CV2 = optional_module("cv2")
    _IMAGE = optional_module("PIL.Image")
    _IMAGE_TK = optional_module("PIL.ImageTk")
    require_vision_dependencies("cv2", "PIL.Image", "PIL.ImageTk")


class FaceAttendanceApp:
    def __init__(
        self,
        runtime: Runtime,
        camera_factory: CameraFactory | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = runtime.config
        self.root = tk.Tk()
        try:
            _load_gui_dependencies()
            factory = camera_factory or OpenCVCameraFactory()
            self.camera: FrameSource = factory.create(self.config.camera_index)
        except Exception:
            self.root.destroy()
            raise
        self._latest_frame: Any | None = None
        self._photo_references: dict[ttk.Label, Any] = {}
        self._after_id: str | None = None
        self._task_poll_id: str | None = None
        self._task_results: queue.Queue[tuple[Future[Any], Callable[[Any], None]]] = queue.Queue()
        self._closed = False
        self._busy = False
        self._registration_window: tk.Toplevel | None = None
        self._registration_image_label: ttk.Label | None = None
        self.registration_accept_button: ttk.Button | None = None
        self._registration_frame: Any | None = None
        self._registration_name: tk.StringVar | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-attendance")
        self.remove_button: ttk.Button | None = None
        self._users_listbox: tk.Listbox | None = None
        self._configure_window()
        self._build_widgets()
        self._refresh_users()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._schedule_frame(0)
        self._schedule_task_poll()

    def _configure_window(self) -> None:
        self.root.title("Face Attendance")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}+100+100")
        self.root.minsize(800, 600)
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _build_widgets(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 22, "bold"))
        style.configure("Status.TLabel", font=("Helvetica", 11))

        container = ttk.Frame(self.root, padding=16)
        container.grid(row=0, column=0, columnspan=2, sticky="nsew")
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        video_panel = ttk.LabelFrame(container, text="Live camera", padding=8)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.video_label = ttk.Label(video_panel, anchor="center")
        self.video_label.pack(fill="both", expand=True)

        controls = ttk.Frame(container, padding=(8, 8))
        controls.grid(row=0, column=1, sticky="ns")
        ttk.Label(controls, text="Face Attendance", style="Title.TLabel").pack(
            anchor="w", pady=(0, 24)
        )
        self.login_button = ttk.Button(controls, text="Sign in", command=self._on_login)
        self.login_button.pack(fill="x", pady=(0, 10))
        self.logout_button = ttk.Button(controls, text="Sign out", command=self._on_logout)
        self.logout_button.pack(fill="x", pady=(0, 10))
        self.register_button = ttk.Button(
            controls, text="Register user", command=self._open_registration
        )
        self.register_button.pack(fill="x", pady=(0, 20))
        ttk.Label(controls, text="Registered users").pack(anchor="w")
        self._users_listbox = tk.Listbox(
            controls,
            height=6,
            selectmode=tk.BROWSE,
            exportselection=False,
            activestyle="none",
        )
        self._users_listbox.pack(fill="x", pady=(4, 8))
        self.remove_button = ttk.Button(
            controls,
            text="Remove selected user",
            command=self._remove_selected_user,
        )
        self.remove_button.pack(fill="x", pady=(0, 20))
        self.status_var = tk.StringVar(value="Starting camera…")
        ttk.Label(
            controls,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=240,
        ).pack(anchor="w", fill="x")
        if self.runtime.liveness_policy.checker is None:
            liveness_status = (
                "Liveness required but unavailable"
                if self.runtime.liveness_policy.required
                else "Liveness optional and unavailable"
            )
        else:
            liveness_status = (
                "Liveness active and required"
                if self.runtime.liveness_policy.required
                else "Liveness active and optional"
            )
        self.liveness_var = tk.StringVar(value=liveness_status)
        ttk.Label(controls, textvariable=self.liveness_var, wraplength=240).pack(
            anchor="w", fill="x", pady=(16, 0)
        )

    def _schedule_frame(self, delay: int) -> None:
        if self._closed:
            return
        try:
            self._after_id = self.root.after(delay, self._process_webcam)
        except tk.TclError:
            self._after_id = None

    def _process_webcam(self) -> None:
        if self._closed:
            return
        delay = 30
        try:
            success, frame = self.camera.read()
            if not success or frame is None:
                self.status_var.set("Waiting for camera frame…")
                delay = 100
                return
            self._latest_frame = frame
            self._show_frame(self.video_label, frame)
        except FaceAttendanceError as exc:
            self.status_var.set(str(exc))
            delay = 250
        except Exception:
            self.status_var.set("Camera processing failed")
            delay = 500
        finally:
            self._schedule_frame(delay)

    def _show_frame(self, label: ttk.Label, frame: Any) -> None:
        cv2 = _CV2
        image_module = _IMAGE
        image_tk_module = _IMAGE_TK
        if cv2 is None or image_module is None or image_tk_module is None:
            raise DependencyError("GUI dependencies are unavailable")
        image = image_module.fromarray(frame)
        width = max(320, label.winfo_width() - 12)
        height = max(240, label.winfo_height() - 12)
        image.thumbnail((width, height))
        photo = image_tk_module.PhotoImage(image=image)
        label.configure(image=photo)
        self._photo_references[label] = photo

    def _snapshot(self) -> Any | None:
        if self._latest_frame is None:
            return None
        copier = getattr(self._latest_frame, "copy", None)
        return copier() if callable(copier) else self._latest_frame

    def _on_login(self) -> None:
        self._begin_attendance("in")

    def _on_logout(self) -> None:
        self._begin_attendance("out")

    def _begin_attendance(self, action: AttendanceAction) -> None:
        if self._busy:
            return
        frame = self._snapshot()
        if frame is None:
            messagebox.showwarning("No camera frame", "Wait for a camera frame and try again.")
            return
        self._set_busy(True)
        self._submit(
            lambda: self._authenticate(frame, action),
            self._attendance_finished,
        )

    def _authenticate(
        self, frame: Any, action: AttendanceAction
    ) -> tuple[RecognitionResult, AttendanceEvent | None]:
        result = self.runtime.service.authenticate(frame, self.runtime.liveness_policy)
        if not result.matched or result.name is None:
            return result, None
        event = self.runtime.attendance.record(result.name, action)
        return result, event

    def _attendance_finished(self, value: Any) -> None:
        result, event = value
        if not result.matched or result.name is None or event is None:
            self._show_recognition_failure(result)
            return
        quality = (
            f"{result.match_quality:.0%} match quality"
            if result.match_quality is not None
            else "confirmed"
        )
        action = "signed in" if event.action == "in" else "signed out"
        self.status_var.set(f"{result.name} {action}")
        messagebox.showinfo(
            "Face recognized",
            f"{result.name} {action} with {quality}.",
        )

    def _show_recognition_failure(self, result: RecognitionResult) -> None:
        messages = {
            RecognitionStatus.NO_FACE: "No face was detected. Center your face in the camera.",
            RecognitionStatus.MULTIPLE_FACES: (
                "Multiple faces were detected. Only one person may sign in at a time."
            ),
            RecognitionStatus.UNKNOWN: "This face is not registered. Register the user first.",
        }
        message = messages.get(result.status, "Face recognition failed. Try again.")
        self.status_var.set("Recognition failed")
        messagebox.showwarning("Face not recognized", message)

    def _refresh_users(self) -> None:
        if self._users_listbox is None:
            return
        try:
            names = self.runtime.registry.names()
        except FaceAttendanceError:
            self.status_var.set("Unable to load registered users")
            return
        self._users_listbox.delete(0, tk.END)
        for name in names:
            self._users_listbox.insert(tk.END, name)

    def _remove_selected_user(self) -> None:
        if self._busy or self._users_listbox is None:
            return
        selection = self._users_listbox.curselection()
        if not selection:
            messagebox.showwarning("No user selected", "Select a user to remove.")
            return
        name = self._users_listbox.get(selection[0])
        if not messagebox.askyesno(
            "Remove user",
            f"Remove the registered face data for {name}?",
        ):
            return
        self._set_busy(True)
        self._submit(
            lambda: (self.runtime.registry.remove(name), name),
            self._removal_finished,
        )

    def _removal_finished(self, value: Any) -> None:
        removed, name = value
        self._refresh_users()
        if removed:
            self.status_var.set(f"Removed {name}")
        else:
            self.status_var.set(f"{name} was not registered")

    def _open_registration(self) -> None:
        if self._busy:
            return
        frame = self._snapshot()
        if frame is None:
            messagebox.showwarning("No camera frame", "Wait for a camera frame and try again.")
            return
        window = tk.Toplevel(self.root)
        window.title("Register user")
        window.geometry("760x560+180+140")
        window.transient(self.root)
        window.grab_set()
        window.protocol("WM_DELETE_WINDOW", self._close_registration)
        self._registration_window = window
        self._registration_frame = frame
        self._registration_name = tk.StringVar()

        container = ttk.Frame(window, padding=16)
        container.pack(fill="both", expand=True)
        ttk.Label(
            container,
            text="Register a new face",
            style="Title.TLabel",
        ).pack(anchor="w", pady=(0, 12))
        self._registration_image_label = ttk.Label(container, anchor="center")
        self._registration_image_label.pack(fill="both", expand=True, pady=(0, 12))
        self._show_frame(self._registration_image_label, frame)
        ttk.Label(container, text="Name").pack(anchor="w")
        ttk.Entry(container, textvariable=self._registration_name).pack(fill="x", pady=(4, 12))
        buttons = ttk.Frame(container)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel", command=self._close_registration).pack(
            side="right", padx=(8, 0)
        )
        self.registration_accept_button = ttk.Button(
            buttons, text="Register", command=self._accept_registration
        )
        self.registration_accept_button.pack(side="right")

    def _close_registration(self) -> None:
        image_label = self._registration_image_label
        if image_label is not None:
            self._photo_references.pop(image_label, None)
            with contextlib.suppress(tk.TclError):
                image_label.configure(image="")
        if self._registration_window is not None:
            try:
                if self._registration_window.winfo_exists():
                    self._registration_window.destroy()
            except tk.TclError:
                pass
        self._registration_window = None
        self._registration_image_label = None
        self.registration_accept_button = None
        self._registration_frame = None
        self._registration_name = None

    def _accept_registration(self) -> None:
        if self._busy or self._registration_name is None or self._registration_frame is None:
            return
        window = self._registration_window
        if window is None:
            return
        name = self._registration_name.get()
        try:
            normalized_name = normalize_name(name)
        except FaceAttendanceError as exc:
            messagebox.showerror("Invalid name", str(exc), parent=window)
            return
        frame = self._registration_frame
        self._set_busy(True)
        self._submit(
            lambda: self._register_face(frame, normalized_name),
            self._registration_finished,
        )

    def _register_face(self, frame: Any, name: str) -> tuple[bool, str]:
        embedding = self.runtime.service.embedding_for(frame)
        return self.runtime.registry.register(name, embedding), name

    def _registration_finished(self, value: Any) -> None:
        added, name = value
        if added:
            messagebox.showinfo("Registration complete", f"{name} was registered successfully.")
            self.status_var.set(f"Registered {name}")
        else:
            messagebox.showinfo("Already registered", "That face sample is already registered.")
        self._refresh_users()
        self._close_registration()

    def _submit(self, task: Callable[[], Any], on_success: Callable[[Any], None]) -> None:
        try:
            future = self._executor.submit(task)
        except Exception as exc:
            self._task_failed(exc)
            return
        future.add_done_callback(
            lambda completed_future: self._task_results.put((completed_future, on_success))
        )

    def _schedule_task_poll(self) -> None:
        if self._closed:
            return
        try:
            self._task_poll_id = self.root.after(50, self._drain_tasks)
        except tk.TclError:
            self._task_poll_id = None

    def _drain_tasks(self) -> None:
        self._task_poll_id = None
        try:
            while True:
                try:
                    future, on_success = self._task_results.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._complete_task(future, on_success)
                except Exception as exc:
                    self._task_failed(exc)
        finally:
            self._schedule_task_poll()

    def _complete_task(self, future: Future[Any], on_success: Callable[[Any], None]) -> None:
        if self._closed:
            return
        self._set_busy(False)
        try:
            value = future.result()
        except Exception as exc:
            self._task_failed(exc)
            return
        on_success(value)

    def _task_failed(self, exc: Exception) -> None:
        self._set_busy(False)
        self.status_var.set("Action failed")
        window = self._registration_window
        try:
            if window is not None:
                messagebox.showerror("Action failed", str(exc), parent=window)
            else:
                messagebox.showerror("Action failed", str(exc))
        except tk.TclError:
            pass

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.login_button, self.logout_button, self.register_button):
            button.configure(state=state)
        if self.remove_button is not None:
            self.remove_button.configure(state=state)
        if self.registration_accept_button is not None:
            self.registration_accept_button.configure(state=state)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        if self._task_poll_id is not None:
            self.root.after_cancel(self._task_poll_id)
        self._close_registration()
        try:
            self.camera.release()
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self.root.destroy()


def run_gui(config: AppConfig) -> int:
    try:
        runtime = build_runtime(config)
        app = FaceAttendanceApp(runtime)
        app.run()
    except tk.TclError as exc:
        raise DependencyError("A usable Tkinter display is required for the desktop app") from exc
    return 0
