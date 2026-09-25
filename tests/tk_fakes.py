"""In-process stand-ins for Tkinter and Pillow.

Tkinter's widget geometry is implemented by Tcl, so a headless test cannot
drive a real widget tree. These fakes record what the application *asks* for
and replay scheduled callbacks on demand, which lets the GUI's own logic --
the attendance state machine, registration flow, task polling, and error
handling -- be exercised without a display. They intentionally do not reimplement
Tk; every method here corresponds to a call the application makes.
"""

from __future__ import annotations

import tkinter as real_tk
from typing import ClassVar

TclError = real_tk.TclError


class FakeVar:
    """A mutable string holder standing in for ``tk.StringVar``."""

    def __init__(self, value="") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class FakeWidget:
    """A generic widget recording layout and configuration calls."""

    def __init__(self, parent=None, **options) -> None:
        self.parent = parent
        self.options = dict(options)
        self.children: list[FakeWidget] = []
        self.pack_options: dict = {}
        self.grid_options: dict = {}
        self.destroyed = False
        self.title_text: str | None = None
        self.geometry_text: str | None = None
        self.minsize_args: tuple = ()
        self.protocols: dict[str, object] = {}
        if isinstance(parent, FakeWidget):
            parent.children.append(self)

    def title(self, text) -> None:
        self.title_text = text

    def geometry(self, text) -> None:
        self.geometry_text = text

    def minsize(self, *args) -> None:
        self.minsize_args = args

    def protocol(self, name, callback) -> None:
        self.protocols[name] = callback

    def pack(self, **kwargs) -> None:
        self.pack_options = kwargs

    def grid(self, **kwargs) -> None:
        self.grid_options = kwargs

    def configure(self, **kwargs) -> None:
        self.options.update(kwargs)

    config = configure

    def winfo_width(self) -> int:
        return 640

    def winfo_height(self) -> int:
        return 480

    def winfo_exists(self) -> bool:
        return not self.destroyed

    def destroy(self) -> None:
        self.destroyed = True

    def columnconfigure(self, *args, **kwargs) -> None:
        return None

    def rowconfigure(self, *args, **kwargs) -> None:
        return None


class FakeListbox(FakeWidget):
    """A listbox holding selectable string items."""

    def __init__(self, parent=None, **options) -> None:
        super().__init__(parent, **options)
        self.items: list[str] = []
        self.selection: tuple[int, ...] = ()

    def delete(self, start, end=None) -> None:
        self.items = []

    def insert(self, index, value) -> None:
        self.items.append(value)

    def curselection(self) -> tuple[int, ...]:
        return self.selection

    def get(self, index) -> str:
        return self.items[index]

    def select(self, index: int) -> None:
        self.selection = (index,)


class FakeRoot(FakeWidget):
    """A root window whose ``after`` callbacks are replayed manually."""

    def __init__(self) -> None:
        super().__init__(None)
        self.scheduled: list[tuple[str, int, object, tuple]] = []
        self.mainloop_calls = 0

    def after(self, delay, callback=None, *args) -> str:
        handle = f"after#{len(self.scheduled)}"
        self.scheduled.append((handle, delay, callback, args))
        return handle

    def after_cancel(self, handle) -> None:
        self.scheduled = [item for item in self.scheduled if item[0] != handle]

    def mainloop(self) -> None:
        self.mainloop_calls += 1

    def fire_after(self, delay: int | None = None) -> None:
        """Invoke the callback registered for ``delay``.

        Args:
            delay: The delay the callback was scheduled with. When ``None``,
                the most recently scheduled callback is used.

        Raises:
            LookupError: If no callback is registered for that delay.
        """
        candidates = [item for item in self.scheduled if delay is None or item[1] == delay]
        if not candidates:
            raise LookupError(f"no callback scheduled for delay={delay}")
        handle, _, callback, args = candidates[-1]
        self.scheduled = [item for item in self.scheduled if item[0] != handle]
        callback(*args)


class FakeToplevel(FakeWidget):
    """A secondary window with modal-window behaviour."""

    def __init__(self, master=None) -> None:
        super().__init__(master)
        self.transient_for = None
        self.grabbed = False

    def transient(self, master) -> None:
        self.transient_for = master

    def grab_set(self) -> None:
        self.grabbed = True


class FakeStyle:
    """A ``ttk.Style`` reporting a configurable theme list."""

    def __init__(self, themes) -> None:
        self._themes = list(themes)
        self.used: str | None = None
        self.configured: dict[str, dict] = {}

    def theme_names(self) -> list[str]:
        return list(self._themes)

    def theme_use(self, name) -> None:
        self.used = name

    def configure(self, name, **kwargs) -> None:
        self.configured[name] = kwargs


class FakeTkinter:
    """The ``tkinter`` namespace as the application sees it."""

    TclError = TclError
    END = "end"
    BROWSE = "browse"

    def __init__(self) -> None:
        self.roots: list[FakeRoot] = []
        self.toplevels: list[FakeToplevel] = []
        self.vars: list[FakeVar] = []
        self.listboxes: list[FakeListbox] = []

    def Tk(self) -> FakeRoot:
        root = FakeRoot()
        self.roots.append(root)
        return root

    def Toplevel(self, master=None) -> FakeToplevel:
        window = FakeToplevel(master)
        self.toplevels.append(window)
        return window

    def StringVar(self, value="") -> FakeVar:
        var = FakeVar(value)
        self.vars.append(var)
        return var

    def Listbox(self, parent=None, **options) -> FakeListbox:
        box = FakeListbox(parent, **options)
        self.listboxes.append(box)
        return box


class FakeTtk:
    """The ``ttk`` namespace as the application sees it."""

    def __init__(self, themes=("clam",)) -> None:
        self._themes = list(themes)
        self.styles: list[FakeStyle] = []
        self.widgets: list[FakeWidget] = []

    def _make(self, kind, parent, options):
        widget = FakeWidget(parent, **options)
        widget.kind = kind
        self.widgets.append(widget)
        return widget

    def Style(self, master=None) -> FakeStyle:
        style = FakeStyle(self._themes)
        self.styles.append(style)
        return style

    def Frame(self, parent=None, **options) -> FakeWidget:
        return self._make("frame", parent, options)

    def Label(self, parent=None, **options) -> FakeWidget:
        return self._make("label", parent, options)

    def LabelFrame(self, parent=None, **options) -> FakeWidget:
        return self._make("labelframe", parent, options)

    def Button(self, parent=None, **options) -> FakeWidget:
        return self._make("button", parent, options)

    def Entry(self, parent=None, **options) -> FakeWidget:
        return self._make("entry", parent, options)


class FakeMessagebox:
    """A message box recording every prompt and answering confirmations."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict]] = []
        self.confirm = True

    def _record(self, kind, title, message, **kwargs) -> None:
        self.calls.append((kind, title, message, kwargs))

    def showinfo(self, title, message, **kwargs) -> None:
        self._record("info", title, message, **kwargs)

    def showwarning(self, title, message, **kwargs) -> None:
        self._record("warning", title, message, **kwargs)

    def showerror(self, title, message, **kwargs) -> None:
        self._record("error", title, message, **kwargs)

    def askyesno(self, title, message, **kwargs) -> bool:
        self._record("askyesno", title, message, **kwargs)
        return self.confirm

    def of_kind(self, kind: str) -> list[tuple[str, str, str, dict]]:
        return [call for call in self.calls if call[0] == kind]


class FakeImage:
    """A Pillow image stub that records thumbnail requests."""

    def __init__(self, frame) -> None:
        self.frame = frame
        self.thumbnail_sizes: list[tuple[int, int]] = []

    def thumbnail(self, size) -> None:
        self.thumbnail_sizes.append(size)


class FakeImageModule:
    """The ``PIL.Image`` namespace as the application sees it."""

    created: ClassVar[list[FakeImage]] = []

    @classmethod
    def fromarray(cls, frame) -> FakeImage:
        image = FakeImage(frame)
        cls.created.append(image)
        return image


class FakeImageTkModule:
    """The ``PIL.ImageTk`` namespace as the application sees it."""

    photos: ClassVar[list[object]] = []

    @classmethod
    def PhotoImage(cls, image=None):
        photo = ("photo", image)
        cls.photos.append(photo)
        return photo


class GuiHarness:
    """Installed Tkinter fakes plus the handles needed to drive them."""

    def __init__(self, themes=("clam",)) -> None:
        self.tk = FakeTkinter()
        self.ttk = FakeTtk(themes)
        self.messagebox = FakeMessagebox()

    @property
    def root(self) -> FakeRoot:
        return self.tk.roots[-1]

    def install(self, monkeypatch) -> GuiHarness:
        """Replace the GUI module's Tkinter references with these fakes.

        Args:
            monkeypatch: The active ``monkeypatch`` fixture.

        Returns:
            This harness, for chaining.
        """
        FakeImageModule.created = []
        FakeImageTkModule.photos = []
        monkeypatch.setattr("face_attendance.gui.tk", self.tk)
        monkeypatch.setattr("face_attendance.gui.ttk", self.ttk)
        monkeypatch.setattr("face_attendance.gui.messagebox", self.messagebox)
        monkeypatch.setattr("face_attendance.gui._load_gui_dependencies", lambda: None)
        self.set_image_dependencies(monkeypatch)
        return self

    def set_image_dependencies(
        self,
        monkeypatch,
        image=FakeImageModule,
        image_tk=FakeImageTkModule,
    ) -> GuiHarness:
        """Point the GUI's image helpers at specific modules.

        Args:
            monkeypatch: The active ``monkeypatch`` fixture.
            image: Module to use as ``PIL.Image``.
            image_tk: Module to use as ``PIL.ImageTk``.

        Returns:
            This harness, for chaining.
        """
        monkeypatch.setattr("face_attendance.gui._CV2", object(), raising=False)
        monkeypatch.setattr("face_attendance.gui._IMAGE", image, raising=False)
        monkeypatch.setattr("face_attendance.gui._IMAGE_TK", image_tk, raising=False)
        return self
