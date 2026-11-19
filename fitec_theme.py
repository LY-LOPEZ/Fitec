"""Day/night appearance for FITEC's native desktop widgets and charts."""

from contextlib import contextmanager
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from weakref import WeakKeyDictionary


@dataclass(frozen=True)
class NightColors:
    background: str = "#101214"
    navigation: str = "#131b1e"
    panel: str = "#191d20"
    input: str = "#252b2e"
    text: str = "#e7edef"
    muted: str = "#afbec3"
    border: str = "#384449"
    selection: str = "#255851"
    focus: str = "#66dac9"
    plot: str = "#111517"
    grid: str = "#354044"
    inactive_bar: str = "#3f686c"
    shade: str = "#4e9f96"
    reference: str = "#ffd290"


NIGHT = NightColors()

STYLES = (
    ".", "TFrame", "Panel.TFrame", "Header.TFrame", "Mint.TFrame",
    "TLabel", "Help.TLabel", "Section.TLabel", "Error.TLabel",
    "TEntry", "TSpinbox", "TCombobox", "TButton", "Primary.TButton",
    "Toolbar.TButton", "Theme.TButton", "Menu.TMenubutton",
    "Toolbutton", "Nav.Toolbutton", "Graph.Toolbutton", "Surface.Graph.Toolbutton", "TCheckbutton",
    "TRadiobutton", "TSeparator", "Horizontal.TScale", "TScrollbar",
    "TNotebook", "TNotebook.Tab", "Sample.TNotebook", "Sample.TNotebook.Tab",
    "Treeview", "Treeview.Heading", "Sample.Treeview", "Sample.Treeview.Heading",
)
WIDGET_COLORS = (
    "background", "foreground", "disabledforeground", "activebackground", "activeforeground",
    "highlightbackground", "highlightcolor", "insertbackground", "selectbackground", "selectforeground",
    "selectcolor", "readonlybackground", "disabledbackground", "troughcolor",
)
BACKGROUND_COLORS = {
    "white": NIGHT.panel, "#ffffff": NIGHT.panel, "#edf3f5": NIGHT.background,
    "#f4f8f9": NIGHT.panel, "#f1f5f6": NIGHT.input, "#f3f7f8": NIGHT.input,
    "#f4faf9": NIGHT.plot, "#e1f5f0": "#172c2b", "#dff3ef": "#263b39",
    "#102f3d": NIGHT.navigation, "#103746": NIGHT.panel, "#204a59": "#223438",
    "#17646b": NIGHT.selection, "#748a8c": NIGHT.input, "#e8edf0": NIGHT.input,
    "#d9d9d9": NIGHT.panel, "#dcdad5": NIGHT.panel, "systembuttonface": NIGHT.panel,
    "systemwindow": NIGHT.input, "systemhighlight": NIGHT.selection, "systemmenu": NIGHT.panel,
}
TEXT_COLORS = {
    "black": NIGHT.text, "#000000": NIGHT.text, "#153744": NIGHT.text, "#103746": NIGHT.text,
    "#506671": NIGHT.muted, "#687d84": NIGHT.muted, "#075a56": NIGHT.focus, "#087d77": NIGHT.focus,
    "#a52620": "#ffc0b9", "systemwindowtext": NIGHT.text, "systembuttontext": NIGHT.text,
    "systemgraytext": NIGHT.muted, "systemhighlighttext": NIGHT.text, "systemmenutext": NIGHT.text,
}


def is_night(widget):
    theme = getattr(widget.nametowidget("."), "_fitec_theme", None)
    return theme is not None and theme.night


def _night_color(option, value):
    key = str(value).lower()
    if option == "fieldbackground" and key in ("white", "#ffffff"):
        return NIGHT.input
    if option in ("highlightbackground", "bordercolor", "lightcolor", "darkcolor"):
        if key in ("#d6e3e7", "#a5d7cf", "#9e9a91", "#ffffff", "white", "#f4f8f9",
                   "systembuttonface", "systemwindow", "#102f3d", "#dcdad5", "#cfcdc8"):
            return NIGHT.border
    if option == "selectbackground":
        return NIGHT.selection
    if option in ("insertbackground", "insertcolor"):
        return NIGHT.text
    if option.endswith("foreground") or option in ("arrowcolor", "indicatorcolor"):
        return TEXT_COLORS.get(key, value)
    return BACKGROUND_COLORS.get(key, value)


class FitecTheme:
    def __init__(self, root):
        self.root = root
        root._fitec_theme = self
        self.style = ttk.Style(root)
        self.day_theme = self.style.theme_use()
        self.name = "fitec-night"
        self.night = False
        self._originals = WeakKeyDictionary()
        self._dropdowns = {}
        root.bind_all("<Map>", self._on_map, add="+")

    def _night_settings(self):
        settings = {}
        for name in STYLES:
            configuration = self.style.configure(name) or {}
            mapping = self.style.map(name) or {}
            settings[name] = {
                "configure": {key: _night_color(key, value) for key, value in configuration.items()},
                "map": {key: [(*item[:-1], _night_color(key, item[-1])) for item in states]
                        for key, states in mapping.items()},
            }
        settings["Menu.TMenubutton"]["layout"] = self.style.layout("Menu.TMenubutton")
        settings["."]["configure"].update(
            background=NIGHT.panel, foreground=NIGHT.text, fieldbackground=NIGHT.input,
            bordercolor=NIGHT.border, lightcolor=NIGHT.border, darkcolor=NIGHT.border,
            troughcolor=NIGHT.background, focuscolor=NIGHT.focus, arrowcolor=NIGHT.text,
            selectbackground=NIGHT.selection, selectforeground=NIGHT.text, insertcolor=NIGHT.text,
        )
        settings["."]["map"].update(
            foreground=[("disabled", NIGHT.muted)],
            background=[("active", NIGHT.input)],
        )
        for name in ("TEntry", "TSpinbox", "TCombobox"):
            settings[name]["configure"].update(
                fieldbackground=NIGHT.input, foreground=NIGHT.text, bordercolor=NIGHT.border,
                insertcolor=NIGHT.text, arrowcolor=NIGHT.text,
            )
            settings[name]["map"].update(
                fieldbackground=[("disabled", NIGHT.panel), ("readonly", NIGHT.input)],
                foreground=[("disabled", NIGHT.muted), ("readonly", NIGHT.text)],
                bordercolor=[("focus", NIGHT.focus)],
            )
        for name in ("Treeview", "Sample.Treeview"):
            settings[name]["configure"].update(
                background=NIGHT.panel, fieldbackground=NIGHT.panel, foreground=NIGHT.text,
            )
            settings[name]["map"].update(
                background=[("selected", NIGHT.selection)],
                foreground=[("selected", NIGHT.text)],
            )
        for name in ("TNotebook.Tab", "Sample.TNotebook.Tab"):
            settings[name]["map"].update(
                foreground=[("selected", NIGHT.focus), ("disabled", NIGHT.muted)],
                background=[("selected", NIGHT.input), ("active", NIGHT.input)],
            )
        return settings

    def set_night(self, enabled):
        if enabled:
            self.style.theme_use(self.day_theme)
            if self.name not in self.style.theme_names():
                self.style.theme_create(self.name, parent=self.day_theme)
            self.style.theme_settings(self.name, self._night_settings())
            self.style.theme_use(self.name)
        else:
            self.style.theme_use(self.day_theme)
        self.night = enabled
        self._recolor_tree(self.root)

    @contextmanager
    def building_window(self):
        # Embedded explorers configure the day theme during construction.
        try:
            yield
        finally:
            if self.night:
                self.set_night(True)

    def update_colors(self, widget, **colors):
        day = self._day_colors(widget)
        day.update(colors)
        widget.configure(**{key: _night_color(key, value) if self.night else value
                            for key, value in colors.items()})

    def _recolor_tree(self, widget):
        self._recolor(widget)
        for child in widget.winfo_children():
            self._recolor_tree(child)

    def _recolor(self, widget):
        day = self._day_colors(widget)
        changes = {key: _night_color(key, value) if self.night else value
                   for key, value in day.items() if str(value)}
        if isinstance(widget, ttk.Combobox):
            self._recolor_dropdown(widget)
        if changes:
            widget.configure(**changes)

    def _day_colors(self, widget):
        if widget not in self._originals:
            keys = widget.keys()
            self._originals[widget] = {key: widget.cget(key) for key in WIDGET_COLORS if key in keys}
        return self._originals[widget]

    def _recolor_dropdown(self, widget):
        popup = f"{widget}.popdown.f.l"
        if not widget.tk.getboolean(widget.tk.call("winfo", "exists", popup)):
            return
        if popup not in self._dropdowns:
            self._dropdowns[popup] = tuple(widget.tk.call(popup, "cget", option) for option in
                                           ("-background", "-foreground", "-selectbackground", "-selectforeground"))
        values = (NIGHT.input, NIGHT.text, NIGHT.selection, NIGHT.text) if self.night else self._dropdowns[popup]
        args = [part for pair in zip(("-background", "-foreground", "-selectbackground", "-selectforeground"),
                                     values) for part in pair]
        widget.tk.call(popup, "configure", *args)

    def _on_map(self, event):
        if self.night and isinstance(event.widget, tk.Misc):
            self._recolor(event.widget)
        elif self.night and isinstance(event.widget, str) and event.widget.endswith(".popdown"):
            try:
                self._recolor_dropdown(self.root.nametowidget(event.widget[:-8]))
            except KeyError:
                pass
