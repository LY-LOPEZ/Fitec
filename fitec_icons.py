"""Icons and header marks shared by the native FITEC windows."""

from io import BytesIO
from pathlib import Path
import tkinter as tk
from weakref import WeakSet

from PIL import Image, ImageDraw

TEAL = "#087d77"
ASSETS = Path(__file__).with_name("probviz") / "assets"


class InterfaceIcon(tk.PhotoImage):
    def __init__(self, master, image, size):
        super().__init__(master=master)
        self.source, self.base_size = image, size
        root = master.nametowidget(".")
        if not hasattr(root, "_fitec_icons"):
            root._fitec_icons = WeakSet()
        root._fitec_icons.add(self)
        theme = getattr(root, "_fitec_theme", None)
        self.resize(theme.zoom / 100 if theme else 1)

    def resize(self, factor):
        size = tuple(max(1, round(length * factor)) for length in self.base_size)
        if size == (self.width(), self.height()):
            return
        with BytesIO() as data:
            self.source.resize(size, Image.Resampling.LANCZOS).save(data, "PNG")
            self.configure(data=data.getvalue(), format="png")


def add_header_mark(header, filename, label, size, **placement):
    badge = tk.Label(header, text=label, background=header.cget("background"), foreground="white",
                     font=("Inter", 12, "bold"), borderwidth=0)
    try:
        with Image.open(ASSETS / filename) as source:
            logo = source.convert("RGBA")
            bounds = logo.getbbox()
            if bounds:
                logo = logo.crop(bounds)
            height = max(1, round(header.winfo_fpixels(size)))
            width = max(1, round(logo.width * height / logo.height))
        badge.logo_image = InterfaceIcon(header, logo, (width, height))
        badge.configure(image=badge.logo_image)
    except OSError:
        pass
    badge.pack(**placement)


def add_usfx_mark(header):
    tk.Label(header, text="USFX", background=header.cget("background"), foreground="white",
             font=("Inter", 18, "bold")).pack(side="right", padx=(0, 14))
    add_header_mark(header, "usfx.png", "Escudo de la USFX", "37p", side="right", padx=(8, 8), pady=6)

ICON_PATHS = {
    "table": (((2, 3), (18, 3), (18, 17), (2, 17), (2, 3)),
              ((2, 8), (18, 8)), ((2, 12), (18, 12)), ((8, 3), (8, 17))),
    "chart": (((2, 3), (2, 17), (18, 17)), ((5, 12), (9, 8), (12, 10), (17, 4))),
    "model": (((2, 16), (5, 14), (8, 7), (10, 4), (12, 7), (15, 14), (18, 16)),),
    "add": (((10, 3), (10, 17)), ((3, 10), (17, 10))),
    "open": (((2, 16), (2, 5), (8, 5), (10, 7), (18, 7), (16, 16), (2, 16)),),
    "paste": (((7, 4), (4, 4), (4, 17), (16, 17), (16, 4), (13, 4)),
              ((7, 2), (13, 2), (13, 6), (7, 6), (7, 2)), ((7, 10), (13, 10)), ((7, 13), (12, 13))),
    "remove": (((3, 5), (17, 5)), ((5, 5), (6, 17), (14, 17), (15, 5)), ((8, 3), (12, 3))),
    "settings": (((3, 5), (17, 5)), ((3, 10), (17, 10)), ((3, 15), (17, 15)),
                 ((7, 3), (7, 7)), ((13, 8), (13, 12)), ((9, 13), (9, 17))),
    "prepare": (((3, 5), (16, 5), (13, 2)), ((16, 5), (13, 8)),
                ((17, 15), (4, 15), (7, 12)), ((4, 15), (7, 18))),
    "arrow": (((3, 10), (17, 10), (12, 5)), ((17, 10), (12, 15))),
    "back": (((17, 10), (3, 10), (8, 5)), ((3, 10), (8, 15))),
    "density": (((2, 17), (5, 16), (7, 12), (10, 5), (12, 4), (14, 8), (16, 14), (18, 17)),),
    "mass": (((3, 17), (3, 12), (6, 12), (6, 17)), ((9, 17), (9, 4), (12, 4), (12, 17)),
             ((15, 17), (15, 8), (18, 8), (18, 17))),
    "cumulative": (((2, 17), (5, 17), (5, 13), (9, 13), (9, 8), (13, 8), (13, 4), (18, 4)),),
    "reset": (((4, 8), (4, 3), (4, 8), (9, 8)),
              ((4, 8), (7, 4), (12, 3), (16, 5), (18, 9), (17, 14), (13, 17), (8, 17), (4, 14))),
    "save": (((3, 3), (15, 3), (18, 6), (18, 17), (3, 17), (3, 3)),
             ((7, 3), (7, 8), (14, 8), (14, 3)), ((6, 17), (6, 12), (15, 12), (15, 17))),
    "dice": (((3, 3), (17, 3), (17, 17), (3, 17), (3, 3)),
             ((6, 6), (7, 7)), ((9.5, 9.5), (10.5, 10.5)), ((13, 13), (14, 14))),
    "zoom_in": (((8, 2), (12, 4), (14, 8), (12, 12), (8, 14), (4, 12), (2, 8), (4, 4), (8, 2)),
                ((12, 12), (18, 18)), ((5, 8), (11, 8)), ((8, 5), (8, 11))),
    "zoom_out": (((8, 2), (12, 4), (14, 8), (12, 12), (8, 14), (4, 12), (2, 8), (4, 4), (8, 2)),
                 ((12, 12), (18, 18)), ((5, 8), (11, 8))),
    "zoom": (((8, 2), (12, 4), (14, 8), (12, 12), (8, 14), (4, 12), (2, 8), (4, 4), (8, 2)),
             ((12, 12), (18, 18))),
    "help": (((10, 2), (15, 4), (18, 9), (17, 14), (13, 17), (8, 18), (3, 15), (2, 10),
              (4, 5), (10, 2)), ((7, 8), (8, 6), (11, 6), (13, 8), (11, 10), (10, 11)),
             ((10, 14), (10, 15))),
    "team": (((8, 3), (11, 4), (12, 7), (10, 9), (7, 9), (5, 7), (5, 5), (8, 3)),
             ((3, 17), (3, 14), (6, 12), (10, 12), (13, 14), (13, 17)),
             ((14, 5), (16, 6), (16, 9), (14, 10)), ((15, 12), (18, 14), (18, 17))),
    "assistant": (((9, 2), (11, 7), (16, 9), (11, 11), (9, 17), (7, 11), (2, 9), (7, 7), (9, 2)),
                  ((16, 2), (16, 6)), ((14, 4), (18, 4)), ((16, 13), (16, 17)), ((14, 15), (18, 15))),
    "moon": (((11, 2), (8, 3), (5, 5), (3, 9), (4, 13), (7, 16), (11, 17),
              (15, 15), (17, 11), (13, 12), (10, 10), (9, 6), (11, 2)),),
    "sun": (((10, 6), (13, 7), (14, 10), (13, 13), (10, 14), (7, 13), (6, 10), (7, 7), (10, 6)),
            ((10, 2), (10, 4)), ((10, 16), (10, 18)), ((2, 10), (4, 10)), ((16, 10), (18, 10)),
            ((4, 4), (5, 5)), ((15, 15), (16, 16)), ((4, 16), (5, 15)), ((15, 5), (16, 4))),
}


def make_icons(root, *, color=TEAL, names=None, size=None):
    if size is None:
        size = max(16, round(root.winfo_fpixels("15p")))
    images = {}
    for name, paths in {**ICON_PATHS, "fit": ICON_PATHS["chart"]}.items():
        if names is not None and name not in names:
            continue
        image = Image.new("RGBA", (80, 80))
        draw = ImageDraw.Draw(image)
        ink = "white" if name == "fit" else color
        for path in paths:
            draw.line([(4 * x, 4 * y) for x, y in path], fill=ink, width=6, joint="curve")
        images[name] = InterfaceIcon(root, image, (size, size))
    return images
