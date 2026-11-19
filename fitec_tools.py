"""FITEC's desktop workspace and sample generator."""

import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from fitec_assistant import FitecAssistant
from fitec_demo import Galton, NormalSurface
from fitec_icons import make_icons
from fitec_menus import FitecMenus
from fitec_queues import QueueSimulator
from fitec_theme import FitecTheme, NIGHT, is_night


PAPER = "#edf3f5"
INK = "#103746"
HEADER_INK = "#0e1117"
SIDEBAR = "#102f3d"
BANNER_TEXT = "#c9e1e5"
TEAL = "#087d77"
PREVIEW_ROWS = 200
ASSETS = Path(__file__).with_name("probviz") / "assets"
WORKSPACES = (
    ("distributions", "Distribuciones", "model"),
    ("random", "Números aleatorios", "dice"),
    ("assistant", "Asistente FITEC", "assistant"),
    ("queues", "Simulador de filas", "team"),
)


def toolbar_button(parent, icons, label, icon, command, *, primary=False):
    return ttk.Button(parent, text=label, image=icons[icon], compound="left",
                      command=command, style="Primary.TButton" if primary else "Toolbar.TButton")


class FitecTools:
    def __init__(self, root, explorer_type, parse_parameter, parse_observations):
        root.title("FI-TEC – Probabilidad")
        if root.tk.call("tk", "windowingsystem") == "win32":
            try:
                root.iconbitmap(default=str(ASSETS / "FI-TEC-Dado.ico"))
            except tk.TclError:
                pass
        width = min(1420, root.winfo_screenwidth() - 40)
        height = min(820, root.winfo_screenheight() - 88)
        root.geometry(f"{width}x{height}")
        root.minsize(min(1080, width), min(620, height))
        root.configure(background=PAPER)
        self.icons = make_icons(root)
        symbol_size = max(20, round(root.winfo_fpixels("18p")))
        self.nav_icons = make_icons(root, color="#d0ebeb", names=("chart", "team"), size=symbol_size)
        for _, _, icon in WORKSPACES:
            if icon in self.nav_icons:
                continue
            with Image.open(ASSETS / f"symbol-{icon}.png") as source:
                glyph = source.resize((symbol_size, symbol_size), Image.Resampling.LANCZOS)
            self.nav_icons[icon] = ImageTk.PhotoImage(glyph, master=root)

        self.content = tk.Frame(root, background=PAPER)
        self.content.pack(fill="both", expand=True)
        self.sidebar = tk.Frame(self.content, background=SIDEBAR)
        self.sidebar.pack(side="left", fill="y")
        brand = tk.Frame(self.sidebar, background=SIDEBAR)
        brand.pack(fill="x")
        self._add_header_mark(root, brand, "symbol-brand.png", "", "29p",
                              side="left", padx=(14, 4), pady=16)
        tk.Label(brand, text="FI-TEC", background=SIDEBAR, foreground="white",
                 font=("Inter", 20, "bold")).pack(side="left", padx=(0, 12))
        workspace = tk.Frame(self.content, background=PAPER)
        workspace.pack(side="left", fill="both", expand=True)
        banner = tk.Frame(workspace, background=SIDEBAR)
        banner.pack(fill="x", padx=12)
        tk.Label(banner, text="USFX", background=SIDEBAR, foreground="white",
                 font=("Inter", 18, "bold")).pack(side="right", padx=(0, 14))
        self._add_header_mark(root, banner, "usfx.png", "Escudo de la USFX", "37p",
                              side="right", padx=(8, 8), pady=6)
        self._add_header_mark(root, banner, "fitec-blender.png", "", "41p",
                              side="right", padx=(4, 8))
        headings = tk.Frame(banner, background=SIDEBAR)
        headings.pack(side="left", padx=18, pady=9)
        self.heading = tk.Label(headings, background=SIDEBAR, foreground="white",
                                font=("Inter", 19, "bold"), borderwidth=0)
        self.heading.pack(side="left")
        tk.Frame(headings, background=BANNER_TEXT, width=1, height=23).pack(side="left", padx=12)
        tk.Label(headings, text="Simulación de Sistemas", background=SIDEBAR, foreground=BANNER_TEXT,
                 font=("Inter", 11), borderwidth=0).pack(side="left")
        self.corner_radius = max(6, round(root.winfo_fpixels("7p")))
        self.banner_corners = []
        for anchor, x, y in (("nw", 0, 0), ("ne", 1, 0), ("sw", 0, 1), ("se", 1, 1)):
            corner = tk.Label(banner, borderwidth=0, highlightthickness=0, padx=0, pady=0)
            corner.place(relx=x, rely=y, anchor=anchor)
            self.banner_corners.append((corner, x, y))
        self._paint_banner_corners(False)

        self.toolbar = ttk.Frame(workspace, style="Panel.TFrame", padding=(10, 6))
        page_host = ttk.Frame(workspace)
        self.page_host = page_host
        page_host.pack(fill="both", expand=True)
        page_host.rowconfigure(0, weight=1)
        page_host.columnconfigure(0, weight=1)
        self.pages = {name: ttk.Frame(page_host) for name, _, _ in WORKSPACES}
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
        self.explorer = explorer_type(self.pages["distributions"], embedded=True, show_moments=True,
                                      external_controls=True)
        self.explorer.stats_frame.pack_forget()
        style = ttk.Style(root)
        style.configure("Nav.Toolbutton", padding=(12, 11), font=("Inter", 10, "bold"),
                        background=SIDEBAR, foreground="#e9f5f6", bordercolor=SIDEBAR,
                        lightcolor=SIDEBAR, darkcolor=SIDEBAR, anchor="w")
        style.map("Nav.Toolbutton",
                  background=[("selected", "#17646b"), ("active", "#204a59")],
                  bordercolor=[("focus", "#61cdc6"), ("selected", "#17646b")],
                  lightcolor=[("selected", "#17646b"), ("active", "#204a59")],
                  darkcolor=[("selected", "#17646b"), ("active", "#204a59")])
        style.configure("Graph.Toolbutton", padding=(7, 5), font=("Inter", 9, "bold"),
                        background="#f3f7f8", foreground=INK, bordercolor="#d6e3e7")
        style.map("Graph.Toolbutton", foreground=[("disabled", "#687d84"), ("selected", "white")],
                  background=[("disabled", PAPER), ("selected", TEAL), ("active", PAPER)],
                  bordercolor=[("selected", "#96d6ce")])
        style.configure("Surface.Graph.Toolbutton", borderwidth=0)
        style.configure("Toolbar.TButton", padding=(7, 5), font=("Inter", 9, "bold"),
                        foreground=INK, background="white", bordercolor="#d6e3e7")
        style.map("Toolbar.TButton", foreground=[("disabled", "#687d84")],
                  background=[("disabled", PAPER), ("active", "#dff3ef")])
        style.configure("Primary.TButton", padding=(10, 7), foreground="white", background=TEAL)
        style.map("Primary.TButton", background=[("active", "#086863"), ("disabled", "#748a8c")])

        self.active_view = tk.StringVar(value="distributions")
        self.toolbar_visible = True
        self.commands = {
            name: ttk.Frame(self.explorer.graph_actions if name == "distributions" else self.toolbar,
                            style="Panel.TFrame")
            for name in self.pages if name != "queues"
        }
        self._build_distribution_commands(self.commands["distributions"])
        self.generator = RandomNumbers(self.pages["random"], self.commands["random"], self.icons,
                                       self.explorer.catalog, parse_parameter, parse_observations)
        self.assistant = FitecAssistant(self.pages["assistant"], self.commands["assistant"], self,
                                        parse_parameter, parse_observations)
        self.queues = QueueSimulator(self.pages["queues"])
        self.explorer.mode.trace_add("write", self._sync_distribution_commands)
        self._sync_distribution_commands()
        self.select_view("distributions")
        self.menus = FitecMenus(root, self)
        ttk.Button(self.sidebar, text="Ajustar distribuciones", image=self.nav_icons["chart"], compound="left",
                   style="Nav.Toolbutton", command=self.menus.open_fit_workspace).pack(fill="x", padx=7, pady=(7, 0))
        for name, label, icon in WORKSPACES:
            ttk.Radiobutton(self.sidebar, text=label, image=self.nav_icons[icon], compound="left",
                            value=name, variable=self.active_view, style="Nav.Toolbutton",
                            command=lambda name=name: self.select_view(name)).pack(fill="x", padx=7, pady=(7, 0))
        self.theme = FitecTheme(root)

    def _paint_banner_corners(self, night):
        radius = self.corner_radius
        disk = Image.new("RGB", (8 * radius, 8 * radius), NIGHT.background if night else PAPER)
        ImageDraw.Draw(disk).ellipse((0, 0, disk.width - 1, disk.height - 1),
                                     fill=NIGHT.navigation if night else SIDEBAR)
        disk = disk.resize((2 * radius, 2 * radius), Image.Resampling.LANCZOS)
        for corner, x, y in self.banner_corners:
            tile = disk.crop((x * radius, y * radius, (x + 1) * radius, (y + 1) * radius))
            corner.image = ImageTk.PhotoImage(tile, master=corner)
            corner.configure(image=corner.image)

    def toggle_theme(self):
        night = not self.theme.night
        self.theme.set_night(night)
        self.menus.update_theme_control(night)
        self._paint_banner_corners(night)
        self.explorer.update_plot()
        self.generator.repaint_theme()
        self.queues.repaint_theme()
        workspace = self.menus.fit_workspace
        if workspace is not None and workspace.root.winfo_exists():
            workspace.explorer.update_plot()
            workspace._draw_results()

    @staticmethod
    def _add_header_mark(root, header, filename, label, size, *, viewport=None, **placement):
        path = ASSETS / filename
        badge = tk.Label(header, text=label, background=header.cget("background"), foreground="white",
                         font=("Inter", 12, "bold"), borderwidth=0)
        try:
            with Image.open(path) as source:
                logo = source.convert("RGBA")
                if viewport:
                    left, top, right, bottom = viewport
                    bounds = (round(left * logo.width), round(top * logo.height),
                              round(right * logo.width), round(bottom * logo.height))
                else:
                    bounds = logo.getbbox()
                if bounds:
                    logo = logo.crop(bounds)
                height = max(1, round(root.winfo_fpixels(size)))
                width = max(1, round(logo.width * height / logo.height))
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                logo = logo.resize((width, height), resampling)
            badge.logo_image = ImageTk.PhotoImage(logo, master=root)
            badge.configure(image=badge.logo_image)
        except OSError:
            pass
        badge.pack(**placement)

    def _build_distribution_commands(self, parent):
        self.density_button = ttk.Radiobutton(
            parent, text="Densidad", value="Densidad",
            variable=self.explorer.graph_mode, style="Graph.Toolbutton", command=self.explorer.update_plot,
        )
        self.density_button.pack(side="left", padx=(0, 3))
        ttk.Radiobutton(
            parent, text="Acumulada", value="Acumulada",
            variable=self.explorer.graph_mode, style="Graph.Toolbutton", command=self.explorer.update_plot,
        ).pack(side="left")
        self.explorer.surface = NormalSurface(parent, self.explorer)
        self.explorer.distribution.trace_add("write", self._sync_distribution_commands)
        toolbar_button(parent, self.icons, "Usar en generador", "arrow", self._transfer).pack(side="right")
        toolbar_button(parent, self.icons, "Restablecer", "reset", self.explorer.reset_values).pack(
            side="right", padx=(6, 5))

    def _sync_distribution_commands(self, *_):
        mode = self.explorer.mode.get()
        normal = mode == "Continuas" and self.explorer.distribution.get() == "Normal"
        self.explorer.surface.button.state(["!disabled" if normal else "disabled"])
        if not normal:
            self.explorer.surface.close()
        is_discrete = mode == "Discretas"
        self.density_button.configure(text="Masa" if is_discrete else "Densidad")
        self.density_button.state(["disabled" if mode == "Empírica" else "!disabled"])

    def select_view(self, name):
        self.active_view.set(name)
        self.heading.configure(text=next(label for key, label, _ in WORKSPACES if key == name))
        if name == "assistant":
            self.assistant.refresh_guide()
        if name != "queues":
            self.queues.pause()
        self.pages[name].tkraise()
        self.set_toolbar_visible(self.toolbar_visible)

    def set_toolbar_visible(self, enabled):
        self.toolbar_visible = enabled
        name = self.active_view.get()
        for commands in self.commands.values():
            commands.pack_forget()
        self.toolbar.pack_forget()
        if not enabled or name not in self.commands:
            return
        if name == "distributions":
            self.commands[name].pack(fill="x")
        else:
            self.toolbar.pack(fill="x", before=self.page_host, padx=12, pady=(8, 0))
            self.commands[name].pack(side="right")

    def session_settings(self):
        return {"view": self.active_view.get(), "explorer": self.explorer.session_settings(),
                "generator": self.generator.session_settings(), "queues": self.queues.session_settings()}

    def restore_session(self, settings):
        self.explorer.restore_session(settings["explorer"])
        self.generator.restore_session(settings["generator"])
        self.queues.restore_session(settings.get("queues"))
        self.select_view(settings["view"])

    def _transfer(self):
        self.generator.select_distribution(*self.explorer.generator_selection())
        self.select_view("random")


class RandomNumbers:
    def __init__(self, parent, toolbar, icons, catalog, parse_parameter, parse_observations):
        self.parent = parent
        self.models = {name: spec for group in catalog.values() for name, spec in group.items()}
        self.parse_parameter = parse_parameter
        self.parse_observations = parse_observations
        self.name = tk.StringVar(value="Uniforme")
        self.count = tk.StringVar(value="100")
        self.seed = tk.StringVar(value="42")
        self.parameters = {}
        self.sample = None
        self.sample_caption = ""
        self.sample_model = self.sample_observed = None
        self.status = tk.StringVar(value="Sin muestra")
        self._build()
        toolbar_button(toolbar, icons, "Generar", "fit", self.generate, primary=True).pack(
            side="left", padx=(0, 8))
        self.copy_button = toolbar_button(toolbar, icons, "Copiar muestra", "paste", self.copy_sample)
        self.copy_button.pack(side="left", padx=(0, 5))
        self.csv_button = toolbar_button(toolbar, icons, "Guardar CSV", "save", self.save_sample)
        self.csv_button.pack(side="left")
        for button in (self.copy_button, self.csv_button):
            button.state(["disabled"])
        self.select_distribution("Uniforme", {"a": "0", "b": "1"})

    def _build(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        body = ttk.Frame(self.parent, padding=(14, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        controls = ttk.Frame(body, style="Panel.TFrame", padding=18)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ttk.Label(controls, text="Configuración", style="Section.TLabel").pack(anchor="w")
        ttk.Label(controls, text="Distribución").pack(anchor="w", pady=(14, 4))
        picker = ttk.Combobox(controls, textvariable=self.name, width=22, state="readonly",
                              values=(*self.models, "Empírica"))
        picker.pack(fill="x")
        picker.bind("<<ComboboxSelected>>", lambda _event: self.select_distribution(self.name.get()))
        ttk.Separator(controls).pack(fill="x", pady=(12, 4))
        self.parameter_box = ttk.Frame(controls, style="Panel.TFrame")
        self.parameter_box.pack(fill="x")
        counts = ttk.Frame(controls, style="Panel.TFrame")
        counts.pack(fill="x", pady=(16, 8))
        for column, (caption, var) in enumerate((("Cantidad", self.count), ("Semilla", self.seed))):
            field = ttk.Frame(counts, style="Panel.TFrame")
            counts.columnconfigure(column, weight=1, uniform="sample-fields")
            field.grid(row=0, column=column, sticky="ew", padx=(0, 8) if column == 0 else 0)
            ttk.Label(field, text=caption).pack(anchor="w", pady=(0, 3))
            ttk.Entry(field, width=8, textvariable=var).pack(fill="x")
        ttk.Label(controls, text="La misma semilla y los mismos parámetros repiten la muestra.",
                  style="Help.TLabel", wraplength=206).pack(anchor="w", pady=(0, 8))
        self.error = ttk.Label(controls, style="Error.TLabel", wraplength=206)
        self.error.pack(anchor="w", pady=(6, 0))

        results = ttk.Frame(body, style="Panel.TFrame", padding=18)
        results.grid(row=0, column=1, sticky="nsew")
        ttk.Label(results, text="Muestra generada", style="Section.TLabel").pack(anchor="w")
        ttk.Label(results, textvariable=self.status, style="Help.TLabel",
                  wraplength=610).pack(anchor="w", pady=(6, 10))

        style = ttk.Style(self.parent)
        style.configure("Sample.TNotebook", borderwidth=0, background="white")
        style.configure("Sample.TNotebook.Tab", padding=(12, 7), font=("Inter", 10, "bold"),
                        background="#edf3f5", foreground="#506671")
        style.map("Sample.TNotebook.Tab", foreground=[("selected", "#075a56")],
                  background=[("selected", "#e1f5f0"), ("active", "#e1f5f0")])
        style.configure("Sample.Treeview", font=("Inter", 10), rowheight=28, borderwidth=0,
                        fieldbackground="white", background="white", foreground=INK)
        style.configure("Sample.Treeview.Heading", font=("Inter", 10, "bold"),
                        background="#edf3f5", foreground=INK, padding=(8, 7))
        style.map("Sample.Treeview", background=[("selected", "#e1f5f0")],
                  foreground=[("selected", "#075a56")])
        tabs = ttk.Notebook(results, style="Sample.TNotebook")
        tabs.pack(fill="both", expand=True)
        graph = ttk.Frame(tabs, style="Panel.TFrame")
        table = ttk.Frame(tabs, style="Panel.TFrame", padding=(0, 6))
        tabs.add(graph, text="Gráfica")
        tabs.add(table, text="Datos")
        self.galton = Galton(tabs, self)

        figure = Figure(figsize=(4.5, 3), dpi=100, facecolor="white")
        figure.subplots_adjust(left=.15, right=.97, bottom=.19, top=.94)
        self.sample_axes = figure.add_subplot()
        self.sample_canvas = FigureCanvasTkAgg(figure, master=graph)
        self.sample_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._plot_sample(None)

        self.preview = ttk.Treeview(table, columns=("index", "number"), show="headings",
                                    style="Sample.Treeview")
        self.preview.heading("index", text="#")
        self.preview.heading("number", text="Valor")
        self.preview.column("index", width=75, anchor="e", stretch=False)
        self.preview.column("number", width=245, anchor="e")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.preview.yview)
        self.preview.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.preview.pack(side="left", fill="both", expand=True)

    def select_distribution(self, name, parameters=None, observations=""):
        self.name.set(name)
        self.error.configure(text="")
        self.parameters.clear()
        for child in self.parameter_box.winfo_children():
            child.destroy()
        if name == "Empírica":
            ttk.Label(self.parameter_box, text="Observaciones").pack(anchor="w", pady=(8, 3))
            self.empirical = tk.Text(self.parameter_box, height=5, width=25, undo=True,
                                     font=("Inter", 10), relief="solid", borderwidth=1)
            self.empirical.pack(fill="x")
            self.empirical.insert("1.0", observations)
            self.empirical.edit_reset()
            return
        _, defaults, labels = self.models[name]
        for key, default in defaults.items():
            value = (parameters or {}).get(key, str(default).replace(".", ","))
            var = tk.StringVar(value=value)
            self.parameters[key] = var
            ttk.Label(self.parameter_box, text=labels[key], wraplength=245).pack(anchor="w", pady=(7, 3))
            ttk.Entry(self.parameter_box, textvariable=var).pack(fill="x")

    def generate(self):
        self.error.configure(text="")
        try:
            count, seed = int(self.count.get()), int(self.seed.get())
            if not 1 <= count <= 100_000 or not 0 <= seed <= 2**32 - 1:
                raise ValueError
        except ValueError:
            self.error.configure(text="Usa de 1 a 100 000 valores y una semilla entera de 0 a 4 294 967 295.")
            return
        name = self.name.get()
        model = None
        observed = None
        try:
            if name == "Empírica":
                observed = self.parse_observations(self.empirical.get("1.0", "end"))
                values = np.random.default_rng(seed).choice(observed, size=count, replace=True)
                discrete = bool(np.all(observed == np.floor(observed)))
            else:
                factory = self.models[name][0]
                parameters = {key: self.parse_parameter(name, key, var.get())
                              for key, var in self.parameters.items()}
                model = factory(**parameters)
                values = model.rvs(size=count, random_state=seed)
                discrete = model.is_discrete
            if not np.all(np.isfinite(values)):
                raise ValueError("Los parámetros produjeron valores no finitos.")
        except (ValueError, TypeError, OverflowError) as exc:
            self.error.configure(text=str(exc))
            return

        sample = [str(int(value)) if discrete else repr(float(value)).replace(".", ",") for value in values]
        self._show_sample(sample, f"{name} · {count} valores · Semilla {seed}", model=model, observed=observed)

    def _show_sample(self, sample, caption="", *, model=None, observed=None):
        self.sample = sample
        self.sample_caption = caption
        self.sample_model, self.sample_observed = model, observed
        self.preview.delete(*self.preview.get_children())
        if sample is None:
            self.status.set("Sin muestra")
        else:
            for index, value in enumerate(sample[:PREVIEW_ROWS], 1):
                self.preview.insert("", "end", values=(index, value))
            self.status.set(caption + (f" · Vista previa: primeros {PREVIEW_ROWS}"
                                       if len(sample) > PREVIEW_ROWS else ""))
        for button in (self.copy_button, self.csv_button):
            button.state(["disabled" if sample is None else "!disabled"])
        self._plot_sample(sample, model=model, observed=observed)

    def repaint_theme(self):
        self._plot_sample(self.sample, model=self.sample_model, observed=self.sample_observed)
        if hasattr(self.galton, "counts"):
            self.galton.paint()

    def _plot_sample(self, sample, *, model=None, observed=None):
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        ax = self.sample_axes
        ax.clear()
        night = is_night(self.parent)
        ax.figure.set_facecolor(NIGHT.panel if night else "white")
        ax.set_facecolor(NIGHT.plot if night else "white")
        text = NIGHT.text if night else INK
        bars = NIGHT.shade if night else TEAL
        reference = NIGHT.reference if night else INK
        if sample is None:
            self._sample_plot_message("Pulsa Generar para ver la gráfica de la muestra.")
            return

        # Use the serialized values so the plot and the table agree, including after restoring a project.
        values = np.fromiter((float(value.replace(",", ".")) for value in sample),
                             dtype=float, count=len(sample))
        discrete = all(value.lstrip("+-").isdigit() for value in sample)
        low, high = float(values.min()), float(values.max())
        span = high - low
        range_error = "El rango de la muestra no se puede representar en la gráfica."
        if not np.isfinite(span):
            self._sample_plot_message(range_error)
            return

        ax.set_axis_on()
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=NIGHT.grid if night else "#dce8ec")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(NIGHT.border if night else "#a8bec5")
        ax.tick_params(colors=NIGHT.muted if night else "#506671", labelsize=9, length=3)
        ax.set_xlabel("Valor", color=text, labelpad=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=discrete))
        formatter = FuncFormatter(lambda value, _position: f"{value:.5g}".replace(".", ","))
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)
        bins = max(1, min(32, int(np.ceil(np.sqrt(len(sample))))))
        if observed is not None:
            observed_curve = NIGHT.focus if night else TEAL
            for data, label, color in ((observed, "Datos originales", reference), (values, "Muestra", observed_curve)):
                points, counts = np.unique(data, return_counts=True)
                pad = max(float(np.ptp(observed)) * .04, .1)
                ax.step(np.r_[min(observed)-pad, points, max(observed)+pad],
                        np.r_[0, np.cumsum(counts)/len(data), 1], where="post", label=label, color=color)
            ax.set_ylabel("Probabilidad acumulada")
            if night:
                ax.yaxis.label.set_color(NIGHT.text)
        elif discrete and span <= 60:
            positions, counts = np.unique(values, return_counts=True)
            ax.bar(positions, counts / len(sample), width=.72, color=bars, alpha=.82, label="Muestra")
            if model is not None:
                x = np.arange(low, high + 1)
                ax.plot(x, model.pdf(x), "o", color=reference, label="Masa teórica")
            ax.set_ylabel("Frecuencia relativa", color=text, labelpad=7)
        else:
            # A wide discrete range uses grouped frequencies, while continuous bins use density.
            extent = None
            if span == 0.0:
                padding = max(.5, abs(low) * .005)
                extent = (low - padding, high + padding)
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    frequencies, edges = np.histogram(values, bins=bins, range=extent, density=not discrete)
            except (ValueError, IndexError, OverflowError):
                self._sample_plot_message(range_error)
                return
            if not np.all(np.isfinite(frequencies)):
                self._sample_plot_message(range_error)
                return
            if discrete:
                frequencies = frequencies / len(sample)
            ax.bar(edges[:-1], frequencies, width=np.diff(edges), align="edge",
                   color=bars, alpha=.64 if night else .52,
                   edgecolor=NIGHT.plot if night else "white", label="Muestra")
            ax.set_ylabel("Frecuencia relativa por intervalo" if discrete else "Densidad",
                          color=text, labelpad=7)
            if model is not None and discrete:
                lower = np.ceil(edges[:-1]) - 1
                upper = np.ceil(edges[1:]) - 1
                upper[-1] = np.floor(edges[-1])
                masses = np.where(model.cdf(lower) > .5, model.sf(lower)-model.sf(upper),
                                  model.cdf(upper)-model.cdf(lower))
                ax.plot(edges[:-1] + np.diff(edges)/2, masses, "o",
                        color=reference, label="Masa por intervalo")
            elif model is not None:
                x = np.linspace(min(low, model.ppf(.01)), max(high, model.ppf(.99)), 480)
                with np.errstate(all="ignore"):
                    density = model.pdf(x)
                ax.plot(x, np.where(np.isfinite(density), density, np.nan), color=reference, label="Densidad teórica")
        if model is not None or observed is not None:
            ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=NIGHT.text if night else None)
        self.sample_canvas.draw_idle()

    def _sample_plot_message(self, message):
        ax = self.sample_axes
        ax.set_axis_off()
        ax.text(.5, .5, message, transform=ax.transAxes,
                ha="center", va="center", color=NIGHT.muted if is_night(self.parent) else "#506671", fontsize=10)
        self.sample_canvas.draw_idle()

    def session_settings(self):
        name = self.name.get()
        observations = self.empirical.get("1.0", "end-1c") if name == "Empírica" else ""
        return {"name": name, "parameters": {key: var.get() for key, var in self.parameters.items()},
                "observations": observations, "count": self.count.get(), "seed": self.seed.get(),
                "sample": self.sample, "caption": self.sample_caption}

    def restore_session(self, settings):
        self.select_distribution(settings["name"], settings["parameters"], settings["observations"])
        self.count.set(settings["count"])
        self.seed.set(settings["seed"])
        self._show_sample(settings["sample"], settings["caption"])

    def clear_sample(self):
        self._show_sample(None)

    def copy_sample(self):
        if self.sample is not None:
            self.parent.clipboard_clear()
            self.parent.clipboard_append("\n".join(self.sample))
            self.status.set(f"{len(self.sample)} valores copiados al portapapeles.")

    def save_sample(self):
        if self.sample is None:
            return
        path = filedialog.asksaveasfilename(parent=self.parent, defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(("Valor",))
                writer.writerows((value,) for value in self.sample)
            self.status.set(f"{len(self.sample)} valores guardados en {Path(path).name}.")
        except OSError as exc:
            messagebox.showerror("FITEC", str(exc), parent=self.parent)
