"""FITEC's desktop menus, project files, and help."""

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fitec_icons import make_icons
from fitec_queues import validate_queue_settings
from fitec_theme import NIGHT, ZOOM_LEVELS


PAPER = "#edf3f5"
MINT = "#dff3ef"
INK = "#103746"
PROJECT_TYPE = "fitec-session"
PROJECT_FILES = [("Proyecto FITEC", "*.fitec")]


def validate_session(settings, catalog, generator_models):
    if settings["view"] not in ("distributions", "random", "assistant", "queues"):
        raise ValueError("La vista del proyecto no es válida.")
    explorer, generator = settings["explorer"], settings["generator"]
    mode, name = explorer["mode"], explorer["name"]
    if mode == "Empírica" and name == "Empírica":
        explorer_keys = ()
    elif mode in catalog and name in catalog[mode]:
        explorer_keys = catalog[mode][name][1]
    else:
        raise ValueError("La distribución guardada no está disponible.")
    generator_name = generator["name"]
    if generator_name == "Empírica":
        generator_keys = ()
    elif generator_name in generator_models:
        generator_keys = generator_models[generator_name][1]
    else:
        raise ValueError("La distribución del generador no está disponible.")
    for fields, keys, text_keys in (
        (explorer, explorer_keys, ("observations", "first", "second")),
        (generator, generator_keys, ("observations", "count", "seed", "caption")),
    ):
        params = fields["parameters"]
        if not isinstance(params, dict) or set(params) != set(keys):
            raise ValueError("Los parámetros guardados no corresponden a la distribución.")
        if any(not isinstance(value, str) for value in params.values()):
            raise ValueError("El formato de los parámetros guardados no es válido.")
        if any(not isinstance(fields[key], str) for key in text_keys):
            raise ValueError("Hay campos con un formato inválido.")
    if explorer["graph"] not in ("Densidad", "Acumulada"):
        raise ValueError("El tipo de gráfica no es válido.")
    kinds = (
        ("hasta", "menos", "encima", "al_menos", "entre", "exacta")
        if mode in ("Discretas", "Empírica")
        else ("hasta", "menos", "encima", "al_menos", "entre")
    )
    if explorer["probability"] not in kinds:
        raise ValueError("El cálculo de probabilidad no es válido.")
    sample = generator["sample"]
    if sample is not None:
        if not isinstance(sample, list) or not 1 <= len(sample) <= 100_000:
            raise ValueError("El tamaño de la muestra guardada no es válido.")
        if any(not isinstance(value, str) or not math.isfinite(float(value.replace(",", ".")))
               for value in sample):
            raise ValueError("La muestra contiene valores no válidos.")
    return {**settings, "queues": validate_queue_settings(settings.get("queues"))}


class FitecMenus:
    def __init__(self, root, app):
        self.root, self.app = root, app
        self.fit_workspace = None
        self.explorer, self.generator = app.explorer, app.generator
        self.project_path = None
        self.defaults = self._snapshot()
        self.baseline = self.defaults
        self.last_field = None
        self.flash_id = None
        self.show_toolbar = tk.BooleanVar(value=True)
        self.show_status = tk.BooleanVar(value=True)
        self.animate = tk.BooleanVar(value=True)

        self.status = tk.Label(root, text="Proyecto sin título", background=PAPER, foreground=INK,
                               font=("Inter", 9), anchor="w", padx=16, pady=5)
        self.status.pack(side="bottom", fill="x", before=app.content)
        self._build()
        self._shortcuts()
        root.bind("<FocusIn>", self._remember_field, add="+")
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self):
        style = ttk.Style(self.root)
        style.layout("Menu.TMenubutton", [
            ("Menubutton.border", {"sticky": "nswe", "children": [
                ("Menubutton.focus", {"sticky": "nswe", "children": [
                    ("Menubutton.padding", {"sticky": "nswe", "children": [
                        ("Menubutton.label", {"sticky": "w"}),
                    ]}),
                ]}),
            ]}),
        ])
        style.configure("Menu.TMenubutton", font=("Inter", 9, "bold"), padding=(12, 5), width=0,
                        background="#f4f8f9", foreground=INK, bordercolor="#f4f8f9")
        style.map("Menu.TMenubutton", background=[("pressed", MINT), ("active", MINT), ("focus", MINT)],
                  bordercolor=[("pressed", "#a5d7cf"), ("focus", "#a5d7cf")])
        bar = tk.Frame(self.root, background="#f4f8f9", padx=12, pady=3)
        bar.pack(fill="x", before=self.app.content)
        self.bar = bar

        file = self._menu("Archivo", self._update_file)
        self._item(file, "Nuevo proyecto", self.new_project, "add", "Ctrl+N")
        self._item(file, "Nueva muestra", self.clear_sample, "table")
        self._item(file, "Abrir proyecto…", self.open_project, "open", "Ctrl+O")
        file.add_separator()
        self._item(file, "Guardar", self.save, "save", "Ctrl+S")
        self._item(file, "Guardar como…", self.save_as, "save", "Ctrl+Mayús+S")
        export = self._popup(file)
        self._item(export, "Muestra a CSV…", self.generator.save_sample, "table")
        self._item(export, "Gráfica a PNG…", self.export_chart, "chart")
        file.add_cascade(label="Exportar", menu=export)
        file.add_separator()
        self._item(file, "Salir", self.close, accelerator="Ctrl+Q")
        self.file_menu, self.export_menu = file, export

        self.edit_menu = self._menu("Editar", self._update_edit)
        for title, event, shortcut in (
            ("Deshacer", "<<Undo>>", "Ctrl+Z"), ("Rehacer", "<<Redo>>", "Ctrl+Y"),
            ("Cortar", "<<Cut>>", "Ctrl+X"), ("Copiar", "<<Copy>>", "Ctrl+C"),
            ("Pegar", "<<Paste>>", "Ctrl+V"), ("Seleccionar todo", "select", "Ctrl+A"),
        ):
            if title in ("Cortar", "Seleccionar todo"):
                self.edit_menu.add_separator()
            self._item(self.edit_menu, title, lambda event=event: self.edit(event), accelerator=shortcut)

        view = self._menu("Ver", self._update_view)
        self._item(view, "Acercar gráfica", lambda: self.explorer.zoom_graph(.8), "zoom_in")
        self._item(view, "Alejar gráfica", lambda: self.explorer.zoom_graph(1.25), "zoom_out")
        self._item(view, "Tamaño original", self.explorer.zoom_graph, "reset")
        self.graph_menu = self._popup(view)
        for value in ("Densidad", "Acumulada"):
            self.graph_menu.add_radiobutton(label=value, value=value, variable=self.explorer.graph_mode,
                                            command=self.explorer.update_plot)
        view.add_cascade(label="Tipo de gráfica", menu=self.graph_menu)
        view.add_separator()
        view.add_checkbutton(label="Barra de herramientas", variable=self.show_toolbar,
                             command=lambda: self.app.set_toolbar_visible(self.show_toolbar.get()))
        view.add_checkbutton(label="Barra de estado", variable=self.show_status, command=self._toggle_status)
        self.view_menu = view

        analysis = self._menu("Analizar")
        self._item(analysis, "Ajustar distribuciones a datos…", self.open_fit_workspace, "density")
        analysis.add_separator()
        self._item(analysis, "Calcular probabilidad", self.calculate, "density")
        for kind, caption in (("menos", "Menos de (<)"), ("al_menos", "Al menos (≥)")):
            analysis.add_radiobutton(label=caption, variable=self.explorer.prob_kind, value=kind,
                                     command=self.explorer._probability_choice)
        self.show_details = tk.BooleanVar(value=False)
        analysis.add_checkbutton(label="Mostrar momentos y resultado decimal", variable=self.show_details,
                                 command=self._toggle_details)
        self._item(analysis, "Distribución empírica", self.empirical, "cumulative")

        options = self._menu("Opciones")
        options.add_checkbutton(label="Animaciones y efectos", variable=self.animate,
                                command=lambda: self.explorer.set_animation(self.animate.get()))
        self._item(options, "Restablecer parámetros y límites", self.reset_distribution, "reset")

        tools = self._menu("Herramientas", self._update_tools)
        self._item(tools, "Generar números aleatorios", self.generate, "dice")
        self._item(tools, "Usar distribución en generador", self.app._transfer, "arrow")
        self._item(tools, "Copiar muestra", self.generator.copy_sample, "paste")
        self.tools_menu = tools

        window = self._menu("Ventana")
        self._item(window, "Distribuciones", lambda: self.app.select_view("distributions"), "model", "Ctrl+1")
        self._item(window, "Números aleatorios", lambda: self.app.select_view("random"), "dice", "Ctrl+2")
        self._item(window, "Asistente FITEC", lambda: self.app.select_view("assistant"), "assistant", "Ctrl+3")
        self._item(window, "Simulador de filas", lambda: self.app.select_view("queues"), "team", "Ctrl+4")
        window.add_separator()
        self._item(window, "Maximizar / restaurar", self.toggle_maximize)

        help_menu = self._menu("Ayuda")
        self._item(help_menu, "Contenido", self.help, "help", "F1")
        self._item(help_menu, "Explicar mi resultado", self.explain_result, "assistant")
        self._item(help_menu, "Desarrolladores", self.developers, "team")
        help_menu.add_separator()
        self._item(help_menu, "Acuerdo de licencia", self.license)
        self._item(help_menu, "Acerca de FITEC", self.about, "help")
        style.configure("Theme.TButton", padding=(12, 5), font=("Inter", 9, "bold"),
                        background="#f4f8f9", foreground=INK, width=0, borderwidth=0, relief="flat")
        style.map("Theme.TButton", background=[("pressed", MINT), ("active", MINT), ("focus", MINT)])
        self.night_icons = make_icons(self.root, color=NIGHT.focus, names=("sun", "zoom"))
        self.theme_button = ttk.Button(bar, style="Theme.TButton", compound="left",
                                      command=self.app.toggle_theme)
        self.theme_button.pack(side="left", padx=(4, 0))
        self.zoom = tk.IntVar(self.root, value=100)
        self.zoom_button = ttk.Menubutton(bar, text="100 %", compound="left", style="Menu.TMenubutton",
                                         takefocus=True)
        self.zoom_button.pack(side="left", padx=(4, 0))
        self.zoom_menu = self._popup(self.zoom_button)
        self.zoom_button.configure(menu=self.zoom_menu)
        for level in ZOOM_LEVELS:
            self.zoom_menu.add_radiobutton(label="100 % (original)" if level == 100 else f"{level} %",
                                           variable=self.zoom, value=level, command=self._set_zoom)
        self.update_theme_control(False)

    def update_theme_control(self, night):
        self.theme_button.configure(text="Día" if night else "Noche",
                                    image=self.night_icons["sun"] if night else self.app.icons["moon"])
        self.zoom_button.configure(image=self.night_icons["zoom"] if night else self.app.icons["zoom"])

    def _set_zoom(self):
        percentage = self.zoom.get()
        self.app.theme.set_zoom(percentage)
        self.zoom_button.configure(text=f"{percentage} %")

    @staticmethod
    def _popup(parent, prepare=None):
        options = {"tearoff": False, "activebackground": MINT, "activeforeground": INK,
                   "foreground": INK, "background": "white", "font": ("Inter", 10)}
        if prepare is not None:
            options["postcommand"] = prepare
        return tk.Menu(parent, **options)

    def _menu(self, label, prepare=None):
        button = ttk.Menubutton(self.bar, text=label, style="Menu.TMenubutton", takefocus=True)
        button.pack(side="left", padx=(0, 2))
        menu = self._popup(button, prepare)
        button.configure(menu=menu)
        return menu

    def _item(self, menu, label, action, icon=None, accelerator=""):
        options = {"label": label, "command": action, "accelerator": accelerator}
        if icon is not None:
            options.update(image=self.app.icons[icon], compound="left")
        menu.add_command(**options)

    def _shortcuts(self):
        for sequence, action in (
            ("<Control-n>", self.new_project), ("<Control-o>", self.open_project),
            ("<Control-s>", self.save), ("<Control-Shift-S>", self.save_as),
            ("<Control-q>", self.close), ("<F1>", self.help),
            ("<Control-Key-1>", lambda: self.app.select_view("distributions")),
            ("<Control-Key-2>", lambda: self.app.select_view("random")),
            ("<Control-Key-3>", lambda: self.app.select_view("assistant")),
            ("<Control-Key-4>", lambda: self.app.select_view("queues")),
        ):
            self.root.bind(sequence, lambda _event, action=action: self._shortcut(action))
        self.root.bind("<Control-a>", self._select_all)

    def _select_all(self, _event):
        if self._is_field(self.root.focus_get()):
            self.edit("select")
            return "break"

    @staticmethod
    def _shortcut(action):
        action()
        return "break"

    @staticmethod
    def _state(menu, entry, enabled):
        menu.entryconfigure(entry, state="normal" if enabled else "disabled")

    def _update_file(self):
        sample = self.generator.sample is not None
        chart = self.explorer.result.cget("text") != "—"
        self._state(self.file_menu, "Nueva muestra", sample)
        self._state(self.file_menu, "Exportar", sample or chart)
        self._state(self.export_menu, "Muestra a CSV…", sample)
        self._state(self.export_menu, "Gráfica a PNG…", chart)

    def _update_view(self):
        graph = self.app.active_view.get() == "distributions"
        for entry in ("Acercar gráfica", "Alejar gráfica", "Tamaño original", "Tipo de gráfica"):
            self._state(self.view_menu, entry, graph)
        density = "Masa" if self.explorer.mode.get() == "Discretas" else "Densidad"
        self.graph_menu.entryconfigure(0, label=density,
                                       state="disabled" if self.explorer.mode.get() == "Empírica" else "normal")

    def _update_tools(self):
        self._state(self.tools_menu, "Copiar muestra", self.generator.sample is not None)

    @staticmethod
    def _is_field(widget):
        return isinstance(widget, (tk.Entry, ttk.Entry, tk.Text)) and not isinstance(widget, ttk.Combobox)

    def _remember_field(self, event):
        if self._is_field(event.widget):
            self.last_field = event.widget
        elif not isinstance(event.widget, ttk.Menubutton):
            self.last_field = None

    def _edit_field(self):
        field = self.root.focus_get()
        if not self._is_field(field):
            field = self.last_field
        if field is not None and field.winfo_exists() and field.winfo_ismapped():
            return field
        return None

    def _update_edit(self):
        field = self._edit_field()
        selected = False
        if field is not None:
            selected = bool(field.tag_ranges("sel")) if isinstance(field, tk.Text) else field.selection_present()
        for label in ("Deshacer", "Rehacer"):
            self._state(self.edit_menu, label, isinstance(field, tk.Text))
        for label in ("Cortar", "Copiar"):
            self._state(self.edit_menu, label, selected)
        for label in ("Pegar", "Seleccionar todo"):
            self._state(self.edit_menu, label, field is not None)

    def edit(self, action):
        field = self._edit_field()
        if field is None:
            return
        field.focus_set()
        if action == "select":
            if isinstance(field, tk.Text):
                field.tag_add("sel", "1.0", "end-1c")
            else:
                field.select_range(0, "end")
        elif action in ("<<Undo>>", "<<Redo>>") and isinstance(field, tk.Text):
            try:
                (field.edit_undo if action == "<<Undo>>" else field.edit_redo)()
            except tk.TclError:
                self.notify("No hay más cambios de texto para esa acción.")
        else:
            field.event_generate(action)

    def notify(self, text):
        self.status.configure(text=text)
        if self.flash_id is not None:
            self.root.after_cancel(self.flash_id)
        self.app.theme.update_colors(self.status, background=MINT if self.animate.get() else PAPER)
        self.flash_id = self.root.after(190, self._clear_flash) if self.animate.get() else None

    def _clear_flash(self):
        self.flash_id = None
        self.app.theme.update_colors(self.status, background=PAPER)

    def _toggle_status(self):
        if self.show_status.get():
            self.status.pack(side="bottom", fill="x", before=self.app.content)
        else:
            self.status.pack_forget()

    def _snapshot(self):
        return self.app.session_settings()

    def _set_project(self, path):
        self.project_path = path
        self.baseline = self._snapshot()
        self.root.title(f"FITEC — {path.stem}" if path else "FITEC")

    def _confirm_changes(self):
        if self._snapshot() == self.baseline:
            return True
        answer = messagebox.askyesnocancel("FITEC", "¿Guardar los cambios del proyecto actual?", parent=self.root)
        return answer is not None and (not answer or self.save())

    def new_project(self):
        if self._confirm_changes():
            self.app.restore_session(self.defaults)
            self._set_project(None)
            self.notify("Proyecto nuevo.")

    def clear_sample(self):
        if self.generator.sample is not None and not messagebox.askyesno(
            "Nueva muestra", "¿Quitar la muestra actual del generador?", parent=self.root
        ):
            return
        self.generator.clear_sample()
        self.app.select_view("random")
        self.notify("Generador listo para una nueva muestra.")

    def open_project(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=PROJECT_FILES)
        if not path:
            return
        source = Path(path)
        try:
            document = json.loads(source.read_text(encoding="utf-8-sig"))
            if document.get("format") != PROJECT_TYPE or document.get("version") != 1:
                raise ValueError("Este archivo no es un proyecto de Distribuciones y Números aleatorios.")
            settings = validate_session(document["settings"], self.explorer.catalog, self.generator.models)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            messagebox.showerror("Abrir proyecto", str(exc), parent=self.root)
            return
        if self._confirm_changes():
            self.app.restore_session(settings)
            self._set_project(source)
            self.notify(f"Proyecto abierto: {source.name}")

    def save(self):
        return self._write(self.project_path) if self.project_path else self.save_as()

    def save_as(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".fitec", filetypes=PROJECT_FILES)
        return self._write(Path(path)) if path else False

    def _write(self, path):
        temporary = None
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                    prefix=".fitec-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({"format": PROJECT_TYPE, "version": 1, "settings": self._snapshot()},
                          handle, ensure_ascii=False, allow_nan=False)
            temporary.replace(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Guardar proyecto", str(exc), parent=self.root)
            return False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._set_project(path)
        self.notify(f"Proyecto guardado: {path.name}")
        return True

    def export_chart(self):
        if self.explorer.result.cget("text") == "—":
            self.notify("Corrige los parámetros o límites antes de exportar la gráfica.")
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".png",
                                            filetypes=[("Imagen PNG", "*.png")])
        if path:
            try:
                self.explorer.figure.savefig(path, dpi=180)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Exportar gráfica", str(exc), parent=self.root)
                return
            self.notify(f"Gráfica guardada: {Path(path).name}")

    def _toggle_details(self):
        if self.show_details.get():
            self.explorer.stats_frame.pack(side="bottom", fill="x", pady=(10, 0))
            self.explorer.decimal_result.pack(after=self.explorer.result)
        else:
            self.explorer.stats_frame.pack_forget()
            self.explorer.decimal_result.pack_forget()

    def calculate(self):
        self.app.select_view("distributions")
        self.explorer.update_plot()
        self.notify("La probabilidad aparece debajo de la gráfica.")

    def empirical(self):
        self.app.select_view("distributions")
        self.explorer.mode.set("Empírica")
        self.explorer._distribution_changed()
        self.explorer.data_text.focus_set()
        self.notify("Pega tus observaciones para ver la distribución empírica.")

    def reset_distribution(self):
        self.app.select_view("distributions")
        self.explorer.reset_values()
        self.notify("Parámetros y límites restablecidos.")

    def generate(self):
        self.app.select_view("random")
        self.generator.generate()

    def toggle_maximize(self):
        self.root.state("normal" if self.root.state() == "zoomed" else "zoomed")

    def help(self):
        self._help_dialog("Contenido",
                          "Distribuciones: selecciona un modelo y sus parámetros. Escribe o arrastra los límites "
                          "para ver la probabilidad.\n\n"
                          "Números aleatorios: indica la cantidad y la semilla, pulsa Generar y copia o guarda "
                          "la muestra en CSV.\n\n"
                          "Ctrl+1 abre Distribuciones y Ctrl+2 abre Números aleatorios. Guarda el proyecto "
                          "con Ctrl+S para recuperar tus parámetros y la muestra.")

    def developers(self):
        self._help_dialog("Desarrolladores", "Equipo FITEC", "team")

    def explain_result(self):
        self.app.select_view("assistant")
        self.app.assistant.show_result()

    def about(self):
        self._help_dialog("Acerca de FITEC", "FITEC · Simulación de Sistemas\n\n"
                          "Explora distribuciones de probabilidad y genera muestras reproducibles.")

    def license(self):
        self._help_dialog("Acuerdo de licencia",
                          "FITEC es gratuito y es software libre.\n\n"
                          "La licencia MIT permite usar, estudiar, modificar y distribuir el programa, "
                          "incluso con fines comerciales. Al distribuir copias o partes sustanciales, "
                          "incluye el aviso de derechos de autor y la licencia MIT.\n\n"
                          "El programa se ofrece sin garantía. El texto completo se incluye "
                          "en el archivo LICENSE del proyecto.")

    def _help_dialog(self, title, content, icon="help"):
        prior_focus = self.root.focus_get()
        window = tk.Toplevel(self.root)
        window.withdraw()
        window.title(f"FITEC · {title}")
        window.transient(self.root)
        window.resizable(False, False)
        panel = ttk.Frame(window, style="Panel.TFrame", padding=(22, 18))
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text=title, image=self.app.icons[icon], compound="left",
                  style="Section.TLabel").pack(anchor="w")
        ttk.Label(panel, text=content, wraplength=510, justify="left").pack(anchor="w", pady=(12, 16))

        def dismiss(_event=None):
            window.destroy()
            if prior_focus is not None and prior_focus.winfo_exists():
                prior_focus.focus_set()

        button = ttk.Button(panel, text="Cerrar", command=dismiss)
        button.pack(anchor="e")
        window.bind("<Escape>", dismiss)
        window.bind("<Return>", dismiss)
        window.protocol("WM_DELETE_WINDOW", dismiss)
        window.update_idletasks()
        x = max(0, self.root.winfo_rootx() + (self.root.winfo_width() - window.winfo_reqwidth()) // 2)
        y = max(0, self.root.winfo_rooty() + (self.root.winfo_height() - window.winfo_reqheight()) // 2)
        window.geometry(f"+{x}+{y}")
        window.deiconify()
        window.grab_set()
        button.focus_set()

    def open_fit_workspace(self):
        from fitec_workspace import FitecWorkspace

        if self.fit_workspace and self.fit_workspace.root.winfo_exists():
            self.fit_workspace.root.deiconify()
            self.fit_workspace.root.lift()
            return
        window = tk.Toplevel(self.root)
        with self.app.theme.building_window():
            self.fit_workspace = FitecWorkspace(window, type(self.explorer),
                                                on_use_simulation=self._open_simulation_with_fit)

    def _open_simulation_with_fit(self, fit, summary):
        self.app.queues.load_fit(fit, summary)
        self.fit_workspace.root.withdraw()
        self.app.select_view("queues")
        self.root.deiconify()
        self.root.lift()
        self.app.queues.start_button.focus_set()

    def close(self):
        if self.fit_workspace and self.fit_workspace.root.winfo_exists():
            self.fit_workspace.close()
            if self.fit_workspace.root.winfo_exists():
                return
        if self._confirm_changes():
            self.root.destroy()
