"""Desktop workspace for data, distribution fitting, and simulation inputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass, field
import io
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from fitec_icons import make_icons
from fitec_analysis import (
    BOOTSTRAP_REPETITIONS, CONTINUOUS_MODELS, DISCRETE_MODELS, MODELS,
    FitResult, fit_models,
)


PAGE_SIZE = 200
TEAL = "#087d77"
INK = "#103746"
PAPER = "#edf3f5"


def number(value):
    result = float(value.strip().replace(",", "."))
    if not math.isfinite(result):
        raise ValueError("El dato debe ser un número finito.")
    return result


def decimal(value, precision=5):
    return f"{value:.{precision}g}".replace(".", ",")


def parse_grid(text, *, csv_file=False):
    """Read clipboard TSV or CSV without splitting decimal commas in pasted columns."""
    sample = text.lstrip("\ufeff")
    if "\t" in sample:
        delimiter = "\t"
    elif ";" in sample:
        delimiter = ";"
    elif csv_file:
        try:
            delimiter = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = "\t"
    else:
        delimiter = "\t"
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(sample), delimiter=delimiter)
            if any(cell.strip() for cell in row)]


def looks_numeric(cell):
    try:
        number(cell)
        return True
    except ValueError:
        return False


def atomic_json(path, document):
    path = Path(path)
    temp = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".fitec-", suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            json.dump(document, handle, ensure_ascii=False, allow_nan=False)
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


@dataclass
class DataTable:
    name: str
    headers: list[str]
    rows: list[list[str]] = field(default_factory=list)
    fits: dict[int, list[FitResult]] = field(default_factory=dict)
    revision: int = 0

    def values(self, column):
        values = []
        for index, row in enumerate(self.rows, 1):
            cell = row[column] if column < len(row) else ""
            if cell.strip():
                try:
                    values.append(number(cell))
                except ValueError as exc:
                    raise ValueError(f"Fila {index}: «{cell}» no es un número válido.") from exc
        if not values:
            raise ValueError("La columna seleccionada está vacía.")
        return np.asarray(values)

    def as_document(self):
        return {"name": self.name, "headers": self.headers, "rows": self.rows,
                "fits": {str(index): [asdict(result) for result in fits]
                         for index, fits in self.fits.items()}}

    @classmethod
    def from_document(cls, document):
        headers = document["headers"]
        rows = document["rows"]
        if not isinstance(document["name"], str) or not headers or not isinstance(headers, list):
            raise ValueError("Hay una tabla con datos incompletos.")
        if any(not isinstance(header, str) for header in headers) or not isinstance(rows, list):
            raise ValueError("Hay una tabla con un formato inválido.")
        if any(not isinstance(row, list) or len(row) > len(headers)
               or any(not isinstance(cell, str) for cell in row) for row in rows):
            raise ValueError("Hay filas con un formato inválido.")
        table = cls(document["name"], headers, rows)
        for key, raw_fits in document.get("fits", {}).items():
            column = int(key)
            if not 0 <= column < len(headers):
                raise ValueError("Un análisis hace referencia a una columna inexistente.")
            fits = []
            for raw_fit in raw_fits:
                model = MODELS.get(raw_fit["model"])
                if model is None:
                    raise ValueError("El proyecto contiene un modelo desconocido.")
                parameters = tuple(float(value) for value in raw_fit["parameters"])
                expected = model.distribution.numargs + (0 if model.discrete else 2)
                if len(parameters) != expected or not all(map(math.isfinite, parameters)):
                    raise ValueError("El proyecto contiene parámetros de ajuste inválidos.")
                fits.append(FitResult(**{**raw_fit, "parameters": parameters}))
            table.fits[column] = fits
        return table


class FitecWorkspace:
    def __init__(self, root, explorer_type):
        self.root = root
        width = min(1200, root.winfo_screenwidth() - 48)
        height = min(670, root.winfo_screenheight() - 108)
        root.geometry(f"{width}x{height}")
        root.minsize(min(990, width), min(590, height))
        root.configure(background=PAPER)
        self.tables = [DataTable("Tabla 1", ["A"])]
        self.active = 0
        self.page = 0
        self.editing = None
        self.project_path = None
        self.dirty = False
        self.job = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fitec-fit")
        self.result_fits = []
        self.loaded_model = None
        self.trials = None
        self.fit_candidates = None
        self.plot_ready = False
        self.status = tk.StringVar(value="Pega observaciones o abre un CSV para empezar.")
        self.column = tk.StringVar()
        self.table_choice = tk.StringVar()
        self.data_kind = tk.StringVar(value="Automático")
        self.chart_kind = tk.StringVar(value="Histograma")
        self.fit_config_text = tk.StringVar(value="Ajuste automático")
        self.icons = make_icons(root)
        self._build(explorer_type)
        self._refresh_data()
        self._refresh_results()
        self._update_title()
        root.after_idle(self.grid.focus_set)
        root.protocol("WM_DELETE_WINDOW", self.close)

    @property
    def table(self):
        return self.tables[self.active]

    def _build(self, explorer_type):
        self.notebook = ttk.Notebook(self.root, padding=(8, 8, 8, 0))
        self.notebook.pack(fill="both", expand=True)
        self.data_tab = ttk.Frame(self.notebook, padding=12)
        self.results_tab = ttk.Frame(self.notebook, padding=12)
        self.model_tab = ttk.Frame(self.notebook)
        for pane, label, icon in ((self.data_tab, "Datos", "table"),
                                  (self.results_tab, "Ajustes", "chart"),
                                  (self.model_tab, "Modelo elegido", "model")):
            self.notebook.add(pane, text=label, image=self.icons[icon], compound="left")
        self.explorer = self._build_model(explorer_type)
        self.notebook.tab(self.model_tab, state="disabled")
        ttk.Style(self.root).configure("TNotebook.Tab", padding=(18, 8), font=("Inter", 10, "bold"))
        ttk.Style(self.root).map("TNotebook.Tab", foreground=[("selected", TEAL)],
                                 background=[("selected", "white")])
        ttk.Style(self.root).configure("Primary.TButton", foreground="white", background=TEAL)
        ttk.Style(self.root).map("Primary.TButton", background=[("active", "#086863"),
                                                                 ("disabled", "#748a8c")])
        header = tk.Frame(self.root, background=INK)
        header.pack(fill="x", before=self.notebook)
        tk.Label(header, text="FITEC", background=INK, foreground="white",
                 font=("Inter", 25, "bold")).pack(side="left", padx=(24, 18), pady=12)
        tk.Label(header, text="Ajuste de distribuciones a datos", background=INK, foreground="#d7ebe9",
                 font=("Inter", 13)).pack(side="left")
        ttk.Label(self.root, textvariable=self.status, background=PAPER,
                  padding=(20, 8)).pack(side="bottom", fill="x", before=self.notebook)
        self._build_menu()
        self._build_data()
        self._build_results()
        self.notebook.bind("<<NotebookTabChanged>>", self._tab_changed)
        for key, action in (("<Control-n>", self.new_project), ("<Control-o>", self.open_project),
                            ("<Control-s>", self.save_project), ("<F9>", self.fit)):
            self.root.bind(key, lambda _event, action=action: self._shortcut(action))

    @staticmethod
    def _shortcut(action):
        action()
        return "break"

    def _button(self, parent, text, icon, command, **options):
        return ttk.Button(parent, text=text, image=self.icons[icon], compound="left",
                          command=command, **options)

    def _build_model(self, explorer_type):
        header = ttk.Frame(self.model_tab, style="Panel.TFrame", padding=(14, 10))
        header.pack(fill="x", padx=14, pady=(12, 0))
        self._button(header, "Volver a ajustes", "back", lambda: self.notebook.select(self.results_tab)).pack(
            side="right", padx=(6, 0))
        self._button(header, "Generar muestra", "model", self.generator).pack(side="right")
        self.model_title = ttk.Label(header, style="Section.TLabel")
        self.model_title.pack(anchor="w")
        self.model_detail = ttk.Label(header, style="Help.TLabel", wraplength=550)
        self.model_detail.pack(anchor="w", pady=(3, 0))
        host = ttk.Frame(self.model_tab)
        host.pack(fill="both", expand=True)
        return explorer_type(host, embedded=True, fixed_model=True)

    def _build_menu(self):
        bar = tk.Menu(self.root)
        groups = {
            "Archivo": [("Nuevo proyecto", self.new_project, "Ctrl+N"),
                        ("Abrir proyecto…", self.open_project, "Ctrl+O"),
                        ("Guardar proyecto", self.save_project, "Ctrl+S"),
                        ("Guardar proyecto como…", self.save_as, ""),
                        ("Importar CSV…", self.import_csv, ""),
                        ("Exportar datos a CSV…", self.export_data, ""),
                        ("Exportar ajustes a CSV…", self.export_fits, ""),
                        ("Guardar gráfica de resultados…", self.export_plot, "")],
            "Analizar": [("Preparar datos…", self.prepare_data, ""),
                         ("Configurar ajuste…", self.configure_fit, ""),
                         ("Ajustar distribuciones", self.fit, "F9")],
        }
        for title, entries in groups.items():
            menu = tk.Menu(bar, tearoff=False)
            for label, action, shortcut in entries:
                menu.add_command(label=label, command=action, accelerator=shortcut)
            bar.add_cascade(label=title, menu=menu)
        self.root.configure(menu=bar)

    def _build_data(self):
        actions = ttk.Frame(self.data_tab)
        actions.pack(fill="x", pady=(0, 10))
        ttk.Label(actions, text="Tabla", background=PAPER).pack(side="left", padx=(0, 5))
        self.table_picker = ttk.Combobox(actions, textvariable=self.table_choice, state="readonly", width=16)
        self.table_picker.pack(side="left", padx=(0, 8))
        self.table_picker.bind("<<ComboboxSelected>>", self._choose_table)
        for label, icon, action in (("Nueva tabla", "table", self.new_table),
                                    ("Importar CSV", "open", self.import_csv),
                                    ("Pegar datos", "paste", self.paste_grid),
                                    ("Añadir fila", "add", self.add_row),
                                    ("Eliminar fila", "remove", self.delete_rows)):
            self._button(actions, label, icon, action).pack(side="left", padx=(0, 6))
        workspace = ttk.Frame(self.data_tab)
        workspace.pack(fill="both", expand=True)
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)
        sheet = ttk.Frame(workspace, style="Panel.TFrame")
        sheet.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.grid = self._tree(sheet, show="headings")
        self.grid.bind("<Double-1>", self._edit_cell)
        self.grid.bind("<Control-v>", lambda _event: self._shortcut(self.paste_grid))
        self.grid.bind("<Control-c>", lambda _event: self._shortcut(self.copy_rows))
        self.grid.bind("<Delete>", lambda _event: self._shortcut(self.delete_rows))
        self.grid.bind("<F2>", lambda _event: self._shortcut(self.rename_column))
        sheet.grid_rowconfigure(0, weight=1)
        pager = ttk.Frame(sheet, style="Panel.TFrame", padding=6)
        pager.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.page_label = ttk.Label(pager)
        self.page_label.pack(side="left", padx=(2, 9))
        ttk.Button(pager, text="Anterior", command=lambda: self._turn_page(-1)).pack(side="left", padx=3)
        ttk.Button(pager, text="Siguiente", command=lambda: self._turn_page(1)).pack(side="left", padx=3)
        ttk.Button(pager, text="Renombrar columna", command=self.rename_column).pack(side="right")

        sidebar = ttk.Frame(workspace, style="Panel.TFrame", padding=14)
        sidebar.grid(row=0, column=1, sticky="ns")
        ttk.Label(sidebar, text="Analizar columna", style="Section.TLabel").pack(anchor="w")
        ttk.Label(sidebar, text="Columna").pack(anchor="w", pady=(14, 4))
        self.column_picker = ttk.Combobox(sidebar, textvariable=self.column, state="readonly", width=20)
        self.column_picker.pack(fill="x")
        self.column_picker.bind("<<ComboboxSelected>>", self._column_changed)
        ttk.Label(sidebar, text="Tipo de datos").pack(anchor="w", pady=(12, 4))
        data_kind = ttk.Combobox(sidebar, textvariable=self.data_kind, state="readonly",
                                 values=("Automático", "Continuos", "Recuentos"), width=20)
        data_kind.pack(fill="x")
        data_kind.bind("<<ComboboxSelected>>", lambda _event: self._modified())
        self._button(sidebar, "Preparar datos", "prepare", self.prepare_data).pack(fill="x", pady=(12, 5))
        self._button(sidebar, "Configurar ajuste", "settings", self.configure_fit).pack(fill="x")
        ttk.Label(sidebar, textvariable=self.fit_config_text, style="Help.TLabel").pack(anchor="w", pady=(7, 5))
        self.fit_button = self._button(sidebar, "Ajustar · F9", "fit", self.fit, style="Primary.TButton")
        self.fit_button.pack(fill="x")
        ttk.Label(sidebar, text="Doble clic para editar una celda. Puedes pegar varias columnas desde Excel.",
                  style="Help.TLabel", wraplength=190).pack(anchor="w", pady=(15, 0))

    @staticmethod
    def _tree(parent, **kwargs):
        tree = ttk.Treeview(parent, **kwargs)
        vertical = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        return tree

    def _build_results(self):
        top = ttk.Frame(self.results_tab)
        top.pack(fill="x", pady=(0, 10))
        self.result_heading = ttk.Label(top, font=("Inter", 16, "bold"), background=PAPER)
        self.result_heading.pack(side="left")
        self.open_model_button = self._button(top, "Usar modelo seleccionado", "arrow", self.open_selected_fit)
        self.open_model_button.pack(side="right")
        content = ttk.Frame(self.results_tab)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)
        left = ttk.Frame(content, style="Panel.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ttk.Label(left, text="Columna de ajuste", style="Section.TLabel").pack(anchor="w")
        self.summary = ttk.Label(left, justify="left", wraplength=360)
        self.summary.pack(anchor="w", pady=(8, 14))
        ttk.Label(left, text="Modelos ajustados", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        wrapper = ttk.Frame(left, style="Panel.TFrame")
        wrapper.pack(fill="both", expand=True)
        columns = ("model", "aic", "bic", "ks", "p")
        self.fit_grid = self._tree(wrapper, columns=columns, show="headings", selectmode="browse", height=7)
        for column, title, width in zip(columns, ("Distribución", "AIC", "BIC", "KS", "Valor p"),
                                        (137, 70, 70, 55, 60)):
            self.fit_grid.heading(column, text=title)
            self.fit_grid.column(column, width=width, minwidth=45, anchor="w" if column == "model" else "e")
        self.fit_grid.bind("<<TreeviewSelect>>", self._select_result)
        self.fit_grid.bind("<Double-1>", lambda _event: self.open_selected_fit())
        self.fit_details = ttk.Label(left, justify="left", wraplength=370, style="Help.TLabel")
        self.fit_details.pack(fill="x", pady=(7, 0))
        chart = ttk.Frame(content, style="Panel.TFrame", padding=12)
        chart.grid(row=0, column=1, sticky="nsew")
        controls = ttk.Frame(chart, style="Panel.TFrame")
        controls.pack(fill="x")
        ttk.Label(controls, text="Comparar", style="Section.TLabel").pack(side="left")
        for label in ("Histograma", "Acumulada"):
            ttk.Radiobutton(controls, text=label, value=label, variable=self.chart_kind,
                            style="Toolbutton", command=self._draw_results).pack(side="right")
        self.result_figure = Figure(figsize=(5.0, 3.8), dpi=100)
        self.result_axes = self.result_figure.add_subplot(111)
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, master=chart)
        self.result_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

    def _update_title(self):
        name = self.project_path.stem if self.project_path else "Sin título"
        self.root.title(f"FITEC — {name}" + (" *" if self.dirty else ""))

    def _modified(self, *, data_changed=False):
        self.dirty = True
        if data_changed:
            self.table.revision += 1
            self.table.fits.clear()
            self._clear_model()
        self._update_title()

    def _current_column(self):
        return max(0, self.column_picker.current())

    def _refresh_data(self, column=None):
        self._finish_edit()
        old_column = self._current_column() if column is None else column
        self.table_picker.configure(values=[table.name for table in self.tables])
        self.table_picker.current(self.active)
        self.column_picker.configure(values=[f"{index + 1} · {name}" for index, name in enumerate(self.table.headers)])
        self.column_picker.current(min(old_column, len(self.table.headers) - 1))
        columns = ("row", *(f"c{index}" for index in range(len(self.table.headers))))
        self.grid.configure(columns=columns)
        self.grid.heading("row", text="#")
        self.grid.column("row", anchor="e", width=58, stretch=False)
        for key, header in zip(columns[1:], self.table.headers):
            self.grid.heading(key, text=header)
            self.grid.column(key, width=135, minwidth=72)
        self.page = min(self.page, max(0, (len(self.table.rows) - 1) // PAGE_SIZE))
        self.grid.delete(*self.grid.get_children())
        start = self.page * PAGE_SIZE
        visible = self.table.rows[start:start + PAGE_SIZE]
        for index, row in enumerate(visible, start):
            self.grid.insert("", "end", iid=str(index), values=(index + 1, *row))
        if len(visible) < PAGE_SIZE:
            index = len(self.table.rows)
            self.grid.insert("", "end", iid=str(index), values=(index + 1,))
        count = len(self.table.rows)
        self.page_label.configure(text=f"{start + 1}–{start + len(visible)} de {count} filas" if count else "Sin datos")

    def _choose_table(self, _event=None):
        self._finish_edit()
        self.active = self.table_picker.current()
        self._clear_model()
        self.page = 0
        self._refresh_data(0)
        self._refresh_results()

    def _column_changed(self, _event=None):
        self._finish_edit()
        self._clear_model()
        self._refresh_results()

    def _turn_page(self, direction):
        self._finish_edit()
        self.page = max(0, self.page + direction)
        self._refresh_data()

    def _new_table(self, name, headers, rows):
        self._finish_edit()
        self._clear_model()
        existing = {table.name for table in self.tables}
        original = name
        suffix = 2
        while name in existing:
            name = f"{original} ({suffix})"
            suffix += 1
        self.tables.append(DataTable(name, headers, rows))
        self.active = len(self.tables) - 1
        self.page = 0
        self._modified()
        self._refresh_data(0)
        self._refresh_results()
        self.notebook.select(self.data_tab)

    def new_table(self):
        self._new_table(f"Tabla {len(self.tables) + 1}", ["A"], [])

    def _edit_cell(self, event):
        item = self.grid.identify_row(event.y)
        displayed_column = self.grid.identify_column(event.x)
        if not item or not displayed_column or displayed_column == "#1":
            return
        self._finish_edit()
        column = int(displayed_column[1:]) - 2
        self.column_picker.current(column)
        self._start_edit(int(item), column)

    def _start_edit(self, row, column):
        position = self.grid.bbox(str(row), f"#{column + 2}")
        if not position:
            return
        current = self.table.rows[row] if row < len(self.table.rows) else []
        editor = ttk.Entry(self.grid)
        editor.insert(0, current[column] if column < len(current) else "")
        editor.place(x=position[0], y=position[1], width=position[2], height=position[3])
        self.editing = editor, row, column
        editor.focus_set()
        editor.select_range(0, "end")
        editor.bind("<Return>", lambda _event: self._shortcut(self._finish_edit))
        editor.bind("<Escape>", lambda _event: self._shortcut(lambda: self._finish_edit(commit=False)))
        editor.bind("<FocusOut>", lambda _event: self._finish_edit())

    def _finish_edit(self, *, commit=True):
        if self.editing is None:
            return
        editor, row, column = self.editing
        self.editing = None
        value = editor.get().strip()
        editor.destroy()
        if commit:
            old = self.table.rows[row] if row < len(self.table.rows) else []
            if value != (old[column] if column < len(old) else ""):
                self._write_cells(row, column, [[value]])

    def _write_cells(self, row, column, block):
        width = column + max(map(len, block))
        while len(self.table.headers) < width:
            self.table.headers.append(f"Columna {len(self.table.headers) + 1}")
        while len(self.table.rows) < row + len(block):
            self.table.rows.append([])
        for index, cells in enumerate(block, row):
            target = self.table.rows[index]
            target.extend([""] * max(0, column + len(cells) - len(target)))
            target[column:column + len(cells)] = cells
        self._modified(data_changed=True)
        self._refresh_data(column)
        self._refresh_results()

    def paste_grid(self):
        try:
            rows = parse_grid(self.root.clipboard_get())
        except (tk.TclError, csv.Error):
            self.status.set("No hay una tabla de texto en el portapapeles.")
            return
        if rows:
            self._finish_edit()
            selected = self.grid.selection()
            row = int(selected[0]) if selected else len(self.table.rows)
            column = self._current_column()
            if (not self.table.rows and column == 0 and len(rows) > 1
                    and not any(map(looks_numeric, rows[0]))
                    and any(map(looks_numeric, rows[1]))
                    and messagebox.askyesno("Encabezados", "¿Usar la primera fila como nombres de columna?",
                                            parent=self.root)):
                self.table.headers = [cell or f"Columna {index + 1}"
                                      for index, cell in enumerate(rows.pop(0))]
            self._write_cells(row, column, rows)
            self.status.set(f"{len(rows)} filas pegadas en {self.table.name}.")

    def copy_rows(self):
        rows = [self.table.rows[int(key)] for key in self.grid.selection() if int(key) < len(self.table.rows)]
        if rows:
            self._copy_text("\n".join("\t".join(row) for row in rows))

    def _copy_text(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def add_row(self):
        self._finish_edit()
        index = len(self.table.rows)
        self.table.rows.append([])
        self.page = index // PAGE_SIZE
        self._modified(data_changed=True)
        self._refresh_data()
        self._refresh_results()
        self.grid.selection_set(str(index))
        self.grid.see(str(index))
        self.root.update_idletasks()
        self._start_edit(index, self._current_column())

    def delete_rows(self):
        self._finish_edit()
        indexes = sorted((int(key) for key in self.grid.selection() if int(key) < len(self.table.rows)), reverse=True)
        if indexes:
            for index in indexes:
                del self.table.rows[index]
            self._modified(data_changed=True)
            self._refresh_data()
            self._refresh_results()

    def rename_column(self):
        self._finish_edit()
        column = self._current_column()
        name = simpledialog.askstring("Renombrar columna", "Nombre de la columna:",
                                      initialvalue=self.table.headers[column], parent=self.root)
        if name and name.strip():
            self.table.headers[column] = name.strip()
            self._modified()
            self._refresh_data(column)
            self._refresh_results()

    def import_csv(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("CSV o TSV", "*.csv *.tsv"), ("Todos", "*.*")])
        if not path:
            return
        try:
            source = Path(path)
            try:
                text = source.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                text = source.read_text(encoding="cp1252")
            rows = parse_grid(text, csv_file=True)
            if not rows:
                raise ValueError("El archivo está vacío.")
            headers = []
            if len(rows) > 1 and not any(map(looks_numeric, rows[0])):
                if messagebox.askyesno("Encabezados", "¿Usar la primera fila como nombres de columna?", parent=self.root):
                    headers = rows.pop(0)
            width = max(len(headers), max(map(len, rows)))
            headers.extend(f"Columna {index + 1}" for index in range(len(headers), width))
            self._new_table(source.stem, headers, rows)
            self.status.set(f"{len(rows)} filas importadas.")
        except (OSError, ValueError, csv.Error) as exc:
            self._error(exc)

    def _tab_changed(self, _event=None):
        if self.notebook.select() == str(self.results_tab):
            self._finish_edit()
            self._refresh_results()
        elif self.notebook.select() == str(self.data_tab):
            self.grid.focus_set()

    def configure_fit(self):
        FitSettingsDialog(self)

    def _set_fit_configuration(self, candidates, data_kind):
        self.fit_candidates = candidates
        self.data_kind.set(data_kind)
        self.fit_config_text.set("Ajuste automático" if candidates is None
                                 else f"Selección manual · {len(candidates)}")

    def prepare_data(self):
        self._finish_edit()
        column = self._current_column()
        try:
            self.table.values(column)
        except ValueError as exc:
            self._error(exc)
            return
        PrepareDataDialog(self, self.table, column)

    def _append_prepared_column(self, table, name, values):
        column = len(table.headers)
        table.headers.append(name)
        for row, value in zip(table.rows, values, strict=True):
            row.extend([""] * (column - len(row)))
            row.append(value)
        self._clear_model()
        self._modified()
        self._refresh_data(column)
        self._refresh_results()
        self.notebook.select(self.data_tab)
        self.status.set(f"Columna creada: {name}. Puedes ajustarla con F9.")

    def fit(self):
        if self.job is not None:
            self.status.set("El ajuste ya está en curso.")
            return
        self._finish_edit()
        table, column = self.table, self._current_column()
        try:
            values = table.values(column)
        except ValueError as exc:
            self._error(exc)
            return
        kind = self.data_kind.get()
        counts = bool(np.all(values >= 0) and np.all(values == np.floor(values)))
        discrete = kind == "Recuentos" or (kind == "Automático" and counts)
        future = self.worker.submit(fit_models, values, discrete, model_names=self.fit_candidates, trials=self.trials)
        self.job = future, table, column, table.revision
        self.fit_button.state(["disabled"])
        self.status.set(f"Ajustando {table.headers[column]} como {'recuentos' if discrete else 'datos continuos'}…")
        self.root.after(125, self._check_fit)

    def _check_fit(self):
        future, table, column, revision = self.job
        if not future.done():
            self.root.after(125, self._check_fit)
            return
        self.job = None
        self.fit_button.state(["!disabled"])
        if not any(current is table for current in self.tables) or table.revision != revision:
            self.status.set("Los datos cambiaron durante el ajuste. Ejecuta F9 con la columna actualizada.")
            return
        try:
            fits = future.result()
        except (ValueError, RuntimeError, OverflowError) as exc:
            self._error(exc)
            return
        table.fits[column] = fits
        if (self.loaded_model and self.loaded_model[0] is table
                and self.loaded_model[1] == column):
            self._clear_model()
        self._modified()
        self.status.set(f"{len(fits)} modelos ajustados en {table.name} · {table.headers[column]}.")
        if self.table is table and self._current_column() == column:
            self.notebook.select(self.results_tab)
            self._refresh_results()

    def _refresh_results(self):
        previous = self._selected_fit()
        column = self._current_column()
        self.result_heading.configure(text=f"{self.table.name}  /  {self.table.headers[column]}")
        try:
            values = self.table.values(column)
        except ValueError as exc:
            self.summary.configure(text=str(exc))
        else:
            self.summary.configure(text=f"{len(values)} observaciones")
        self.result_fits = self.table.fits.get(column, [])
        self.fit_grid.delete(*self.fit_grid.get_children())
        for index, fit in enumerate(self.result_fits):
            self.fit_grid.insert("", "end", iid=str(index), values=(
                fit.candidate.label, decimal(fit.aic), decimal(fit.bic), decimal(fit.ks),
                "—" if fit.p_value is None else decimal(fit.p_value)))
        if self.result_fits:
            self.fit_grid.selection_set(str(self.result_fits.index(previous) if previous in self.result_fits else 0))
        self._select_result()

    def _selected_fit(self):
        selected = self.fit_grid.selection()
        return self.result_fits[int(selected[0])] if selected and int(selected[0]) < len(self.result_fits) else None

    def _select_result(self, _event=None):
        fit = self._selected_fit()
        self.open_model_button.state(["!disabled" if fit else "disabled"])
        if fit:
            kind = "Discretos" if fit.candidate.discrete else "Continuos"
            summary = self._parameter_summary(fit)
            values = self.table.values(self._current_column())
            note = ""
            if not fit.candidate.discrete and len(np.unique(values)) < len(values):
                note = "\nHay empates: interpreta el valor p con cautela si los datos están redondeados."
            self.fit_details.configure(text=f"{kind} · {summary}\nOrden AIC ascendente. Valor p de KS por "
                                           f"simulación con reajuste ({fit.repetitions}/{BOOTSTRAP_REPETITIONS})."
                                           f"{note}")
        else:
            self.fit_details.configure(text="Pulsa F9 para ajustar la columna seleccionada.")
        self._draw_results()

    def _draw_results(self):
        ax = self.result_axes
        ax.clear()
        self.plot_ready = False
        try:
            values = self.table.values(self._current_column())
        except ValueError:
            ax.set_axis_off()
            ax.text(.5, .5, "Carga una columna numérica para ver su gráfica.", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            fit = self._selected_fit()
            x, frequencies = np.unique(values, return_counts=True)
            discrete = (fit.candidate.discrete if fit else
                        bool(np.all(values >= 0) and np.all(values == np.floor(values))))
            span = max(float(x[-1] - x[0]), .1)
            low, high = x[0] - span * .08, x[-1] + span * .08
            if self.chart_kind.get() == "Acumulada":
                ax.step(np.r_[low, x, high], np.r_[0, np.cumsum(frequencies) / len(values), 1],
                        where="post", color="#586c79", linewidth=2, label="Empírica")
                ax.set_ylabel("Probabilidad acumulada")
            elif discrete and len(x) <= 250:
                ax.bar(x, frequencies / len(values), width=.78, color="#afded8", label="Observada")
                ax.set_ylabel("Frecuencia relativa")
            elif discrete:
                ax.plot(x, frequencies / len(values), ".", color="#72998f", label="Observada")
                ax.set_ylabel("Frecuencia relativa")
            else:
                bins = min(60, max(5, int(np.sqrt(len(values)))))
                ax.hist(values, bins=bins, density=True, color="#afded8", edgecolor="white", label="Observada")
                ax.set_ylabel("Densidad")
            if fit:
                frozen = fit.frozen()
                if self.chart_kind.get() == "Acumulada":
                    if discrete:
                        grid = (np.arange(math.floor(low), math.ceil(high) + 1)
                                if high - low < 500 else np.unique(np.r_[low, x - 1, x, high]))
                        ax.step(grid, frozen.cdf(grid), where="post", color=TEAL, linewidth=2, label=fit.candidate.label)
                    else:
                        grid = np.linspace(low, high, 500)
                        ax.plot(grid, frozen.cdf(grid), color=TEAL, linewidth=2, label=fit.candidate.label)
                elif discrete:
                    grid = np.arange(math.floor(low), math.ceil(high) + 1) if high - low < 500 else x
                    ax.plot(grid, frozen.pmf(grid), "o", color=TEAL, markersize=4, label=fit.candidate.label)
                else:
                    grid = np.linspace(low, high, 500)
                    density = frozen.pdf(grid)
                    ax.plot(grid, np.where(np.isfinite(density), density, np.nan),
                            color=TEAL, linewidth=2, label=fit.candidate.label)
            ax.set_xlabel(self.table.headers[self._current_column()])
            ax.set_axis_on()
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#d6e3e7")
            ax.legend(frameon=False)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FuncFormatter(lambda value, _: decimal(value)))
            self.plot_ready = True
        self.result_figure.tight_layout()
        self.result_canvas.draw_idle()

    @staticmethod
    def _parameter_summary(fit):
        parameters = fit.candidate.explorer_parameters(fit.parameters)
        return ", ".join(f"{'λ' if key == 'lambda_param' else key} = {decimal(value)}"
                         for key, value in parameters.items())

    def _clear_model(self):
        self.loaded_model = None
        if self.notebook.select() == str(self.model_tab):
            self.notebook.select(self.results_tab)
        self.notebook.tab(self.model_tab, state="disabled")

    def open_selected_fit(self):
        fit = self._selected_fit()
        if fit:
            model = fit.candidate
            column = self._current_column()
            self.loaded_model = (self.table, column, fit)
            self.explorer.select_model(model.explorer_name, model.explorer_parameters(fit.parameters),
                                       discrete=model.discrete)
            self.model_title.configure(text=model.label)
            self.model_detail.configure(text=f"{self.table.name} / {self.table.headers[column]}  ·  "
                                             f"{self._parameter_summary(fit)}")
            self.notebook.tab(self.model_tab, state="normal")
            self.notebook.select(self.model_tab)

    def _error(self, exc):
        self.status.set(str(exc))
        messagebox.showerror("FITEC", str(exc), parent=self.root)

    def _confirm_close_project(self):
        self._finish_edit()
        if self.dirty:
            answer = messagebox.askyesnocancel("FITEC", "¿Guardar los cambios de este proyecto?", parent=self.root)
            return answer is not None and (not answer or self.save_project())
        return True

    def _replace_project(self, tables, path=None, active=0, candidates=None, data_kind="Automático", trials=None):
        self._clear_model()
        self.trials = trials
        self.tables, self.project_path = tables, path
        self.active, self.page = active, 0
        self.dirty = False
        self._set_fit_configuration(candidates, data_kind)
        self._update_title()
        self._refresh_data(0)
        self._refresh_results()
        self.notebook.select(self.data_tab)

    def new_project(self):
        if self._confirm_close_project():
            self._replace_project([DataTable("Tabla 1", ["A"])])
            self.status.set("Proyecto nuevo.")

    def open_project(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("Proyecto FITEC", "*.fitec")])
        if not path:
            return
        try:
            source = Path(path)
            document = json.loads(source.read_text(encoding="utf-8-sig"))
            if document["version"] != 1:
                raise ValueError("Versión de proyecto no compatible.")
            tables = [DataTable.from_document(item) for item in document["tables"]]
            active = int(document.get("active", 0))
            trials = document.get("binomial_trials")
            if trials is not None and (type(trials) is not int or trials < 1):
                raise ValueError("Número de ensayos inválido en el proyecto.")
            candidates = document.get("fit_candidates")
            data_kind = document.get("data_kind", "Automático")
            if not tables or not 0 <= active < len(tables):
                raise ValueError("El proyecto no tiene una tabla activa válida.")
            if (candidates is not None and
                    (not isinstance(candidates, list) or not candidates
                     or any(name not in MODELS for name in candidates))):
                raise ValueError("El proyecto contiene una selección de modelos inválida.")
            if data_kind not in ("Automático", "Continuos", "Recuentos"):
                raise ValueError("El proyecto contiene un tipo de datos inválido.")
            if self._confirm_close_project():
                self._replace_project(tables, source, active,
                                      None if candidates is None else tuple(candidates), data_kind, trials)
                self.status.set(f"Proyecto abierto: {source.name}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self._error(exc)

    def save_project(self):
        return self._save(self.project_path) if self.project_path else self.save_as()

    def save_as(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".fitec",
                                            filetypes=[("Proyecto FITEC", "*.fitec")])
        return self._save(Path(path)) if path else False

    def _save(self, path):
        self._finish_edit()
        try:
            atomic_json(path, {"version": 1, "active": self.active,
                               "binomial_trials": self.trials, "fit_candidates": self.fit_candidates, "data_kind": self.data_kind.get(),
                               "tables": [table.as_document() for table in self.tables]})
        except (OSError, ValueError) as exc:
            self._error(exc)
            return False
        self.project_path, self.dirty = path, False
        self._update_title()
        self.status.set(f"Proyecto guardado: {path.name}")
        return True

    def _save_csv(self, rows):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return False
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                csv.writer(handle, delimiter=";").writerows(rows)
            self.status.set(f"CSV guardado: {Path(path).name}")
            return True
        except OSError as exc:
            self._error(exc)
            return False

    def export_data(self):
        self._finish_edit()
        self._save_csv([self.table.headers, *self.table.rows])

    def export_fits(self):
        fits = self.table.fits.get(self._current_column(), [])
        if not fits:
            self.status.set("Ajusta la columna seleccionada antes de exportar.")
            return
        rows = [["Distribución", "Parámetros", "AIC", "BIC", "KS", "Valor p", "Réplicas válidas"]]
        for fit in fits:
            params = fit.candidate.explorer_parameters(fit.parameters)
            rows.append([fit.candidate.label, json.dumps(params, ensure_ascii=False),
                         decimal(fit.aic, 12), decimal(fit.bic, 12), decimal(fit.ks, 12),
                         "" if fit.p_value is None else decimal(fit.p_value, 12), fit.repetitions])
        self._save_csv(rows)

    def export_plot(self):
        if not self.plot_ready:
            self.status.set("Carga una columna para generar una gráfica antes de exportar.")
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                self.result_figure.savefig(path, dpi=180)
                self.status.set(f"Gráfica guardada: {Path(path).name}")
            except OSError as exc:
                self._error(exc)

    def generator(self):
        if self.loaded_model:
            SampleDialog(self, self.loaded_model[2])

    def close(self):
        if self.job is not None:
            self.status.set("Espera a que termine el ajuste antes de cerrar el programa.")
            return
        if self._confirm_close_project():
            self.worker.shutdown(wait=False, cancel_futures=True)
            self.root.destroy()


def dialog_body(workspace, title):
    window = tk.Toplevel(workspace.root)
    window.title(f"FITEC · {title}")
    window.transient(workspace.root)
    window.grab_set()
    window.resizable(False, False)
    window.bind("<Escape>", lambda _event: window.destroy())
    body = ttk.Frame(window, style="Panel.TFrame", padding=18)
    body.pack(fill="both", expand=True)
    return window, body


class FitSettingsDialog:
    def __init__(self, workspace):
        self.workspace = workspace
        self.window, body = dialog_body(workspace, "Configurar ajuste")
        self.strategy = tk.StringVar(value="auto" if workspace.fit_candidates is None else "manual")
        self.kind = tk.StringVar(value=workspace.data_kind.get())
        self.trials = tk.StringVar(value=str(workspace.trials or ""))
        self.checks = {}
        self.checkbuttons = []
        ttk.Label(body, text="Configurar ajuste", style="Section.TLabel",
                  image=workspace.icons["settings"], compound="left").pack(anchor="w")
        choices = ttk.Frame(body, style="Panel.TFrame")
        choices.pack(fill="x", pady=(10, 0))
        for label, value in (("Automático", "auto"), ("Solo las distribuciones marcadas", "manual")):
            ttk.Radiobutton(choices, text=label, value=value, variable=self.strategy,
                            command=self._update_controls).pack(anchor="w", pady=3)
        ttk.Label(body, text="Tipo de datos").pack(anchor="w", pady=(8, 3))
        self.kind_picker = ttk.Combobox(body, textvariable=self.kind, state="readonly",
                                        values=("Automático", "Continuos", "Recuentos"))
        self.kind_picker.pack(fill="x")
        self.kind_picker.bind("<<ComboboxSelected>>", lambda _event: self._update_controls())
        groups = ttk.Frame(body, style="Panel.TFrame")
        groups.pack(fill="x", pady=12)
        for label, models in (("Continuos", CONTINUOUS_MODELS), ("Recuentos", DISCRETE_MODELS)):
            panel = ttk.Frame(groups, style="Panel.TFrame", padding=(0, 0, 18, 0))
            panel.pack(side="left", anchor="n", fill="x", expand=True)
            ttk.Label(panel, text=label, font=("Inter", 10, "bold")).pack(anchor="w", pady=(0, 4))
            for model in models:
                enabled = workspace.fit_candidates is None or model.scipy_name in workspace.fit_candidates
                var = tk.BooleanVar(value=enabled)
                self.checks[model.scipy_name] = var
                button = ttk.Checkbutton(panel, text=model.label, variable=var)
                button.pack(anchor="w", pady=2)
                self.checkbuttons.append((label, button))
        ttk.Label(body, text="Binomial · Ensayos conocidos (n)").pack(anchor="w")
        ttk.Entry(body, textvariable=self.trials).pack(fill="x")
        ttk.Label(body, text="Deja el campo vacío para omitir la binomial.").pack(anchor="w")
        self.message = ttk.Label(body, style="Help.TLabel", wraplength=340)
        self.message.pack(anchor="w")
        workspace._button(body, "Guardar configuración", "arrow", self._save).pack(fill="x", pady=(10, 0))
        self._update_controls()

    def _update_controls(self):
        manual = self.strategy.get() == "manual"
        kind = self.kind.get()
        for label, button in self.checkbuttons:
            enabled = manual and kind in ("Automático", label)
            button.state(["!disabled" if enabled else "disabled"])
        self.message.configure(text=("F9 ajustará los modelos marcados que correspondan al tipo de datos."
                                     if manual else "F9 probará los modelos compatibles con la columna."))

    def _save(self):
        kind = self.kind.get()
        candidates = None
        if self.strategy.get() == "manual":
            candidates = tuple(name for name, var in self.checks.items() if var.get())
            allowed = {
                "Continuos": CONTINUOUS_MODELS,
                "Recuentos": DISCRETE_MODELS,
            }.get(kind, (*CONTINUOUS_MODELS, *DISCRETE_MODELS))
            if not any(model.scipy_name in candidates for model in allowed):
                self.message.configure(text="Marca al menos una distribución del tipo elegido.")
                return
        app = self.workspace
        raw = self.trials.get().strip()
        try:
            trials = int(raw) if raw else None
            if trials is not None and trials < 1:
                raise ValueError
        except ValueError:
            self.message.configure(text="Indica un número entero positivo de ensayos.")
            return
        if (candidates, kind, trials) != (app.fit_candidates, app.data_kind.get(), app.trials):
            app.trials = trials
            app._set_fit_configuration(candidates, kind)
            app._modified()
        app.status.set("Configuración guardada. Pulsa F9 para aplicarla a la columna.")
        self.window.destroy()


class PrepareDataDialog:
    OPERATIONS = ("Conversión lineal", "Logaritmo natural", "Raíz cuadrada", "Estandarizar (z)")

    def __init__(self, workspace, table, column):
        self.workspace = workspace
        self.table = table
        self.column = column
        self.source = table.values(column)
        self.window, body = dialog_body(workspace, "Preparar datos")
        self.operation = tk.StringVar(value=self.OPERATIONS[0])
        self.factor = tk.StringVar(value="1")
        self.offset = tk.StringVar(value="0")
        stem = f"{table.headers[column]} (preparada)"
        name = stem
        suffix = 2
        while name in table.headers:
            name = f"{stem} {suffix}"
            suffix += 1
        self.name = tk.StringVar(value=name)
        ttk.Label(body, text="Preparar datos", style="Section.TLabel",
                  image=workspace.icons["prepare"], compound="left").pack(anchor="w")
        ttk.Label(body, text=f"Origen: {table.name} / {table.headers[column]}",
                  style="Help.TLabel", wraplength=360).pack(anchor="w", pady=(5, 10))
        selector = ttk.Combobox(body, textvariable=self.operation, values=self.OPERATIONS,
                                width=28, state="readonly")
        selector.pack(fill="x")
        self.controls = ttk.Frame(body, style="Panel.TFrame")
        self.controls.pack(fill="x", pady=6)
        for label, var in (("Factor", self.factor), ("Desplazamiento", self.offset)):
            ttk.Label(self.controls, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(self.controls, textvariable=var, width=7).pack(side="left", padx=(0, 10))
        self.operation_help = ttk.Label(body, style="Help.TLabel", wraplength=365)
        self.operation_help.pack(anchor="w")
        ttk.Label(body, text="Nombre de la columna nueva").pack(anchor="w", pady=(10, 4))
        ttk.Entry(body, textvariable=self.name).pack(fill="x")
        self.preview = ttk.Treeview(body, columns=("original", "prepared"), show="headings", height=6)
        for key, label in (("original", "Original"), ("prepared", "Vista previa")):
            self.preview.heading(key, text=label)
            self.preview.column(key, width=175, anchor="e")
        self.preview.pack(fill="x", pady=(12, 5))
        self.message = ttk.Label(body, style="Error.TLabel", wraplength=365)
        self.message.pack(anchor="w")
        self.apply_button = workspace._button(body, "Crear columna", "add", self._apply)
        self.apply_button.pack(fill="x", pady=(8, 0))
        for var in (self.operation, self.factor, self.offset):
            var.trace_add("write", self._refresh)
        self._refresh()

    def _transform(self):
        values = self.source
        operation = self.operation.get()
        with np.errstate(over="ignore", invalid="ignore"):
            if operation == self.OPERATIONS[0]:
                result = values * number(self.factor.get()) + number(self.offset.get())
            elif operation == self.OPERATIONS[1]:
                if np.any(values <= 0):
                    raise ValueError("El logaritmo requiere valores mayores que cero.")
                result = np.log(values)
            elif operation == self.OPERATIONS[2]:
                if np.any(values < 0):
                    raise ValueError("La raíz cuadrada requiere valores no negativos.")
                result = np.sqrt(values)
            else:
                scale = np.std(values, ddof=1) if len(values) > 1 else 0
                if not np.isfinite(scale) or scale == 0:
                    raise ValueError("Estandarizar requiere al menos dos valores distintos y varianza finita.")
                result = (values - np.mean(values)) / scale
        if not np.all(np.isfinite(result)):
            raise ValueError("La transformación produjo valores no finitos.")
        return result

    def _refresh(self, *_):
        operation = self.operation.get()
        if operation == self.OPERATIONS[0]:
            self.controls.pack(fill="x", pady=6, before=self.operation_help)
        else:
            self.controls.pack_forget()
        descriptions = {
            self.OPERATIONS[0]: "Resultado = original × factor + desplazamiento. De minutos a segundos, usa factor 60.",
            self.OPERATIONS[1]: "Aplica ln(x) a cada valor positivo.",
            self.OPERATIONS[2]: "Aplica √x a cada valor no negativo.",
            self.OPERATIONS[3]: "Resta la media y divide por la desviación estándar muestral.",
        }
        self.operation_help.configure(text=descriptions[operation])
        self.preview.delete(*self.preview.get_children())
        try:
            result = self._transform()
        except ValueError as exc:
            self.message.configure(text=str(exc))
            self.apply_button.state(["disabled"])
            return
        self.message.configure(text="")
        self.apply_button.state(["!disabled"])
        for original, prepared in zip(self.source[:6], result[:6]):
            self.preview.insert("", "end", values=(decimal(original, 8), decimal(prepared, 8)))

    def _apply(self):
        name = self.name.get().strip()
        if not name or name in self.table.headers:
            self.message.configure(text="Escribe un nombre nuevo para la columna.")
            return
        result = iter(self._transform())
        cells = [repr(float(next(result))).replace(".", ",")
                 if self.column < len(row) and row[self.column].strip() else ""
                 for row in self.table.rows]
        self.workspace._append_prepared_column(self.table, name, cells)
        self.window.destroy()


class SampleDialog:
    def __init__(self, workspace, fit):
        self.workspace = workspace
        self.fit = fit
        self.window = tk.Toplevel(workspace.root)
        self.window.title("FITEC · Generar desde el modelo")
        self.window.transient(workspace.root)
        self.window.grab_set()
        self.window.resizable(False, False)
        body = ttk.Frame(self.window, style="Panel.TFrame", padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=fit.candidate.label, style="Section.TLabel").pack(anchor="w")
        ttk.Label(body, text=workspace._parameter_summary(fit), wraplength=295,
                  style="Help.TLabel").pack(anchor="w", pady=(4, 8))
        self.count = tk.StringVar(value="500")
        self.seed = tk.StringVar(value="42")
        self.destination = tk.StringVar(value="Nueva tabla")
        for label, var in (("Tamaño de muestra", self.count), ("Semilla", self.seed)):
            ttk.Label(body, text=label).pack(anchor="w", pady=(8, 3))
            ttk.Entry(body, textvariable=var, width=30).pack(fill="x")
        ttk.Label(body, text="Destino").pack(anchor="w", pady=(8, 3))
        ttk.Combobox(body, textvariable=self.destination, state="readonly",
                     values=("Nueva tabla", "Portapapeles", "Archivo CSV")).pack(fill="x")
        self.error = ttk.Label(body, style="Error.TLabel", wraplength=275)
        self.error.pack(anchor="w", pady=(6, 0))
        ttk.Button(body, text="Generar muestra", command=self._generate).pack(fill="x", pady=(8, 0))

    def _generate(self):
        try:
            count, seed = int(self.count.get()), int(self.seed.get())
            if not 1 <= count <= 100_000 or not 0 <= seed <= 2**32 - 1:
                raise ValueError("Usa entre 1 y 100 000 puntos y una semilla entera entre 0 y 4 294 967 295.")
            values = self.fit.frozen().rvs(size=count, random_state=seed)
            if not np.all(np.isfinite(values)):
                raise ValueError("El modelo produjo valores no finitos.")
            rows = [[str(int(value)) if self.fit.candidate.discrete
                     else repr(float(value)).replace(".", ",")] for value in values]
        except (ValueError, TypeError, OverflowError) as exc:
            self.error.configure(text=str(exc))
            return
        app = self.workspace
        name = self.fit.candidate.label
        target = self.destination.get()
        header = f"Muestra {name} (semilla {seed})"
        if target == "Nueva tabla":
            app._new_table(f"Muestra {name}", [header], rows)
        elif target == "Portapapeles":
            app._copy_text("\n".join(row[0] for row in rows))
        elif not app._save_csv([[header], *rows]):
            return
        self.window.destroy()
        app.status.set(f"{count} valores de {name} generados con semilla {seed} · {target}.")
