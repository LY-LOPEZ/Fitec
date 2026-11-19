"""FITEC: native desktop probability explorer for Windows and Python."""

from __future__ import annotations

import math
import re

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover - headless environments without Tk
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

import numpy as np

if tk is not None:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from fitec_theme import NIGHT, is_night
from src.distributions import (
    BetaDistribution,
    BinomialDistribution,
    CauchyDistribution,
    ChiSquareDistribution,
    DiscreteUniformDistribution,
    ErlangDistribution,
    ExponentialDistribution,
    FisherFDistribution,
    GammaDistribution,
    GeometricDistribution,
    HypergeometricDistribution,
    LognormalDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    StudentTDistribution,
    UniformDistribution,
    WeibullDistribution,
    ZeroGeometricDistribution,
)


CONTINUOUS = {
    "Uniforme": (UniformDistribution, {"a": 0.0, "b": 10.0}, {"a": "a · Límite inferior", "b": "b · Límite superior"}),
    "Exponencial": (ExponentialDistribution, {"lambda_param": 0.2}, {"lambda_param": "λ · Tasa"}),
    "Erlang": (ErlangDistribution, {"k": 2, "lambda_param": 1.0}, {"k": "k · Forma (entero)", "lambda_param": "λ · Tasa"}),
    "Normal": (NormalDistribution, {"mu": 0.0, "sigma": 1.0}, {"mu": "μ · Media", "sigma": "σ · Desviación estándar"}),
    "Chi-cuadrado": (ChiSquareDistribution, {"df": 4}, {"df": "m · Grados de libertad"}),
    "t de Student": (StudentTDistribution, {"df": 10.0}, {"df": "m · Grados de libertad"}),
    "F de Fisher": (
        FisherFDistribution,
        {"dfn": 5.0, "dfd": 10.0},
        {"dfn": "m · Grados de libertad del numerador",
         "dfd": "n · Grados de libertad del denominador"},
    ),
    # Distribuciones adicionales que ya ofrecía Probviz.
    "Beta": (BetaDistribution, {"alpha": 2.0, "beta": 2.0}, {"alpha": "α", "beta": "β"}),
    "Gamma": (GammaDistribution, {"shape": 2.0, "scale": 2.0}, {"shape": "Forma", "scale": "Escala"}),
    "Weibull": (WeibullDistribution, {"shape": 1.5, "scale": 1.0}, {"shape": "Forma", "scale": "Escala"}),
    "Lognormal": (LognormalDistribution, {"mu": 0.0, "sigma": 1.0}, {"mu": "μ", "sigma": "σ"}),
    "Cauchy": (CauchyDistribution, {"x0": 0.0, "gamma": 1.0}, {"x0": "Ubicación", "gamma": "Escala"}),
}
DISCRETE = {
    "Geométrica": (ZeroGeometricDistribution, {"p": 0.3}, {"p": "p · Probabilidad de éxito"}),
    "Geométrica (ensayos)": (GeometricDistribution, {"p": 0.3}, {"p": "p · Probabilidad de éxito"}),
    "Binomial negativa": (NegativeBinomialDistribution, {"r": 5, "p": 0.4}, {"r": "r · Éxitos", "p": "p · Probabilidad de éxito"}),
    "Binomial": (BinomialDistribution, {"n": 10, "p": 0.5}, {"n": "n · Ensayos", "p": "p · Probabilidad de éxito"}),
    "Poisson": (PoissonDistribution, {"lambda_param": 4.0}, {"lambda_param": "λ · Tasa"}),
    "Uniforme discreta": (DiscreteUniformDistribution, {"low": 1, "high": 6}, {"low": "a · Mínimo", "high": "b · Máximo"}),
    "Hipergeométrica": (HypergeometricDistribution, {"M": 20, "n": 7, "N": 8}, {"M": "M · Población", "n": "K · Éxitos en población", "N": "n · Extracciones"}),
}
PROBABILITY_LABELS = {
    "hasta": "Hasta (≤)",
    "encima": "Más de (>)",
    "menos": "Menos de (<)",
    "al_menos": "Al menos (≥)",
    "entre": "Entre",
    "exacta": "Exacto (=)",
}
INTEGER_PARAMETERS = {"k", "n", "r", "M", "N", "low", "high"}

BACKGROUND = "#edf3f5"
SURFACE = "#ffffff"
HEADER = "#103746"
TEXT = "#153744"
MUTED = "#506671"
BORDER = "#d6e3e7"
TEAL = "#087d77"
MINT = "#e1f5f0"
ERROR = "#a52620"
FONT = "Inter"


def format_number(value, precision=6):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    val = float(value)
    if val == 0.0:
        return "0"
    return f"{val:.{precision}g}".replace(".", ",")


def format_probability(result: float) -> tuple[str, str]:
    """Format a probability in [0, 1] as both percentage and exact decimal."""
    if result <= 0.0:
        return "0,00 %", "= 0"
    if result >= 1.0:
        return "100,00 %", "= 1"
    pct = result * 100.0
    if pct < 0.01:
        pct_text = f"{pct:.4g}".replace(".", ",") + " %"
    elif pct > 99.99:
        pct_text = f"{pct:.6f}".rstrip("0").rstrip(".").replace(".", ",") + " %"
        if pct_text == "100 %":
            pct_text = "> 99,999999 %"
    else:
        pct_text = f"{pct:.2f}".replace(".", ",") + " %"
    decimal_text = format_number(result, 6)
    if decimal_text == "1":
        decimal_text = repr(float(result)).replace(".", ",")
    return pct_text, "= " + decimal_text


def read_number(text):
    cleaned = text.strip().replace(",", ".")
    if not cleaned:
        raise ValueError("Ingresa un número finito.")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"«{text.strip()}» no es un número válido.") from exc
    if not math.isfinite(value):
        raise ValueError("Ingresa un número finito.")
    return value


def _translate_error(message: str) -> str:
    translations = {
        "sigma must be positive": "σ debe ser mayor que cero.",
        "lambda must be positive": "λ debe ser mayor que cero.",
        "a must be less than b": "El límite inferior a debe ser menor que b.",
        "alpha and beta must be positive": "α y β deben ser mayores que cero.",
        "shape and scale must be positive": "La forma y la escala deben ser mayores que cero.",
        "df must be positive": "m debe ser mayor que cero.",
        "gamma must be positive": "La escala debe ser mayor que cero.",
        "n must be positive": "n debe ser un entero positivo.",
        "r must be positive": "r debe ser un entero positivo.",
        "p must be between 0 and 1 (exclusive of 0)": "p debe estar en el intervalo (0, 1].",
        "p must be between 0 and 1": "p debe estar entre 0 y 1.",
        "low must be less than high": "El mínimo a debe ser menor que el máximo b.",
        "M must be positive, n and N must be non-negative": "M debe ser positivo; K y n no pueden ser negativos.",
        "n cannot be greater than M": "K (éxitos en la población) no puede superar a M.",
        "N cannot be greater than M": "n (extracciones) no puede superar a M.",
    }
    return translations.get(message, message)


class ParameterInputError(ValueError):
    def __init__(self, key, message):
        self.key = key
        super().__init__(message)


def parse_parameter(name, key, text):
    symbol = {"df": "m", "dfn": "m", "dfd": "n"}.get(key, key)
    try:
        value = read_number(text)
    except ValueError as exc:
        raise ParameterInputError(key, f"{symbol}: ingresa un valor numérico finito.") from exc
    if key in {"df", "dfn", "dfd"} and value <= 0:
        raise ParameterInputError(key, f"{symbol} debe ser mayor que cero.")
    if key in INTEGER_PARAMETERS or (name == "Chi-cuadrado" and key == "df"):
        if not value.is_integer():
            raise ParameterInputError(key, f"{symbol} debe ser un número entero.")
        return int(value)
    return value


def parse_observations(text):
    tokens = re.split(r",(?=\s)|[;\s]+", text.strip())
    values = [
        read_number(part)
        for token in tokens if token
        for part in (
            token.split(",")
            if token.count(",") > 1 or ("." in token and "," in token)
            else [token]
        )
    ]
    if not values:
        raise ValueError("Pega al menos un dato numérico.")
    return np.asarray(values, dtype=float)


def probability(dist, kind: str, first: float, second: float | None = None) -> float:
    """Return the requested probability, including exact tail and discrete endpoint handling."""
    if not math.isfinite(first):
        raise ValueError("Ingresa un límite finito.")
    if kind == "exacta":
        if not dist.is_discrete or not float(first).is_integer():
            raise ValueError("P(X=k) requiere una distribución discreta y un k entero.")
        return float(np.clip(dist.pdf(int(round(first))), 0.0, 1.0))
    if kind == "hasta":
        boundary = math.floor(first) if dist.is_discrete else first
        return float(np.clip(dist.cdf(boundary), 0.0, 1.0))
    if kind == "menos":
        boundary = math.ceil(first) - 1 if dist.is_discrete else first
        return float(np.clip(dist.cdf(boundary), 0.0, 1.0))
    if kind == "encima":
        boundary = math.floor(first) if dist.is_discrete else first
        return float(np.clip(dist.sf(boundary), 0.0, 1.0))
    if kind == "al_menos":
        boundary = math.ceil(first) - 1 if dist.is_discrete else first
        return float(np.clip(dist.sf(boundary), 0.0, 1.0))
    if kind != "entre":
        raise ValueError(f"Tipo de probabilidad no válido: {kind}")
    if second is None or not math.isfinite(second):
        raise ValueError("Indica ambos extremos del intervalo.")
    if first > second:
        raise ValueError("El límite inferior debe ser menor o igual al superior.")
    if dist.is_discrete:
        lo, hi = math.ceil(first), math.floor(second)
        if lo > hi:
            return 0.0
        if lo == hi:
            return float(np.clip(dist.pdf(lo), 0.0, 1.0))
        lower_cdf = float(dist.cdf(lo - 1))
        diff = dist.sf(lo - 1) - dist.sf(hi) if lower_cdf > 0.5 else dist.cdf(hi) - lower_cdf
        return float(np.clip(diff, 0.0, 1.0))
    if first == second:
        return 0.0
    lower_cdf = float(dist.cdf(first))
    diff = dist.sf(first) - dist.sf(second) if lower_cdf > 0.5 else dist.cdf(second) - lower_cdf
    return float(np.clip(diff, 0.0, 1.0))


class FitecApp:
    catalog = {"Continuas": CONTINUOUS, "Discretas": DISCRETE}

    def __init__(self, root, *, embedded=False, fixed_model=False, show_moments=True,
                 external_controls=False):
        self.root = root
        self.embedded = embedded
        self.fixed_model = fixed_model
        self.show_moments = show_moments
        self.external_controls = external_controls
        if not embedded:
            root.title("FITEC")
            root.geometry("1240x730")
            root.minsize(990, 610)
            root.configure(bg=BACKGROUND)
        self.mode = tk.StringVar(value="Continuas")
        self.distribution = tk.StringVar(value="Normal")
        self.prob_kind = tk.StringVar(value="entre")
        self.graph_mode = tk.StringVar(value="Densidad")
        self.parameter_vars: dict[str, tk.StringVar] = {}
        self.parameter_errors: dict[str, ttk.Label] = {}
        self.parameter_sliders: dict[str, ttk.Scale] = {}
        self.first_var = tk.StringVar(value="-1")
        self.second_var = tk.StringVar(value="1")
        self.sample_text = "0, 1, 2, 2, 3, 4"
        self._update_id = None
        self._animation_id = None
        self.animate = True
        self._zoom_factor = 1.0
        self._dragged_limit = None
        self._plot_limits = {}
        self._plot_data = None
        self._last_plot_style = None
        self._build()
        self._rebuild_parameters()
        self.update_plot()

    def _build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("Panel.TFrame", background=SURFACE)
        style.configure("Header.TFrame", background=HEADER)
        style.configure("Mint.TFrame", background=MINT)
        style.configure("TLabel", font=(FONT, 10), foreground=TEXT, background=SURFACE)
        style.configure("Section.TLabel", font=(FONT, 15, "bold"))
        style.configure("Help.TLabel", font=(FONT, 10), foreground=MUTED)
        style.configure("Error.TLabel", font=(FONT, 9), foreground=ERROR)
        style.configure("TEntry", padding=(8, 5), fieldbackground=SURFACE, bordercolor=BORDER)
        style.configure("TCombobox", padding=(8, 5), foreground=TEXT, bordercolor=BORDER)
        style.map("TCombobox", fieldbackground=[("readonly", SURFACE)])
        style.configure("TButton", font=(FONT, 10, "bold"), padding=(8, 8),
                        foreground=TEAL, background=SURFACE, bordercolor=TEAL)
        style.map("TButton", background=[("active", MINT)])
        style.configure("Toolbutton", font=(FONT, 9), padding=(6, 5),
                        foreground=TEXT, background="#f1f5f6", bordercolor=BORDER)
        style.map("Toolbutton", background=[("selected", TEAL), ("active", MINT)],
                  foreground=[("selected", SURFACE), ("disabled", MUTED)])
        style.configure("Horizontal.TScale", background=SURFACE, troughcolor=MINT)

        if not self.embedded:
            header = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 15))
            header.pack(fill="x")
            ttk.Label(header, text="FITEC", background=HEADER, foreground=SURFACE,
                      font=(FONT, 27, "bold")).pack(side="left")
            ttk.Separator(header, orient="vertical").pack(side="left", fill="y", padx=19, pady=4)
            ttk.Label(header, text="Explorador de distribuciones", background=HEADER,
                      foreground=SURFACE, font=(FONT, 15)).pack(side="left")

        body = ttk.Frame(self.root, padding=(12, 10))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, minsize=0 if self.fixed_model else 224)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self.left = ttk.Frame(body, style="Panel.TFrame", padding=12)
        center = ttk.Frame(body, style="Panel.TFrame", padding=12)
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        center.grid(row=0, column=1, sticky="nsew")
        if self.fixed_model:
            self.left.grid_remove()

        ttk.Label(self.left, text="Distribución", style="Section.TLabel").pack(anchor="w")
        modes = ttk.Frame(self.left, style="Panel.TFrame")
        modes.pack(fill="x", pady=(10, 12))
        for label in ("Continuas", "Discretas", "Empírica"):
            ttk.Radiobutton(modes, text=label, value=label, variable=self.mode,
                            style="Toolbutton", command=self._distribution_changed).pack(
                                side="left", fill="x", expand=True)
        self.dist_controls = ttk.Frame(self.left, style="Panel.TFrame")
        self.dist_controls.pack(fill="x")
        self.dist_combo = ttk.Combobox(self.dist_controls, textvariable=self.distribution,
                                       state="readonly", width=19)
        self.dist_combo.pack(fill="x")
        self.dist_combo.bind("<<ComboboxSelected>>", self._distribution_changed)
        ttk.Separator(self.dist_controls).pack(fill="x", pady=12)
        ttk.Label(self.dist_controls, text="Parámetros", style="Section.TLabel").pack(anchor="w")
        self.parameter_box = ttk.Frame(self.dist_controls, style="Panel.TFrame")
        self.parameter_box.pack(fill="x", pady=(6, 0))
        self.data_box = ttk.Frame(self.left, style="Panel.TFrame")
        ttk.Label(self.data_box, text="Datos observados", style="Section.TLabel").pack(anchor="w")
        ttk.Label(self.data_box, text="Pega una columna o separa los datos por espacios o punto y coma.",
                  style="Help.TLabel", wraplength=204).pack(anchor="w", pady=(6, 4))
        self.data_text = tk.Text(self.data_box, height=8, width=20, font=(FONT, 10),
                                 relief="flat", highlightthickness=1, highlightbackground=BORDER,
                                 undo=True)
        self.data_text.pack(fill="x")
        self.data_text.insert("1.0", self.sample_text)
        self.data_text.bind("<<Modified>>", self._data_changed)
        self.data_text.edit_modified(False)
        if not self.external_controls:
            ttk.Button(self.left, text="Restablecer", command=self.reset_values).pack(
                side="bottom", fill="x", pady=(14, 0))

        center.columnconfigure(0, weight=1)
        center.rowconfigure(2, weight=1)
        toolbar = ttk.Frame(center, style="Panel.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Label(toolbar, text="Gráfica", style="Section.TLabel").pack(side="left")
        self.graph_actions = ttk.Frame(center, style="Panel.TFrame")
        self.graph_actions.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.graph_buttons = {}
        if not self.external_controls:
            toggles = ttk.Frame(self.graph_actions, style="Panel.TFrame")
            toggles.pack(side="right")
            for option in ("Densidad", "Acumulada"):
                button = ttk.Radiobutton(toggles, text=option, value=option, variable=self.graph_mode,
                                         style="Toolbutton", command=self.update_plot)
                button.pack(side="left")
                self.graph_buttons[option] = button
        self.figure = Figure(figsize=(5.5, 2.5), dpi=100, facecolor=SURFACE)
        self.figure.subplots_adjust(left=.12, right=.975, bottom=.18, top=.93)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=center)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        self.stats_labels = []
        if self.show_moments:
            stats_frame = ttk.Frame(center if self.fixed_model else self.left, style="Panel.TFrame")
            self.stats_frame = stats_frame
            if self.fixed_model:
                stats_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
            else:
                stats_frame.pack(side="bottom", fill="x", pady=(10, 0))
                ttk.Separator(stats_frame).grid(row=0, column=0, columnspan=2,
                                                sticky="ew", pady=(2, 7))
                stats_frame.columnconfigure(1, weight=1)
            for column, caption in enumerate(("Media E[X]", "Varianza V[X]", "Desv. estándar σ")):
                label = ttk.Label(stats_frame, text=caption, style="Help.TLabel")
                value = ttk.Label(stats_frame, font=(FONT, 10, "bold"))
                if self.fixed_model:
                    stats_frame.columnconfigure(column, weight=1)
                    label.grid(row=0, column=column)
                    value.grid(row=1, column=column)
                else:
                    label.grid(row=column + 1, column=0, sticky="w", pady=2)
                    value.grid(row=column + 1, column=1, sticky="e", padx=(8, 0), pady=2)
                self.stats_labels.append(value)

        self.prob_frame = ttk.Frame(center, style="Mint.TFrame", padding=(12, 8))
        self.prob_frame.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        result_panel = ttk.Frame(self.prob_frame, style="Mint.TFrame", padding=(8, 0))
        result_panel.pack(side="right", padx=(4, 0))
        ttk.Separator(self.prob_frame, orient="vertical").pack(side="right", fill="y", padx=8)
        controls = ttk.Frame(self.prob_frame, style="Mint.TFrame")
        controls.pack(side="left", fill="x", expand=True)
        ttk.Label(controls, text="Probabilidad", background=MINT, font=(FONT, 11, "bold")).pack(anchor="w")
        row = ttk.Frame(controls, style="Mint.TFrame")
        row.pack(fill="x", pady=(4, 0))
        choices = ttk.Frame(row, style="Mint.TFrame")
        choices.grid(row=0, column=0, sticky="w")
        for idx, (key, caption) in enumerate((("hasta", "Hasta"), ("encima", "Más de"),
                                               ("entre", "Entre"), ("exacta", "Exacto"))):
            btn = ttk.Radiobutton(
                choices, text=caption, value=key,
                variable=self.prob_kind, style="Toolbutton",
                command=self._probability_choice,
            )
            btn.grid(row=0, column=idx, sticky="ew", padx=(0, 3), pady=1)
            if key == "exacta":
                self.exact_button = btn
        self.first_group, self.first_label, self.first_error = self._limit_input(row, self.first_var)
        self.second_group, self.second_label, self.second_error = self._limit_input(row, self.second_var)
        self.second_label.configure(text="y")

        def arrange_inputs(_event=None):
            between = self.prob_kind.get() == "entre"
            required = choices.winfo_reqwidth() + self.first_group.winfo_reqwidth() + 5
            if between:
                required += self.second_group.winfo_reqwidth() + 7
            inline = row.winfo_width() >= required
            choices.grid_configure(columnspan=1 if inline else 2)
            field_row, column = (0, 1) if inline else (1, 0)
            for index, (group, error) in enumerate(((self.first_group, self.first_error),
                                                     (self.second_group, self.second_error))):
                if index and not between:
                    group.grid_remove()
                    error.grid_remove()
                    continue
                pad = (7 if index else 5, 0)
                group.grid(row=field_row, column=column + index, sticky="w", padx=pad)
                error.grid(row=field_row + 1, column=column + index, sticky="w", padx=pad)

        self._arrange_inputs = arrange_inputs
        row.bind("<Configure>", arrange_inputs)
        self.result = ttk.Label(result_panel, text="—", background=MINT, foreground="#075a56",
                                font=(FONT, 22, "bold"))
        self.result.pack()
        self.decimal_result = ttk.Label(result_panel, text="", background=MINT, foreground="#075a56",
                                        font=(FONT, 11))
        self.formula = ttk.Label(result_panel, background=MINT, foreground="#075a56",
                                 font=(FONT, 10, "bold"), wraplength=155, justify="center")
        self.formula.pack(pady=(4, 0))
        self.prob_error = ttk.Label(controls, style="Error.TLabel", background=MINT, wraplength=260)
        self.prob_error.pack(anchor="w")
        for var in (self.first_var, self.second_var):
            var.trace_add("write", self._schedule_update)
        self.canvas.mpl_connect("motion_notify_event", self._plot_motion)
        self.canvas.mpl_connect("button_press_event", self._begin_drag)
        self.canvas.mpl_connect("button_release_event", self._end_drag)
        self.canvas.mpl_connect("axes_leave_event", self._hide_coordinates)

    def _limit_input(self, parent, var):
        group = ttk.Frame(parent, style="Mint.TFrame")
        label = ttk.Label(group, background=MINT, style="Help.TLabel")
        label.grid(row=0, column=0, padx=(0, 4), sticky="w")
        entry = ttk.Spinbox(group, textvariable=var, width=6, from_=-1e15, to=1e15, increment=1)
        entry.grid(row=0, column=1, sticky="w", pady=2)
        entry.bind("<Return>", lambda _event: self.update_plot())
        error = ttk.Label(parent, style="Error.TLabel", background=MINT, wraplength=115)
        return group, label, error

    def _schedule_update(self, *_):
        if self._update_id is None:
            self._update_id = self.root.after(55, self.update_plot)

    def _data_changed(self, _event):
        if self.data_text.edit_modified():
            self.data_text.edit_modified(False)
            if self.mode.get() == "Empírica":
                self._schedule_update()

    def _distribution_changed(self, _event=None):
        self._dragged_limit = None
        self._zoom_factor = 1.0
        self._rebuild_parameters()
        self.update_plot()

    def _rebuild_parameters(self):
        for child in self.parameter_box.winfo_children():
            child.destroy()
        self.parameter_vars.clear()
        self.parameter_errors.clear()
        self.parameter_sliders.clear()
        mode = self.mode.get()
        self.data_box.pack_forget()
        if mode == "Empírica":
            self.dist_controls.pack_forget()
            self.data_box.pack(fill="x")
            self.graph_mode.set("Acumulada")
        else:
            self.dist_controls.pack(fill="x")
            registry = CONTINUOUS if mode == "Continuas" else DISCRETE
            self.dist_combo.configure(values=tuple(registry))
            if self.distribution.get() not in registry:
                default = "Exponencial" if mode == "Continuas" else "Binomial"
                self.distribution.set(default)
            _, defaults, labels = registry[self.distribution.get()]
            for key, value in defaults.items():
                ttk.Label(self.parameter_box, text=labels[key], wraplength=205).pack(anchor="w", pady=(5, 3))
                var = tk.StringVar(value=format_number(value))
                var.trace_add("write", self._schedule_update)
                self.parameter_vars[key] = var
                if self.distribution.get() == "Normal":
                    entry = ttk.Spinbox(self.parameter_box, textvariable=var, width=18,
                                        from_=0.1 if key == "sigma" else -1e15,
                                        to=1e15, increment=0.1)
                else:
                    entry = ttk.Entry(self.parameter_box, textvariable=var, width=18)
                entry.pack(fill="x")
                entry.bind("<Return>", lambda _event: self.update_plot())
                bounds = {"lambda_param": (.05, 2 if mode == "Continuas" else 20),
                          "sigma": (.1, 5), "p": (.01, 1)}.get(key)
                if bounds and self.distribution.get() != "Normal":
                    self._add_slider(key, var, value, bounds)
                error = ttk.Label(self.parameter_box, style="Error.TLabel", wraplength=205)
                error.pack(anchor="w")
                self.parameter_errors[key] = error
        if self.graph_buttons:
            density_button = self.graph_buttons["Densidad"]
            density_button.configure(text="Masa" if mode == "Discretas" else "Densidad")
            density_button.state(["disabled" if mode == "Empírica" else "!disabled"])
        if mode in ("Discretas", "Empírica"):
            self.exact_button.state(["!disabled"])
            self.exact_button.grid()
        else:
            self.exact_button.grid_remove()
            self.exact_button.state(["disabled"])
            if self.prob_kind.get() == "exacta":
                self.prob_kind.set("hasta")
        self._prob_controls()

    def _add_slider(self, key, var, value, bounds):
        slider = ttk.Scale(self.parameter_box, from_=bounds[0], to=bounds[1], value=value)
        slider.pack(fill="x", pady=(6, 2))
        self.parameter_sliders[key] = slider

        def slide(raw_value):
            new_value = format_number(float(raw_value), 3)
            if var.get() != new_value:
                var.set(new_value)

        slider.configure(command=slide)

    def _probability_choice(self):
        self._prob_controls()
        self.update_plot()

    def _prob_controls(self):
        kind = self.prob_kind.get()
        self.first_label.configure(text={"entre": "", "exacta": "k", "menos": "X <", "al_menos": "X ≥"}.get(kind, "x"))
        self._arrange_inputs()

    def reset_values(self):
        self._zoom_factor = 1.0
        normal = self.mode.get() == "Continuas" and self.distribution.get() == "Normal"
        self.first_var.set("-1" if normal else "5")
        self.second_var.set("1" if normal else "10")
        self.prob_kind.set("entre" if normal else "hasta")
        self.graph_mode.set("Acumulada" if self.mode.get() == "Empírica" else "Densidad")
        if self.mode.get() == "Empírica":
            self.data_text.delete("1.0", "end")
            self.data_text.insert("1.0", self.sample_text)
        self._rebuild_parameters()
        self.update_plot()

    def zoom_graph(self, factor=None):
        self._zoom_factor = 1.0 if factor is None else min(16, max(.05, self._zoom_factor * factor))
        self.update_plot()

    def set_animation(self, enabled):
        self.animate = enabled
        if not enabled and self._animation_id is not None:
            self.update_plot()

    def _begin_drag(self, event):
        if event.button != 1 or event.inaxes is not self.ax or not self._plot_limits:
            return
        nearest = min(self._plot_limits, key=lambda field: abs(event.x - self._limit_pixel(field)))
        if abs(event.x - self._limit_pixel(nearest)) <= 12:
            self._dragged_limit = nearest
            self._hide_coordinates()

    def _limit_pixel(self, field):
        return self.ax.get_xaxis_transform().transform((self._plot_limits[field], 0))[0]

    def _end_drag(self, _event):
        if self._dragged_limit is not None:
            self.update_plot()
            self._dragged_limit = None

    def _hide_coordinates(self, _event=None):
        if self._plot_data is not None and self._tooltip.get_visible():
            self._tooltip.set_visible(False)
            self.canvas.draw_idle()

    def _plot_motion(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self._dragged_limit:
            self._move_limit(event.xdata)
        elif self._plot_data is not None:
            self._show_coordinates(event.xdata)

    def _move_limit(self, x):
        field = self._dragged_limit
        if self.mode.get() == "Discretas":
            x = round(x)
        if self.prob_kind.get() == "entre":
            other = self.second_var if field == "first" else self.first_var
            try:
                boundary = read_number(other.get())
            except ValueError:
                return
            if field == "first":
                x = min(x, math.floor(boundary) if self.mode.get() == "Discretas" else boundary)
            else:
                x = max(x, math.ceil(boundary) if self.mode.get() == "Discretas" else boundary)
        var = self.first_var if field == "first" else self.second_var
        value = format_number(x, 10)
        if value != var.get():
            var.set(value)

    def _show_coordinates(self, x):
        xs, ys, style = self._plot_data
        if style == "escalón":
            index = max(0, np.searchsorted(xs, x, side="right") - 1)
            y = ys[index]
        elif style == "barras":
            index = np.abs(xs - x).argmin()
            x, y = xs[index], ys[index]
        else:
            y = np.interp(x, xs, ys)
        if not np.isfinite(y):
            self._hide_coordinates()
            return
        self._tooltip.xy = (x, y)
        self._tooltip.set_text(f"x = {format_number(x, 4)}\ny = {format_number(y, 4)}")
        self._tooltip.set_position((-10, 12) if x > np.mean(self.ax.get_xlim()) else (10, 12))
        self._tooltip.set_ha("right" if x > np.mean(self.ax.get_xlim()) else "left")
        self._tooltip.set_visible(True)
        self.canvas.draw_idle()

    def _parse_empirical(self):
        return parse_observations(self.data_text.get("1.0", "end"))

    def generator_selection(self):
        if self.mode.get() == "Empírica":
            return "Empírica", {}, self.data_text.get("1.0", "end-1c")
        return self.distribution.get(), {key: var.get() for key, var in self.parameter_vars.items()}, ""

    def session_settings(self):
        name, parameters, _ = self.generator_selection()
        return {"mode": self.mode.get(), "name": name, "parameters": parameters,
                "observations": self.data_text.get("1.0", "end-1c"), "probability": self.prob_kind.get(),
                "graph": self.graph_mode.get(), "first": self.first_var.get(),
                "second": self.second_var.get()}

    def restore_session(self, settings):
        self._zoom_factor = 1.0
        self.mode.set(settings["mode"])
        if settings["mode"] != "Empírica":
            self.distribution.set(settings["name"])
        self.prob_kind.set(settings["probability"])
        self.graph_mode.set(settings["graph"])
        self.first_var.set(settings["first"])
        self.second_var.set(settings["second"])
        self._rebuild_parameters()
        self.data_text.delete("1.0", "end")
        self.data_text.insert("1.0", settings["observations"])
        self.data_text.edit_reset()
        if settings["mode"] != "Empírica":
            for key, value in settings["parameters"].items():
                self.parameter_vars[key].set(value)
        self.update_plot()

    def _create_distribution(self):
        registry = CONTINUOUS if self.mode.get() == "Continuas" else DISCRETE
        cls, _, _ = registry[self.distribution.get()]
        params = {}
        for key, var in self.parameter_vars.items():
            value = parse_parameter(self.distribution.get(), key, var.get())
            slider = self.parameter_sliders.get(key)
            if slider is not None and slider.get() != value:
                # Mute the slider callback so typed values outside its range stay intact.
                callback = slider.cget("command")
                slider.configure(command="")
                slider.set(value)
                slider.configure(command=callback)
            params[key] = value
        return cls(**params)

    def select_model(self, name, parameters, *, discrete, event=None):
        """Load the exact fitted parameters into the probability explorer."""
        self.mode.set("Discretas" if discrete else "Continuas")
        self.distribution.set(name)
        if event is not None:
            kind, first, second = event
            self.prob_kind.set(kind)
            self.first_var.set(
                str(int(round(first))) if discrete and float(first).is_integer()
                else format_number(float(first), 12)
            )
            if second is not None:
                self.second_var.set(
                    str(int(round(second))) if discrete and float(second).is_integer()
                    else format_number(float(second), 12)
                )
        self._rebuild_parameters()
        for key, value in parameters.items():
            is_int = (
                (key in INTEGER_PARAMETERS or (name == "Chi-cuadrado" and key == "df"))
                and float(value).is_integer()
            )
            self.parameter_vars[key].set(
                str(int(round(value))) if is_int else format_number(float(value), 12)
            )
        self.update_plot()

    def _window(self, dist, limits):
        support_lo, support_hi = dist.get_support()
        if dist.is_discrete:
            q001 = float(dist.ppf(.001))
            q999 = float(dist.ppf(.999))
            q_lo = float(support_lo) if np.isfinite(support_lo) and q001 - support_lo <= 40 else q001
            q_hi = float(min(support_hi, q999))
            lo = max(support_lo, min(q_lo, *(math.floor(v) for v in limits)))
            hi = min(support_hi, max(q_hi, *(math.ceil(v) for v in limits)))
            if hi - lo > 300:
                center = round(float(dist.ppf(.5)))
                lo = max(support_lo, center - 150)
                hi = min(support_hi, lo + 300)
                lo = max(support_lo, hi - 300)
            if hi < lo + 5 and np.isfinite(lo):
                hi = min(support_hi, lo + 6)
            return np.arange(math.ceil(lo), math.floor(hi) + 1)

        q05, q95 = float(dist.ppf(.05)), float(dist.ppf(.95))
        q001, q999 = float(dist.ppf(.001)), float(dist.ppf(.999))
        heavy_tail = (q999 - q001) > 12 * max(q95 - q05, 1e-9)
        low_q, high_q = (.01, .99) if heavy_tail else (.001, .999)
        lo = support_lo if np.isfinite(support_lo) else float(dist.ppf(low_q))
        hi = support_hi if np.isfinite(support_hi) else float(dist.ppf(high_q))
        if isinstance(dist, NormalDistribution):
            lo, hi = dist.mean() - 4 * dist.std(), dist.mean() + 4 * dist.std()
        if not np.isfinite(lo):
            lo = -10.0
        if not np.isfinite(hi):
            hi = 10.0
        if hi <= lo:
            hi = lo + max(1.0, abs(lo) * .001)
        span = max(hi - lo, 1e-6)
        if not np.isfinite(support_lo):
            for lim in limits:
                if lo - 1.5 * span <= lim < lo:
                    lo = lim - 0.05 * span
        if not np.isfinite(support_hi):
            for lim in limits:
                if hi < lim <= hi + 1.5 * span:
                    hi = lim + 0.05 * span
        grid = np.linspace(lo, hi, 600)
        return np.unique(np.r_[grid, [limit for limit in limits if lo <= limit <= hi]])

    def update_plot(self):
        self.figure.set_facecolor(NIGHT.panel if is_night(self.root) else SURFACE)
        if self._update_id is not None:
            self.root.after_cancel(self._update_id)
            self._update_id = None
        for label in (self.first_error, self.second_error, self.prob_error,
                      *self.parameter_errors.values()):
            label.configure(text="")
        mode, kind = self.mode.get(), self.prob_kind.get()
        for var, error in [(self.first_var, self.first_error)] + (
            [(self.second_var, self.second_error)] if kind == "entre" else []
        ):
            try:
                read_number(var.get())
            except ValueError as exc:
                error.configure(text=str(exc))
                self._clear_result()
                return
        first = read_number(self.first_var.get())
        second = read_number(self.second_var.get()) if kind == "entre" else None
        limits = (first,) if second is None else (first, second)
        if second is not None and first > second:
            self.second_error.configure(text="Debe ser mayor o igual que a.")
            self._clear_result()
            return
        if kind == "exacta" and mode == "Discretas" and not first.is_integer():
            self.first_error.configure(text="Ingresa un número entero.")
            self._clear_result()
            return
        try:
            if mode == "Empírica":
                ordered = np.sort(self._parse_empirical())
                result = self._empirical_probability(ordered, kind, first, second)
                x, counts = np.unique(ordered, return_counts=True)
                padding = max((x[-1] - x[0]) * .06, 0.1)
                x = np.r_[x[0] - padding, x, x[-1] + padding]
                y = np.r_[0, np.cumsum(counts) / ordered.size, 1]
                if self.show_moments:
                    self._show_stats(np.mean(ordered), np.var(ordered), np.std(ordered))
                self._draw(x, y, "escalón", "Fₙ(x)", limits, kind, cumulative=True)
            else:
                dist = self._create_distribution()
                result = probability(dist, kind, first, second)
                x = self._window(dist, limits)
                cumulative = self.graph_mode.get() == "Acumulada"
                if cumulative and dist.is_discrete:
                    x = np.r_[x[0] - 1, x, x[-1] + 1]
                y = dist.cdf(x) if cumulative else dist.pdf(x)
                if cumulative:
                    style = "escalón" if dist.is_discrete else "curva"
                    ylabel = "Acumulada"
                else:
                    style = "barras" if dist.is_discrete else "curva"
                    ylabel = "Masa" if dist.is_discrete else "Densidad"
                if self.show_moments:
                    self._show_stats(dist.mean(), dist.var(), dist.std())
                self._draw(x, y, style, ylabel, limits, kind, cumulative=cumulative)
        except ParameterInputError as exc:
            self._clear_result()
            self.parameter_errors[exc.key].configure(text=str(exc))
            return
        except (ValueError, TypeError, OverflowError, IndexError) as exc:
            self._clear_result()
            raw_message = str(exc)
            translated = _translate_error(raw_message)
            matched = False
            for key, label in self.parameter_errors.items():
                alias = "lambda" if key == "lambda_param" else key
                if re.search(rf"\b(?:{re.escape(key)}|{re.escape(alias)})\b", raw_message):
                    label.configure(text=translated)
                    matched = True
            if not matched:
                self.prob_error.configure(text=translated)
            return

        a = format_number(first)
        formula = {
            "hasta": f"P(X ≤ {a})",
            "menos": f"P(X < {a})",
            "encima": f"P(X > {a})",
            "al_menos": f"P(X ≥ {a})",
            "exacta": f"P(X = {a})",
        }.get(kind)
        if second is not None:
            formula = f"P({a} ≤ X ≤ {format_number(second)})"
        pct_text, dec_text = format_probability(result)
        self.result.configure(text=pct_text)
        self.decimal_result.configure(text=dec_text)
        self.formula.configure(text=formula)
        if hasattr(self, "surface"):
            self.surface.draw()

    def _clear_result(self):
        if hasattr(self, "surface"):
            self.surface.close()
        self.result.configure(text="—")
        self.decimal_result.configure(text="")
        self.formula.configure(text="")
        if self._animation_id is not None:
            self.root.after_cancel(self._animation_id)
            self._animation_id = None
        self._plot_data = None
        self._plot_limits.clear()
        for label in self.stats_labels:
            label.configure(text="—")
        self.ax.clear()
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def _show_stats(self, mean, variance, deviation):
        for label, value in zip(self.stats_labels, (mean, variance, deviation)):
            label.configure(text=format_number(value, 5) if np.isfinite(value) else "No definida")

    @staticmethod
    def _empirical_probability(values, kind, first, second):
        if kind == "hasta":
            return float(np.mean(values <= first))
        if kind == "menos":
            return float(np.mean(values < first))
        if kind == "encima":
            return float(np.mean(values > first))
        if kind == "al_menos":
            return float(np.mean(values >= first))
        if kind == "exacta":
            return float(np.mean(values == first))
        if kind == "entre":
            if second is None or first > second:
                raise ValueError("El límite inferior debe ser menor o igual al superior.")
            return float(np.mean((values >= first) & (values <= second)))
        raise ValueError("Tipo de probabilidad no válido.")

    def _draw(self, x, y, style, ylabel, limits, kind, *, cumulative):
        if self._animation_id is not None:
            self.root.after_cancel(self._animation_id)
            self._animation_id = None
        self.ax.clear()
        night = is_night(self.root)
        canvas_color = NIGHT.plot if night else SURFACE
        curve_color = NIGHT.focus if night else TEAL
        fill_color = NIGHT.shade if night else "#a7e1d8"
        text_color = NIGHT.text if night else TEXT
        self.ax.set_axis_on()
        self.ax.set_facecolor(canvas_color)
        self.ax.grid(axis="y", color=NIGHT.grid if night else BORDER, linewidth=.8)
        self.ax.set_axisbelow(True)
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            self.ax.spines[spine].set_color(NIGHT.border if night else "#9fb7be")
        self.ax.tick_params(colors=NIGHT.muted if night else MUTED, labelsize=9)
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_number(value, 4)))
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_number(value, 4)))
        first, *upper = limits
        second = upper[0] if upper else None
        finite_y = np.where(np.isfinite(y), y, np.nan)
        if style == "escalón":
            (line,) = self.ax.step(x, finite_y, where="post", color=curve_color, linewidth=2.4)
        elif style == "curva":
            (line,) = self.ax.plot(x, finite_y, color=curve_color, linewidth=2.4)
        else:
            line = None
            outside = NIGHT.inactive_bar if night else "#d2e1e5"
            colors = [curve_color if self._inside_probability(int(v), kind, first, second)
                      else outside for v in x]
            self.ax.bar(x, finite_y, width=.8, color=colors)
        if cumulative and kind != "exacta":
            lo = x[0] if kind in ("hasta", "menos") else max(x[0], first)
            hi = x[-1] if kind in ("encima", "al_menos") else min(x[-1], first if second is None else second)
            if hi > lo:
                self.ax.axvspan(lo, hi, color=fill_color, alpha=.46, linewidth=0)
        elif style == "curva" and not cumulative:
            self.ax.fill_between(x, 0, finite_y,
                                 where=self._continuous_mask(x, kind, first, second),
                                 color=fill_color, alpha=.58, linewidth=0)
        padding = .55 if style == "barras" else max((x[-1] - x[0]) * .015, 1e-12)
        if x[0] == x[-1]:
            padding = .5
        midpoint = (x[0] + x[-1]) / 2
        half_span = ((x[-1] - x[0]) / 2 + padding) * self._zoom_factor
        self.ax.set_xlim(midpoint - half_span, midpoint + half_span)
        if cumulative:
            top = 1.07
        else:
            valid_y = finite_y[np.isfinite(finite_y)]
            if valid_y.size == 0:
                top = 1.0
            elif style == "curva" and not np.all(np.isfinite(y)) and valid_y.size > 10:
                top = max(1e-12, float(np.percentile(valid_y, 98))) * 1.15
            else:
                top = max(1e-12, float(np.max(valid_y))) * 1.12
        self.ax.set_ylim(0, top)
        self.ax.set_xlabel("x", color=text_color)
        self.ax.set_ylabel("f(x)" if ylabel == "Densidad" else ylabel, color=text_color)
        self._plot_limits.clear()
        left, right = self.ax.get_xlim()
        for field, limit in zip(("first", "second"), limits):
            if not left <= limit <= right:
                continue
            marker = limit
            self._plot_limits[field] = marker
            height = np.interp(marker, x, finite_y) if style == "curva" and not cumulative else top * .96
            self.ax.vlines(marker, 0, height, color=curve_color, linewidth=1.35, zorder=4)
            self.ax.plot(marker, 0, "o", clip_on=False, transform=self.ax.get_xaxis_transform(),
                         markersize=7, markerfacecolor=canvas_color, markeredgecolor=curve_color,
                         markeredgewidth=1.5, zorder=5)
        self._plot_data = (x, finite_y, style)
        self._tooltip = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 12), textcoords="offset points",
            color=text_color, fontsize=9,
            bbox={"facecolor": NIGHT.panel if night else SURFACE,
                  "edgecolor": NIGHT.border if night else BORDER,
                  "boxstyle": "round,pad=.45"}, zorder=8,
        )
        self._tooltip.set_visible(False)
        plot_style = (self.mode.get(), self.distribution.get(), style, cumulative)
        if self.animate and line is not None and plot_style != self._last_plot_style:
            self._fade_line(line, 1)
        self._last_plot_style = plot_style
        self.canvas.draw_idle()

    def _fade_line(self, line, frame):
        line.set_alpha(frame / 5)
        self.canvas.draw_idle()
        self._animation_id = self.root.after(25, self._fade_line, line, frame + 1) if frame < 5 else None

    @staticmethod
    def _inside_probability(k, kind, first, second):
        if kind == "exacta":
            return float(k) == first
        if kind == "hasta":
            return k <= math.floor(first)
        if kind == "menos":
            return k < first
        if kind == "encima":
            return k > math.floor(first)
        if kind == "al_menos":
            return k >= math.ceil(first)
        return math.ceil(first) <= k <= math.floor(second)

    @staticmethod
    def _continuous_mask(x, kind, first, second):
        if kind in ("hasta", "menos"):
            return x <= first
        if kind in ("encima", "al_menos"):
            return x >= first
        if kind == "entre":
            return (x >= first) & (x <= second)
        return np.zeros_like(x, dtype=bool)


def main():
    from fitec_tools import FitecTools

    root = tk.Tk()
    FitecTools(root, FitecApp, parse_parameter, parse_observations)
    root.mainloop()


if __name__ == "__main__":
    main()
