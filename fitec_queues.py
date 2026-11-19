"""Simulate and compare two single-server queues with shared arrivals."""

from dataclasses import asdict, dataclass
import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy import stats

from fitec_analysis import MODELS
from fitec_theme import NIGHT, is_night


WARMUP = 500
MAX_CUSTOMERS = 100_000
FRAME_COUNT = 180
FRAME_INTERVAL = 65
QUEUE_DEFAULTS = {"arrival_mean": "15", "service_mean": "10", "count": "10000",
                  "seed": "42", "fit": None}
INK = "#103746"
TEAL = "#087d77"
BLUE = "#387caa"
PAPER = "#edf3f5"
WHITE = "#ffffff"


def service_duration_mean(model):
    with np.errstate(over="ignore", invalid="ignore"):
        lower, _ = model.support()
        mean = float(model.mean())
    if not lower >= 0:
        raise ValueError("El modelo admite tiempos negativos. Elige una distribución de atención con soporte no negativo.")
    if not 0 < mean < math.inf:
        raise ValueError("La atención debe tener una media positiva y finita.")
    return mean


@dataclass(frozen=True)
class ServiceFit:
    model: str
    parameters: tuple[float, ...]
    summary: str

    def frozen(self):
        return MODELS[self.model].distribution(*self.parameters)

    @classmethod
    def from_document(cls, document):
        if document is None:
            return None
        if not isinstance(document, dict):
            raise ValueError("El ajuste guardado para la simulación no es válido.")
        name = document.get("model")
        model = MODELS.get(name) if isinstance(name, str) else None
        parameters = document.get("parameters")
        summary = document.get("summary")
        if model is None or not isinstance(parameters, (list, tuple)) or not isinstance(summary, str):
            raise ValueError("El ajuste guardado para la simulación no es válido.")
        expected = model.distribution.numargs + (0 if model.discrete else 2)
        try:
            valid_parameters = len(parameters) == expected and all(
                type(value) in (float, int) and math.isfinite(value) for value in parameters
            )
        except OverflowError:
            valid_parameters = False
        if not valid_parameters:
            raise ValueError("Los parámetros guardados para la atención no son válidos.")
        result = cls(model.key, tuple(parameters), summary)
        service_duration_mean(result.frozen())
        return result


def validate_queue_settings(settings):
    if settings is None:
        return QUEUE_DEFAULTS.copy()
    if not isinstance(settings, dict) or any(
        not isinstance(settings.get(key), str) for key in ("arrival_mean", "service_mean", "count", "seed")
    ):
        raise ValueError("La configuración de la simulación no es válida.")
    ServiceFit.from_document(settings.get("fit"))
    return settings


@dataclass(frozen=True)
class QueueState:
    waiting: int
    finished: int
    progress: float | None


@dataclass(frozen=True)
class QueueRun:
    arrivals: np.ndarray
    starts: np.ndarray
    departures: np.ndarray
    simulated_wait: float
    theoretical_wait: float | None
    load: float

    def at(self, minute):
        arrived = int(np.searchsorted(self.arrivals, minute, side="right"))
        started = int(np.searchsorted(self.starts, minute, side="right"))
        finished = int(np.searchsorted(self.departures, minute, side="right"))
        progress = None
        if started > finished:
            first, last = self.starts[finished], self.departures[finished]
            progress = float((minute - first) / (last - first))
        return QueueState(arrived - started, finished, progress)


def _simulate(arrivals, arrival_mean, model, random):
    with np.errstate(over="ignore", invalid="ignore"):
        mean, variance = (float(value) for value in model.stats(moments="mv"))
        durations = model.rvs(size=len(arrivals), random_state=random)
    if not 0 < mean < math.inf:
        raise ValueError("La atención debe tener una media positiva y finita.")
    if not np.all(np.isfinite(durations) & (durations >= 0)):
        raise ValueError("La distribución generó duraciones que no se pueden simular.")
    starts = np.empty_like(arrivals)
    departures = np.empty_like(arrivals)
    free = 0.0
    for index, (arrival, duration) in enumerate(zip(arrivals, durations)):
        start = max(float(arrival), free)
        free = start + float(duration)
        starts[index], departures[index] = start, free
    if not math.isfinite(free):
        raise ValueError("Los tiempos son demasiado grandes para simular la fila.")

    waits = starts[WARMUP:] - arrivals[WARMUP:]
    simulated = math.fsum(waits / len(waits))
    load = mean / arrival_mean
    theoretical = None
    if load < 1 and math.isfinite(variance):
        # Pollaczek–Khinchin for the queueing wait in a stable M/G/1 system.
        expected = load * (mean + variance / mean) / (2 * (1 - load))
        if math.isfinite(expected):
            theoretical = expected
    return QueueRun(arrivals, starts, departures, simulated, theoretical, load)


def simulate_queues(arrival_mean, service_mean, count, seed, *, fitted_service=None):
    """Use the same Poisson arrivals and the same mean service time in both queues."""
    if not 0 < arrival_mean < math.inf:
        raise ValueError("El intervalo medio entre llegadas debe ser positivo y finito.")
    if type(count) is not int or not 1 <= count <= MAX_CUSTOMERS:
        raise ValueError("Usa de 1 a 100 000 clientes medidos.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("La semilla debe ser un entero entre 0 y 4 294 967 295.")
    if fitted_service is not None:
        service_mean = service_duration_mean(fitted_service)
    elif not 0 < service_mean < math.inf:
        raise ValueError("La atención debe tener una media positiva y finita.")

    arrival_seed, erlang_seed, second_seed = np.random.SeedSequence(seed).spawn(3)
    random = np.random.default_rng(arrival_seed)
    with np.errstate(over="ignore"):
        arrivals = np.cumsum(random.exponential(arrival_mean, size=WARMUP + count))
    if not math.isfinite(float(arrivals[-1])):
        raise ValueError("Los intervalos de llegada son demasiado grandes para simularlos.")

    regular = stats.gamma(a=4, scale=service_mean / 4)
    variable = fitted_service if fitted_service is not None else stats.expon(scale=service_mean)
    return (_simulate(arrivals, arrival_mean, regular, np.random.default_rng(erlang_seed)),
            _simulate(arrivals, arrival_mean, variable, np.random.default_rng(second_seed)))


class QueueLane:
    def __init__(self, parent, title, accent):
        self.frame = ttk.Frame(parent, style="Panel.TFrame", padding=(15, 9))
        self.frame.pack(fill="both", expand=True, pady=(0, 10))
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        self.title = ttk.Label(self.frame, text=title, style="Section.TLabel")
        self.title.grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(self.frame, height=66, background=WHITE, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=3)
        self.accent = accent
        self.state = QueueState(0, 0, None)
        self.canvas.bind("<Configure>", lambda _event: self._draw())
        self.caption = ttk.Label(self.frame, style="Help.TLabel")
        self.caption.grid(row=2, column=0, sticky="w")
        stats_frame = ttk.Frame(self.frame, style="Panel.TFrame", padding=(13, 0))
        stats_frame.grid(row=0, column=1, rowspan=3, sticky="nsw", padx=(11, 0))
        ttk.Label(stats_frame, text="Espera media simulada", style="Help.TLabel").pack(anchor="w")
        self.average = ttk.Label(stats_frame, text="—", style="Help.TLabel",
                                 foreground=INK, font=("Inter", 16, "bold"))
        self.average.pack(anchor="w", pady=(1, 3))
        self.reference = ttk.Label(stats_frame, text="Teórica · —", style="Help.TLabel")
        self.reference.pack(anchor="w")
        self.utilization = ttk.Label(stats_frame, text="Carga · —", style="Help.TLabel")
        self.utilization.pack(anchor="w", pady=(2, 0))
        self.set_state(self.state)

    def set_run(self, run=None):
        if run is None:
            self.average.configure(text="—")
            self.reference.configure(text="Teórica · —")
            self.utilization.configure(text="Carga · —")
            self.set_state(QueueState(0, 0, None))
            return
        self.average.configure(text=f"{run.simulated_wait:.4g} min".replace(".", ","))
        if run.theoretical_wait is None:
            ref = "Sin régimen estable" if run.load >= 1 else "No definida"
        else:
            ref = f"{run.theoretical_wait:.4g} min".replace(".", ",")
        self.reference.configure(text=f"Teórica · {ref}")
        load = f"{100 * run.load:.3g} %".replace(".", ",") if math.isfinite(run.load) else "Fuera de rango"
        self.utilization.configure(text=f"Carga · {load}")

    def set_state(self, state, completed_before_preview=0):
        self.state = state
        completed = state.finished - completed_before_preview
        self.caption.configure(text=f"{state.waiting} en fila   ·   {completed} atendidos en la vista")
        self._draw()

    def _draw(self):
        scene = self.canvas
        width, height = scene.winfo_width(), scene.winfo_height()
        if width < 170 or height < 25:
            return
        scene.delete("all")
        night = is_night(scene)
        accent = (NIGHT.focus if self.accent == TEAL else "#8fbfff") if night else self.accent
        text = NIGHT.text if night else INK
        mid = height / 2
        counter_left, counter_right = width - 112, width - 5
        scene.create_line(10, mid, counter_left - 12, mid, fill=NIGHT.border if night else "#c9dbdf",
                          dash=(4, 4), arrow=tk.LAST)
        last_slot = counter_left - 34
        slots = max(0, min(12, int((last_slot - 24) // 24) + 1))
        for index in range(min(self.state.waiting, slots)):
            x = last_slot - 24 * index
            scene.create_oval(x - 8, mid - 8, x + 8, mid + 8, fill=accent, outline="")
        if self.state.waiting > slots:
            scene.create_text(10, mid - 15, text=f"+{self.state.waiting - slots}",
                              fill=text, anchor="w", font=("Inter", 9, "bold"))

        progress = self.state.progress
        idle_fill = NIGHT.input if night else PAPER
        busy_fill = NIGHT.selection if night else "#dff3ef"
        scene.create_rectangle(counter_left, mid - 18, counter_right, mid + 18,
                               fill=busy_fill if progress is not None else idle_fill, outline="")
        if progress is not None:
            edge = counter_left + (counter_right - counter_left) * progress
            scene.create_rectangle(counter_left, mid + 14, edge, mid + 18,
                                   fill=accent, outline="")
        scene.create_text((counter_left + counter_right) / 2, mid - 2,
                          text="Atendiendo" if progress is not None else "Ventanilla libre",
                          fill=text, font=("Inter", 9, "bold"))


class QueueSimulator:
    def __init__(self, parent):
        self.parent = parent
        self.fit = None
        self.runs = None
        self.timer = None
        self.vars = {key: tk.StringVar(parent, value=value)
                     for key, value in QUEUE_DEFAULTS.items() if key != "fit"}
        self.frame = ttk.Frame(parent, padding=(14, 10, 14, 7))
        self.frame.pack(fill="both", expand=True)
        controls = ttk.Frame(self.frame, style="Panel.TFrame", padding=(15, 11))
        controls.pack(fill="x", pady=(0, 10))
        for column, (key, label) in enumerate((
            ("arrival_mean", "Entre llegadas (min)"), ("service_mean", "Atención media (min)"),
            ("count", "Clientes medidos"), ("seed", "Semilla"),
        )):
            controls.columnconfigure(column, weight=1)
            field = ttk.Frame(controls, style="Panel.TFrame")
            field.grid(row=0, column=column, sticky="ew", padx=(0, 10) if column < 3 else 0)
            ttk.Label(field, text=label, style="Help.TLabel").pack(anchor="w", pady=(0, 4))
            entry = ttk.Entry(field, textvariable=self.vars[key], width=10)
            entry.pack(fill="x")
            if key == "service_mean":
                self.service_entry = entry
        self.error = ttk.Label(controls, style="Error.TLabel", wraplength=570)
        self.error.grid(row=1, column=0, columnspan=4, sticky="w")
        self.fit_note = ttk.Label(controls, style="Help.TLabel", wraplength=570)
        self.fit_note.grid(row=2, column=0, columnspan=4, sticky="w")

        actions = ttk.Frame(self.frame, padding=(2, 0, 2, 8))
        actions.pack(fill="x")
        self.start_button = ttk.Button(actions, text="▶ Simular", style="Primary.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.pause_button = ttk.Button(actions, text="Pausar", style="Toolbar.TButton", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=6)
        self.reset_fit_button = ttk.Button(actions, text="Usar ejemplo", style="Toolbar.TButton",
                                           command=lambda: self.restore_session(None))
        self.reset_fit_button.pack(side="left")
        self.clock = ttk.Label(actions, background=PAPER, style="Help.TLabel")
        self.clock.pack(side="right", padx=8)
        self.lanes = (QueueLane(self.frame, "Erlang · 4 etapas", TEAL),
                      QueueLane(self.frame, "Exponencial", BLUE))
        self.note = ttk.Label(self.frame, background=PAPER, style="Help.TLabel", wraplength=640)
        self.note.pack(fill="x", padx=2)
        for var in self.vars.values():
            var.trace_add("write", self._reset)
        parent.bind("<Destroy>", self._destroy, add="+")
        self._reset()

    def repaint_theme(self):
        for lane in self.lanes:
            lane._draw()

    def _cancel_tick(self):
        if self.timer is not None:
            self.parent.after_cancel(self.timer)
            self.timer = None

    def _destroy(self, event):
        if event.widget is self.parent:
            self._cancel_tick()

    def _reset(self, *_):
        self._cancel_tick()
        self.runs = None
        self.pause_button.configure(text="Pausar")
        self.pause_button.state(["disabled"])
        self.clock.configure(text="Minuto · —")
        self.error.configure(text="")
        self.note.configure(text="Dos filas con las mismas llegadas exponenciales y una ventanilla por fila.")
        for lane in self.lanes:
            lane.set_run()

    def load_fit(self, fit, summary):
        fitted = ServiceFit(fit.model, fit.parameters, summary)
        try:
            mean = service_duration_mean(fitted.frozen())
        except ValueError as exc:
            raise ValueError(f"{fit.candidate.display_name} · {exc}") from exc
        self._set_fit(fitted)
        self.vars["service_mean"].set(f"{mean:.8g}".replace(".", ","))

    def _set_fit(self, fit):
        self.fit = fit
        if fit is None:
            self.service_entry.state(["!readonly"])
            self.lanes[1].title.configure(text="Exponencial")
            self.fit_note.configure(text="")
        else:
            model = MODELS[fit.model]
            self.service_entry.state(["readonly"])
            self.lanes[1].title.configure(text=f"{model.display_name} ajustada")
            self.fit_note.configure(text=f"Atención ajustada en minutos · {model.parameter_summary(fit.parameters)}")
        self._reset()

    def session_settings(self):
        return {**{key: var.get() for key, var in self.vars.items()},
                "fit": asdict(self.fit) if self.fit is not None else None}

    def restore_session(self, settings):
        settings = QUEUE_DEFAULTS if settings is None else settings
        self._set_fit(ServiceFit.from_document(settings.get("fit")))
        for key, var in self.vars.items():
            var.set(settings[key])

    def start(self):
        self._reset()
        try:
            arrival_mean = float(self.vars["arrival_mean"].get().replace(",", "."))
            service_mean = float(self.vars["service_mean"].get().replace(",", "."))
            count = int(self.vars["count"].get())
            seed = int(self.vars["seed"].get())
        except ValueError:
            self.error.configure(text="Ingresa tiempos numéricos, una cantidad entera y una semilla entera.")
            return
        try:
            fitted = self.fit.frozen() if self.fit is not None else None
            runs = simulate_queues(arrival_mean, service_mean, count, seed, fitted_service=fitted)
        except (ValueError, OverflowError) as exc:
            self.error.configure(text=str(exc))
            return

        self.runs = runs
        self.preview_start = runs[0].arrivals[WARMUP - 1]
        self.preview_length = min(24 * arrival_mean,
                                  max(run.departures[-1] for run in runs) - self.preview_start)
        self.completed_before = tuple(run.at(self.preview_start).finished for run in runs)
        self.frame_index = 0
        self.pause_button.state(["!disabled"])
        count_label = f"{count:,}".replace(",", ".")
        self.note.configure(text=f"Espera simulada con {count_label} clientes después de {WARMUP} de calentamiento. "
                                 "La animación muestra el inicio del período medido.")
        for lane, run in zip(self.lanes, runs):
            lane.set_run(run)
        self._tick()

    def _tick(self):
        self.timer = None
        elapsed = self.preview_length * (self.frame_index / FRAME_COUNT)
        self.clock.configure(text=f"Minuto · {elapsed:.4g}".replace(".", ","))
        for lane, run, starting_count in zip(self.lanes, self.runs, self.completed_before):
            lane.set_state(run.at(self.preview_start + elapsed), starting_count)
        if self.frame_index < FRAME_COUNT:
            self.frame_index += 1
            self.timer = self.parent.after(FRAME_INTERVAL, self._tick)
        else:
            self.pause_button.state(["disabled"])

    def pause(self):
        if self.timer is not None:
            self._cancel_tick()
            self.pause_button.configure(text="Continuar")

    def toggle_pause(self):
        if self.timer is not None:
            self.pause()
        elif self.runs is not None and "disabled" not in self.pause_button.state():
            self.pause_button.configure(text="Pausar")
            self._tick()
