"""The FITEC tutoring workspace and its three guided activities."""

import math
import os
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from fitec_tutor import (
    EXERCISE, REFERENCES, congruential_rows, decimal, local_suggestion,
    online_explanation, online_suggestion, probability_steps, valid_event,
)


class FitecAssistant:
    def __init__(self, parent, toolbar, app, parse_parameter, parse_observations):
        self.parent = parent
        self.app = app
        self.explorer = app.explorer
        self.generator = app.generator
        self.parse_parameter = parse_parameter
        self.parse_observations = parse_observations
        self.catalog = {name: (spec, mode == "Discretas")
                        for mode, registry in self.explorer.catalog.items() for name, spec in registry.items()}
        self.key = os.getenv("OPENAI_API_KEY", "").strip()
        self.connection = tk.StringVar()
        self.busy = False
        self.inbox = Queue()
        self.suggestion = None
        self.facts = ""
        self.online_buttons = []
        self._build(toolbar)
        self._sync_online()

    def _build(self, toolbar):
        ttk.Label(toolbar, textvariable=self.connection, style="Help.TLabel").pack(side="left", padx=(0, 12))
        ttk.Button(toolbar, text="Configurar IA", image=self.app.icons["settings"], compound="left",
                   style="Toolbar.TButton", command=self.configure_ai).pack(side="left")
        body = ttk.Frame(self.parent, style="Panel.TFrame", padding=(16, 12))
        body.pack(fill="both", expand=True, padx=14, pady=14)
        heading = ttk.Frame(body, style="Panel.TFrame")
        heading.pack(fill="x", pady=(0, 9))
        ttk.Label(heading, text="Asistente FITEC", image=self.app.icons["assistant"], compound="left",
                  style="Section.TLabel").pack(anchor="w")
        ttk.Label(heading, text="Interpreta un ejercicio, entiende un resultado y reproduce el método del libro.",
                  style="Help.TLabel").pack(anchor="w", pady=(2, 0))
        self.tabs = ttk.Notebook(body)
        self.tabs.pack(fill="both", expand=True)
        self.exercise_tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=12)
        self.result_tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=12)
        self.guide_tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=12)
        self.tabs.add(self.exercise_tab, text="Pegar ejercicio")
        self.tabs.add(self.result_tab, text="Explicar mi resultado")
        self.tabs.add(self.guide_tab, text="Guía de generación")
        self._build_exercise()
        self._build_result()
        self._build_guide()

    def _button(self, parent, text, icon, callback, *, primary=False):
        return ttk.Button(parent, text=text, image=self.app.icons[icon], compound="left",
                          style="Primary.TButton" if primary else "Toolbar.TButton", command=callback)

    @staticmethod
    def _text(parent, *, height=7, editable=False):
        widget = ScrolledText(parent, wrap="word", height=height, width=34,
                              font=("Inter", 10), background="#f4f8f9", foreground="#103746",
                              padx=10, pady=8, relief="flat", highlightthickness=1,
                              highlightbackground="#d6e3e7", highlightcolor="#087d77", undo=editable)
        if not editable:
            widget.configure(state="disabled")
        return widget

    @staticmethod
    def _display(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _build_exercise(self):
        tab = self.exercise_tab
        tab.columnconfigure(0, weight=1, uniform="exercise")
        tab.columnconfigure(1, weight=1, uniform="exercise")
        tab.rowconfigure(1, weight=1)
        ttk.Label(tab, text="Enunciado").grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Label(tab, text="Distribución y supuestos").grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 5))
        self.exercise = self._text(tab, editable=True)
        self.exercise.grid(row=1, column=0, sticky="nsew")
        self.exercise.bind("<<Modified>>", self._exercise_changed)
        self.exercise_output = self._text(tab)
        self.exercise_output.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        self._display(self.exercise_output, "Pega un enunciado o carga el ejemplo de tiempos entre llegadas.")
        left = ttk.Frame(tab, style="Panel.TFrame")
        left.grid(row=2, column=0, sticky="ew", pady=(9, 0))
        self._button(left, "Cargar ejemplo", "paste", self.load_example).pack(side="left", padx=(0, 5))
        self._button(left, "Interpretar sin conexión", "model", self.interpret).pack(side="left")
        right = ttk.Frame(tab, style="Panel.TFrame")
        right.grid(row=2, column=1, sticky="ew", padx=(12, 0), pady=(9, 0))
        remote = self._button(right, "Consultar IA en línea", "assistant", self.interpret_online)
        remote.pack(side="left", padx=(0, 5))
        self.online_buttons.append(remote)
        self.open_button = self._button(right, "Abrir en Distribuciones", "arrow", self.open_suggestion, primary=True)
        self.open_button.pack(side="left")
        self.open_button.state(["disabled"])

    def _build_result(self):
        tab = self.result_tab
        self.result_output = self._text(tab)
        self.result_output.pack(fill="both", expand=True)
        self._display(self.result_output, "Calcula una probabilidad en Distribuciones y vuelve aquí para ver los pasos.")
        actions = ttk.Frame(tab, style="Panel.TFrame")
        actions.pack(fill="x", pady=(9, 0))
        self._button(actions, "Explicar resultado actual", "chart", self.explain, primary=True).pack(
            side="left", padx=(0, 6))
        remote = self._button(actions, "Pedir explicación a la IA", "assistant", self.explain_online)
        remote.pack(side="left")
        self.online_buttons.append(remote)

    def _build_guide(self):
        tab = self.guide_tab
        self.guide_intro = ttk.Label(tab, style="Help.TLabel", wraplength=870, justify="left")
        self.guide_intro.pack(anchor="w", pady=(0, 6))
        ttk.Label(tab, text="Congruencial lineal mixto", style="Section.TLabel").pack(anchor="w")
        ttk.Label(tab, text="xᵢ₊₁ = (a × xᵢ + c) mod m   ·   rᵢ₊₁ = xᵢ₊₁ / m   ·   0 ≤ r < 1",
                  style="Help.TLabel").pack(anchor="w", pady=(3, 6))
        controls = ttk.Frame(tab, style="Panel.TFrame")
        controls.pack(fill="x")
        self.lcg_inputs = {}
        for key, label, value in (("seed", "Semilla x₀", "4"), ("a", "a", "5"),
                                  ("c", "c", "7"), ("m", "m", "8"), ("count", "Pasos", "8")):
            field = ttk.Frame(controls, style="Panel.TFrame")
            field.pack(side="left", padx=(0, 8))
            var = tk.StringVar(value=value)
            self.lcg_inputs[key] = var
            ttk.Label(field, text=label).pack(anchor="w")
            ttk.Entry(field, textvariable=var, width=8).pack()
        self._button(controls, "Mostrar pasos", "fit", self.calculate_lcg, primary=True).pack(
            side="left", anchor="s", padx=(4, 6))
        self._button(controls, "Ejemplo del libro", "reset", self.load_lcg_example).pack(side="left", anchor="s")
        self.lcg_status = ttk.Label(tab, style="Help.TLabel", wraplength=860)
        self.lcg_status.pack(anchor="w", pady=(7, 4))
        frame = ttk.Frame(tab, style="Panel.TFrame")
        frame.pack(fill="both", expand=True)
        self.lcg_table = ttk.Treeview(frame, columns=("i", "total", "x", "r"), show="headings", height=4)
        for name, text, width in (("i", "Paso", 65), ("total", "a × x anterior + c", 170),
                                  ("x", "Residuo xᵢ", 155), ("r", "Uniforme rᵢ", 190)):
            self.lcg_table.heading(name, text=text)
            self.lcg_table.column(name, width=width, anchor="e")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.lcg_table.yview)
        self.lcg_table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.lcg_table.pack(side="left", fill="both", expand=True)
        ttk.Label(tab, text="Fuente: LibroSimulacionCap2.pdf, págs. 8–9. Ejemplo 5.",
                  style="Help.TLabel").pack(anchor="w", pady=(5, 0))
        self.refresh_guide()
        self.calculate_lcg()

    def configure_ai(self):
        key = simpledialog.askstring(
            "FITEC · Configurar IA", "Pega la clave API de tu proyecto.\n"
            "FITEC la usa durante esta sesión.\n"
            "Solo se envía el enunciado o el cálculo que elijas consultar en línea.\n"
            "Deja el campo vacío para desconectar la IA.",
            show="*", parent=self.parent.winfo_toplevel(),
        )
        if key is not None:
            self.key = key.strip()
            self._sync_online()

    def _sync_online(self):
        self.connection.set("FITEC consultando en línea…" if self.busy else
                            "Clave configurada" if self.key else "Guía local")
        for button in self.online_buttons:
            button.state(["!disabled" if self.key and not self.busy else "disabled"])

    def _ask(self, task, on_success, output):
        if not self.key or self.busy:
            return
        self.busy = True
        self._sync_online()
        self._display(output, "FITEC está consultando en línea…")

        def worker():
            try:
                self.inbox.put((True, task()))
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                self.inbox.put((False, str(exc)))

        Thread(target=worker, daemon=True).start()

        def poll():
            try:
                success, answer = self.inbox.get_nowait()
            except Empty:
                self.parent.after(90, poll)
                return
            self.busy = False
            self._sync_online()
            if success:
                on_success(answer)
            else:
                self._display(output, answer)

        self.parent.after(90, poll)

    def load_example(self):
        self.exercise.delete("1.0", "end")
        self.exercise.insert("1.0", EXERCISE)
        self.exercise.edit_modified(False)
        self.suggestion = None
        self.open_button.state(["disabled"])
        self._display(self.exercise_output, "Pulsa Interpretar sin conexión o Consultar IA en línea.")

    def _exercise_changed(self, _event):
        if self.exercise.edit_modified():
            self.exercise.edit_modified(False)
            self.suggestion = None
            self.open_button.state(["disabled"])

    def _exercise_text(self):
        return self.exercise.get("1.0", "end-1c").strip()

    def _valid_input(self, text):
        if not 1 <= len(text) <= 6000:
            self._display(self.exercise_output, "Pega un ejercicio de hasta 6000 caracteres.")
            return False
        return True

    def interpret(self):
        text = self._exercise_text()
        if self._valid_input(text):
            self._present(local_suggestion(text), text, "Guía local")

    def interpret_online(self):
        text = self._exercise_text()
        if not self._valid_input(text):
            return
        self.open_button.state(["disabled"])
        self.suggestion = None
        key = self.key
        models = {name: spec[2] for name, (spec, _) in self.catalog.items()}
        self._ask(lambda: online_suggestion(key, text, models),
                  lambda result: self._present(result, text, "FITEC en línea"), self.exercise_output)

    def _present(self, suggestion, text, origin):
        if text != self._exercise_text():
            self._display(self.exercise_output, "El enunciado cambió. Interprétalo de nuevo.")
            return
        self.suggestion = None
        self.open_button.state(["disabled"])
        lines = [origin, "", suggestion.explanation]
        if suggestion.assumptions:
            lines.extend(["", "Supuestos", *(f"• {item}" for item in suggestion.assumptions)])
        if suggestion.name in self.catalog:
            spec, discrete = self.catalog[suggestion.name]
            factory, defaults, labels = spec
            lines.extend(["", f"Distribución: {suggestion.name}"])
            for key, value in suggestion.parameters.items():
                if key in labels and isinstance(value, (int, float)):
                    lines.append(f"{labels[key]} = {decimal(value)}")
            missing = defaults.keys() - suggestion.parameters.keys()
            if missing:
                lines.append("Faltan: " + ", ".join(labels[key] for key in defaults if key in missing) + ".")
            elif defaults.keys() == suggestion.parameters.keys():
                try:
                    parameters = {key: self.parse_parameter(suggestion.name, key, str(value))
                                  for key, value in suggestion.parameters.items()}
                    factory(**parameters)
                except (ValueError, TypeError, OverflowError) as exc:
                    lines.append(f"Revisa los parámetros: {exc}")
                else:
                    event = valid_event(suggestion.event, discrete=discrete)
                    self.suggestion = (text, suggestion.name, parameters, discrete, event)
                    self.open_button.state(["!disabled"])
                    if event:
                        kind, first, second = event
                        labels_by_kind = {
                            "hasta": f"X ≤ {decimal(first)}",
                            "menos": f"X < {decimal(first)}",
                            "encima": f"X > {decimal(first)}",
                            "al_menos": f"X ≥ {decimal(first)}",
                            "exacta": f"X = {decimal(first)}",
                        }
                        event_text = (f"{decimal(first)} ≤ X ≤ {decimal(second)}" if kind == "entre"
                                      else labels_by_kind[kind])
                        lines.append(f"Probabilidad a calcular: P({event_text}).")
                    else:
                        lines.append("Se cargarán los parámetros del modelo; indica el límite en Distribuciones.")
            else:
                lines.append("La respuesta incluyó parámetros que no corresponden a este modelo.")
            if suggestion.name in REFERENCES:
                lines.extend(["", f"Referencia: {REFERENCES[suggestion.name]}."])
            if suggestion.name == "Geométrica":
                lines.append("X cuenta fracasos antes del primer éxito (x = 0, 1, 2, …), igual que el capítulo 4.")
            elif suggestion.name == "Geométrica (ensayos)":
                lines.append("X cuenta ensayos hasta el primer éxito (x = 1, 2, 3, …).")
        self._display(self.exercise_output, "\n".join(lines))

    def open_suggestion(self):
        if self.suggestion is None or self.suggestion[0] != self._exercise_text():
            return
        _, name, params, discrete, event = self.suggestion
        self.explorer.select_model(name, params, discrete=discrete, event=event)
        self.app.select_view("distributions")

    def show_result(self):
        self.tabs.select(self.result_tab)
        self.explain()

    def _explain_facts(self):
        explorer = self.explorer
        explorer.update_plot()
        settings = explorer.session_settings()
        result = explorer.result.cget("text")
        if result == "—":
            raise ValueError("Corrige los parámetros o límites en Distribuciones y vuelve a intentarlo.")
        dec_result = explorer.decimal_result.cget("text")
        full_result = f"{result} ({dec_result.lstrip('= ').strip()})" if dec_result else result
        name, kind = settings["name"], settings["probability"]
        lines = [name, f"{explorer.formula.cget('text')} = {full_result}", ""]
        if name == "Empírica":
            values = self.parse_observations(settings["observations"])
            first = float(settings["first"].replace(",", "."))
            if kind == "hasta":
                hits = int((values <= first).sum())
            elif kind == "menos":
                hits = int((values < first).sum())
            elif kind == "encima":
                hits = int((values > first).sum())
            elif kind == "al_menos":
                hits = int((values >= first).sum())
            elif kind == "exacta":
                hits = int((values == first).sum())
            else:
                second = float(settings["second"].replace(",", "."))
                hits = int(((values >= first) & (values <= second)).sum())
            ratio = hits / len(values)
            lines.append(f"Hay {hits} observaciones en el evento de un total de {len(values)}: "
                         f"{hits} / {len(values)} = {decimal(ratio)}.")
            sample_var = f", varianza muestral s² = {decimal(float(values.var(ddof=1)))}" if len(values) > 1 else ""
            lines.append(f"Media = {decimal(float(values.mean()))}, varianza empírica = {decimal(float(values.var()))}{sample_var}.")
        else:
            spec, _ = self.catalog[name]
            params = {key: self.parse_parameter(name, key, value) for key, value in settings["parameters"].items()}
            dist = spec[0](**params)
            first = float(settings["first"].replace(",", "."))
            second = float(settings["second"].replace(",", ".")) if kind == "entre" else None
            lines.append("Parámetros: " + ", ".join(f"{spec[2][key].split(' · ')[0]} = {decimal(value)}"
                                                   for key, value in params.items()) + ".")
            mean, var = dist.mean(), dist.var()
            if math.isfinite(mean) or math.isfinite(var):
                mean_txt = decimal(mean) if math.isfinite(mean) else "no definida"
                var_txt = decimal(var) if math.isfinite(var) else "no definida"
                lines.append(f"Momentos teóricos: E[X] = {mean_txt}, V[X] = {var_txt}.")
            if name == "Exponencial":
                rate = params["lambda_param"]
                lines.append("F(x) = 1 − e^(−λx) para x ≥ 0; para x < 0, F(x) = 0.")
                if kind in ("hasta", "menos", "encima", "al_menos") and first >= 0:
                    lines.append(f"Con λ = {decimal(rate)} y x = {decimal(first)}, "
                                 f"F(x) = 1 − e^(−{decimal(rate)} × {decimal(first)}).")
            elif name == "Normal":
                mu, sigma = params["mu"], params["sigma"]
                if second is None:
                    z = (first - mu) / sigma
                    lines.append(f"Estandarización: z = ({decimal(first)} − {decimal(mu)}) / {decimal(sigma)} = {decimal(z)}.")
                else:
                    z1, z2 = (first - mu) / sigma, (second - mu) / sigma
                    lines.append(f"Estandarización: z₁ = {decimal(z1)}, z₂ = {decimal(z2)}.")
            elif name == "Geométrica":
                p = params["p"]
                lines.append(f"f(x) = p·(1 − p)^x y F(x) = 1 − (1 − p)^(x + 1) para x = 0, 1, 2, … con q = 1 − p = {decimal(1 - p)}.")
            elif name == "Geométrica (ensayos)":
                p = params["p"]
                lines.append(f"f(x) = p·(1 − p)^(x − 1) para x = 1, 2, 3, … con q = 1 − p = {decimal(1 - p)}.")
            lines.append(probability_steps(dist, kind, first, second))
        lines.extend(["", f"Resultado de FITEC: {full_result}."])
        reference = REFERENCES.get(name)
        if reference:
            lines.append(f"Referencia: {reference}.")
        return "\n".join(lines)

    def explain(self):
        try:
            self.facts = self._explain_facts()
        except (ValueError, TypeError, OverflowError) as exc:
            self.facts = ""
            self._display(self.result_output, str(exc))
            return
        self._display(self.result_output, self.facts)

    def explain_online(self):
        self.explain()
        if self.facts:
            key, facts = self.key, self.facts
            self._ask(lambda: online_explanation(key, facts),
                      lambda text: self._display(self.result_output, facts + "\n\nExplicación de FITEC\n" + text),
                      self.result_output)

    def refresh_guide(self):
        config = self.generator.session_settings()
        self.guide_intro.configure(text=f"Generador actual: {config['name']}, semilla {config['seed']}, "
                                  f"{config['count']} valores. Para repetir una muestra, conserva la distribución, "
                                  "sus parámetros, la cantidad y la semilla. La tabla de abajo reproduce "
                                  "por separado el método de tu libro.")

    def load_lcg_example(self):
        for key, value in (("seed", "4"), ("a", "5"), ("c", "7"), ("m", "8"), ("count", "8")):
            self.lcg_inputs[key].set(value)
        self.calculate_lcg()

    def calculate_lcg(self):
        self.lcg_table.delete(*self.lcg_table.get_children())
        try:
            inputs = {key: int(var.get()) for key, var in self.lcg_inputs.items()}
        except ValueError:
            self.lcg_status.configure(text="Ingresa enteros en los cinco campos.", style="Error.TLabel")
            return
        try:
            rows = congruential_rows(**inputs)
        except ValueError as exc:
            self.lcg_status.configure(text=str(exc), style="Error.TLabel")
            return
        for index, total, residue, uniform in rows:
            self.lcg_table.insert("", "end", values=(index, total, residue, decimal(uniform)))
        _, total, residue, uniform = rows[0]
        expr = (f"({inputs['a']} × {inputs['seed']}) mod {inputs['m']}" if inputs["c"] == 0
                else f"({inputs['a']} × {inputs['seed']} + {inputs['c']}) mod {inputs['m']}")
        self.lcg_status.configure(style="Help.TLabel", text=(
            f"Primer paso: {expr} = {total} mod {inputs['m']} = {residue}. "
            f"Luego r₁ = {residue}/{inputs['m']} = {decimal(uniform)}."
        ))
