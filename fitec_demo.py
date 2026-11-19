"""Interactive probability demonstrations for the desktop explorer."""
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom, norm

from fitec_theme import NIGHT, is_night

TEAL = "#087d77"
NIGHT_SURFACE = LinearSegmentedColormap.from_list("fitec-night", ["#234f50", "#43968e", "#9fe0cf"])


class NormalSurface:
    def __init__(self, parent, explorer):
        self.e = explorer
        self.axes = None
        self.timer = None
        self.button = ttk.Button(parent, text="Explorar en 3D", style="Surface.Graph.Toolbutton", command=self.toggle)
        self.button.pack(side="left", padx=5)
        self.scale = ttk.Scale(parent, length=85, command=self.slide)
        self.original = explorer.ax.get_position().frozen()

    def toggle(self):
        if self.axes is not None:
            self.close()
            return
        self.e.graph_mode.set("Densidad")
        self.e.update_plot()
        if self.e._plot_data is None:
            return
        sigma = self.e._create_distribution().std()
        self.scale.configure(from_=sigma / 2, to=sigma * 2)
        self.scale.set(sigma)
        self.scale.pack(side="left")
        self._build_axes(is_night(self.e.root))
        self.e.ax.set_position([.12, .13, .84, .16])
        self.button.configure(text="Volver a 2D")
        self.draw()
        self.tilt(0)

    def _build_axes(self, night):
        self.night = night
        with rc_context({"grid.color": NIGHT.grid, "grid.linewidth": .65} if night else {}):
            self.axes = self.e.figure.add_axes([.02, .32, .96, .67], projection="3d")
        self.axes.set_box_aspect((2.2, 1, .8))

    def slide(self, value):
        if self.axes is not None:
            self.e.parameter_vars["sigma"].set(format(float(value), ".8g"))

    def tilt(self, frame):
        self.timer = None
        self.axes.view_init(elev=9 + 2 * frame, azim=-90 + 3 * frame)
        self.e.canvas.draw_idle()
        if frame < 10:
            self.timer = self.e.root.after(25, self.tilt, frame + 1)

    def close(self):
        if self.timer is not None:
            self.e.root.after_cancel(self.timer)
            self.timer = None
        if self.axes is not None:
            self.axes.remove()
            self.axes = None
            self.e.ax.set_position(self.original)
            self.scale.pack_forget()
            self.button.configure(text="Explorar en 3D")
            self.e.canvas.draw_idle()

    def draw(self):
        if self.axes is None:
            return
        if self.e.distribution.get() != "Normal" or self.e.mode.get() != "Continuas" or self.e.graph_mode.get() != "Densidad":
            self.close()
            return
        model = self.e._create_distribution()
        mu, sigma = model.mean(), model.std()
        lo = min(float(self.scale.cget("from")), sigma)
        hi = max(float(self.scale.cget("to")), sigma)
        self.scale.configure(from_=lo, to=hi)
        if abs(self.scale.get() - sigma) > 1e-7 * sigma:
            self.scale.set(sigma)
        standard = np.linspace(-4, 4, 81)
        xs = np.unique(mu + np.r_[hi * standard, lo * standard[::2], sigma * standard[::2]])
        deviations = np.unique(np.r_[np.geomspace(lo, hi, 35), sigma])
        x, y = np.meshgrid(xs, deviations)
        night = is_night(self.e.root)
        if night != self.night:
            view = self.axes.elev, self.axes.azim, self.axes.roll
            self.axes.remove()
            self._build_axes(night)
            self.axes.view_init(*view)
        ax = self.axes
        ax.clear()
        if night:
            ax.set_facecolor(NIGHT.panel)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color(NIGHT.plot)
                axis.label.set_color(NIGHT.text)
                axis.line.set_color(NIGHT.border)
                axis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(colors=NIGHT.muted)
        ax.plot_surface(x, y, norm.pdf(x, mu, y), rcount=len(deviations), ccount=len(xs),
                        cmap=NIGHT_SURFACE if night else "GnBu", alpha=.65, linewidth=0)
        ax.plot(xs, np.full_like(xs, sigma), norm.pdf(xs, mu, sigma),
                color=NIGHT.reference if night else "#034f50", lw=3)
        ax.set(xlabel="x", ylabel="σ", zlabel="Densidad")
        self.e.canvas.draw_idle()


class Galton:
    def __init__(self, tabs, generator):
        self.g = generator
        self.frame = ttk.Frame(tabs, padding=8)
        tabs.add(self.frame, text="Tablero de Galton")
        bar = ttk.Frame(self.frame)
        bar.pack(fill="x")
        ttk.Button(bar, text="▶ Simular", command=self.start).pack(side="left")
        self.pause = ttk.Button(bar, text="Pausar", command=self.pause_run)
        self.pause.pack(side="left", padx=5)
        self.info = ttk.Label(self.frame)
        self.info.pack(anchor="w", pady=5)
        self.canvas = tk.Canvas(self.frame, background="#f4faf9", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.timer = None
        self.running = False
        self.frame.bind("<Unmap>", lambda _: self.stop())
        self.frame.bind("<Map>", self.enter)
        self.canvas.bind("<Configure>", lambda _: self.paint() if hasattr(self, "counts") else None)
        self.canvas.bind("<Button-1>", self.choose)

    def stop(self):
        if self.timer is not None:
            self.frame.after_cancel(self.timer)
        self.timer = None
        self.running = False

    def enter(self, _event):
        if self.g.name.get() != "Binomial":
            self.g.select_distribution("Binomial", {"n": 3, "p": .5})
        self.start(play=False)

    def start(self, play=True):
        self.stop()
        try:
            n = int(self.g.parameters["n"].get())
            p = float(self.g.parameters["p"].get().replace(",", "."))
            size, seed = int(self.g.count.get()), int(self.g.seed.get())
            if not (1 <= n <= 20 and 0 <= p <= 1 and 1 <= size <= 100000 and 0 <= seed < 2**32):
                raise ValueError
        except (KeyError, ValueError):
            self.info.configure(text="Elige Binomial con 1–20 ensayos, p entre 0 y 1, cantidad y semilla válidas.")
            return
        if getattr(self, "n", None) != n:
            self.selected = {n // 2, (n + 1) // 2}
        self.n, self.p, self.size = n, p, size
        self.seed = seed
        self.sample = []
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n + 1, dtype=int)
        self.step = self.right = 0
        self.pause.configure(text="Pausar")
        self.running = play
        self.paint()
        if play:
            self.tick()

    def pause_run(self):
        if self.running:
            self.stop()
            self.pause.configure(text="Continuar")
        elif hasattr(self, "counts") and self.counts.sum() < self.size:
            self.running = True
            self.pause.configure(text="Pausar")
            self.tick()

    def tick(self):
        self.timer = None
        if self.step == self.n:
            self.counts[self.right] += 1
            self.sample.append(str(self.right))
            self.step = self.right = 0
        else:
            self.step += 1
            self.right += int(self.rng.random() < self.p)
        self.paint()
        self.running = self.counts.sum() < self.size
        if self.running:
            self.timer = self.frame.after(65, self.tick)
        else:
            self.g._show_sample(self.sample.copy(),
                                f"Galton · Binomial(n={self.n}, p={self.p:g}) · "
                                f"{self.size} valores · Semilla {self.seed}",
                                model=self.g.models["Binomial"][0](n=self.n, p=self.p))

    def choose(self, event):
        if not hasattr(self, "counts") or event.y < self.canvas.winfo_height() * .73:
            return
        k = round((event.x - self.canvas.winfo_width()/2) / self.dx + self.n/2)
        if 0 <= k <= self.n:
            self.selected.symmetric_difference_update({k})
            self.paint()

    def paint(self):
        c = self.canvas
        night = is_night(self.frame)
        w, h = max(1, c.winfo_width()), max(1, c.winfo_height())
        c.delete("all")
        self.dx = dx = min(w / (self.n + 2), 85)
        dy = h * .60 / (self.n + 1)
        for row in range(self.n):
            for col in range(row + 1):
                x, y = w/2 + (col - row/2)*dx, 15 + (row + .5)*dy
                c.create_oval(x-3,y-3,x+3,y+3,fill=NIGHT.muted if night else "#76959e",outline="")
        x, y = w/2+(self.right-self.step/2)*dx,15+self.step*dy
        c.create_oval(x-6,y-6,x+6,y+6,fill=NIGHT.reference if night else "#e68137",outline="")
        total = int(self.counts.sum())
        for k, count in enumerate(self.counts):
            x = w/2 + (k-self.n/2)*dx
            base = h*.91
            selected = NIGHT.focus if night else TEAL
            unselected = NIGHT.inactive_bar if night else "#9bb4ba"
            c.create_rectangle(x-dx*.43,base-max(4,h*.20*count/max(1,self.counts.max())),
                               x+dx*.43,base,fill=selected if k in self.selected else unselected,outline="")
            c.create_text(x,base+10,text=str(k),fill=NIGHT.text if night else "black")
        keys = sorted(self.selected)
        theory = sum(binom.pmf(keys,self.n,self.p))
        observed = f"{100*self.counts[keys].sum()/total:.2f} %" if total else "—"
        self.info.configure(text=(f"n={self.n} · p={self.p:g}    Teórica: {100*theory:.2f} %    Observada: {observed}    "
                                  f"Bolitas: {total}/{self.size}    ·    Clic en una columna para marcarla").replace(".", ","))
