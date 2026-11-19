"""Audit the desktop explorer and generator against explicit course parameterizations."""

import tkinter as tk
from tkinter import font as tkfont, ttk

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from scipy import stats

from desktop_app import CONTINUOUS, DISCRETE, TEAL, FitecApp, format_probability, parse_observations, parse_parameter, probability
from fitec_theme import NIGHT
from fitec_tools import FitecTools
from fitec_tutor import congruential_rows, decimal


REFERENCES = {
    "Normal": stats.norm(), "Uniforme": stats.uniform(scale=10),
    "Triangular": stats.triang(.4, scale=10),
    "Exponencial": stats.expon(scale=5), "Erlang": stats.gamma(2),
    "Chi-cuadrado": stats.chi2(4), "t de Student": stats.t(10), "F de Fisher": stats.f(5, 10),
    "Beta": stats.beta(2, 2), "Gamma": stats.gamma(2, scale=2), "Weibull": stats.weibull_min(1.5),
    "Lognormal": stats.lognorm(1), "Cauchy": stats.cauchy(),
    "Geométrica": stats.nbinom(1, .3), "Geométrica (ensayos)": stats.geom(.3),
    "Binomial negativa": stats.nbinom(5, .4), "Binomial": stats.binom(10, .5),
    "Poisson": stats.poisson(4), "Uniforme discreta": stats.randint(1, 7),
    "Hipergeométrica": stats.hypergeom(20, 7, 8),
}


@pytest.fixture(scope="module")
def desktop():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk requiere un escritorio: {exc}")
    root.withdraw()
    app = FitecTools(root, FitecApp, parse_parameter, parse_observations)
    app.explorer.set_animation(False)
    yield app
    root.destroy()


@pytest.mark.parametrize("name", REFERENCES)
def test_explorer_and_generator(desktop, name):
    ref = REFERENCES[name]
    discrete = name in DISCRETE
    factory, params, _ = (DISCRETE if discrete else CONTINUOUS)[name]
    model = factory(**params)
    explorer = desktop.explorer
    explorer.select_model(name, params, discrete=discrete, event=("entre", 1, 2))
    for function, expected in [(model.mean, ref.mean), (model.var, ref.var), (model.std, ref.std)]:
        np.testing.assert_allclose(function(), expected(), equal_nan=True)
    for x in [-3, 0, 1, 1.5, 5, 12]:
        previous = np.ceil(x) - 1 if discrete else x
        boundary = np.floor(x) if discrete else x
        for kind, expected in [("hasta", ref.cdf(boundary)), ("menos", ref.cdf(previous)),
                               ("encima", ref.sf(boundary)), ("al_menos", ref.sf(previous))]:
            assert probability(model, kind, x) == pytest.approx(expected)
        if discrete and float(x).is_integer():
            assert probability(model, "exacta", x) == pytest.approx(ref.pmf(x))
    expected = sum(ref.pmf([1, 2])) if discrete else ref.cdf(2) - ref.cdf(1)
    assert probability(model, "entre", 1, 2) == pytest.approx(expected)
    for cumulative in (False, True):
        explorer.graph_mode.set("Acumulada" if cumulative else "Densidad")
        explorer.update_plot()
        xs, ys, style = explorer._plot_data
        fn = ref.cdf if cumulative else (ref.pmf if discrete else ref.pdf)
        np.testing.assert_allclose(ys, fn(xs), atol=1e-13)
        if discrete and cumulative:
            assert explorer.ax.lines[0].get_drawstyle() == "steps-post"
        elif discrete:
            bars = explorer.ax.patches
            np.testing.assert_allclose([bar.get_height() for bar in bars], ys)
            selected = [k for k, bar in zip(xs, bars) if bar.get_facecolor() == to_rgba(TEAL)]
            assert selected == [k for k in xs if 1 <= k <= 2]
    desktop._transfer()
    generator = desktop.generator
    generator.generate()
    sample = np.array([float(v.replace(",", ".")) for v in generator.sample])
    np.testing.assert_allclose(sample, model.rvs(size=100, random_state=42))
    bars = generator.sample_axes.patches
    total = sum(bar.get_height() * (1 if discrete else bar.get_width()) for bar in bars)
    assert total == pytest.approx(1)


def test_invalid_input_clears_old_plot(desktop):
    explorer = desktop.explorer
    explorer.select_model("Normal", {"mu": 0, "sigma": 1}, discrete=False)
    explorer.parameter_vars["sigma"].set("0")
    explorer.update_plot()
    assert explorer._plot_data is None and not explorer.ax.lines
    explorer.parameter_vars["sigma"].set("1")
    explorer.update_plot()
    assert explorer.ax.axison and explorer.ax.lines


@pytest.mark.parametrize(("name", "point"), [("Geométrica", 0), ("Geométrica (ensayos)", 1)])
def test_certain_geometric_success(name, point):
    model = DISCRETE[name][0](p=1)
    np.testing.assert_array_equal(model.ppf([.001, .5, .999]), [point] * 3)
    assert model.mean() == point and model.var() == 0
    assert (model.rvs(size=50, random_state=42) == point).all()


def test_exponential_shaded_area(desktop):
    explorer = desktop.explorer
    explorer.graph_mode.set("Densidad")
    explorer.select_model("Exponencial", {"lambda_param": .2}, discrete=False, event=("hasta", 5, None))
    vertices = explorer.ax.collections[0].get_paths()[0].vertices
    x, y = vertices.T
    assert (x.min(), x.max()) == (0, 5)
    area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
    assert area == pytest.approx(1 - np.exp(-1), abs=1e-5)


def test_empirical_steps_and_resampling(desktop):
    explorer = desktop.explorer
    explorer.mode.set("Empírica")
    explorer.data_text.delete("1.0", "end")
    explorer.data_text.insert("1.0", "-1; 0,5; 0,5; 3")
    explorer.prob_kind.set("exacta")
    explorer.first_var.set("0,5")
    explorer.update_plot()
    x, y, style = explorer._plot_data
    assert style == "escalón" and explorer.result.cget("text") == "50,00 %"
    np.testing.assert_allclose(x[1:-1], [-1, .5, 3])
    np.testing.assert_allclose(y, [0, .25, .75, 1, 1])
    desktop._transfer()
    desktop.generator.generate()
    sample = [float(v.replace(",", ".")) for v in desktop.generator.sample]
    expected = np.random.default_rng(42).choice([-1, .5, .5, 3], 100)
    np.testing.assert_array_equal(sample, expected)


def test_large_poisson_view_and_extreme_probability():
    model = DISCRETE["Poisson"][0](lambda_param=100000)
    grid = FitecApp._window(None, model, [0])
    assert model.cdf(grid[-1]) - model.cdf(grid[0] - 1) >= .998
    pct, decimal = format_probability(np.nextafter(1., 0.))
    assert pct.startswith(">") and decimal != "= 1"


def test_wide_poisson_groups_preserve_selected_mass(desktop):
    explorer = desktop.explorer
    explorer.graph_mode.set("Densidad")
    low, high = 99992, 100009
    explorer.select_model("Poisson", {"lambda_param": 100000}, discrete=True,
                          event=("entre", low, high))
    starts, masses, style = explorer._plot_data
    ends = explorer._plot_interval_ends
    assert style == "intervalos"
    ref = stats.poisson(100000)
    np.testing.assert_allclose(masses, [ref.pmf(np.arange(a, b + 1)).sum()
                                       for a, b in zip(starts, ends)], rtol=1e-8)
    highlight = [bar.get_height() for bar in explorer.ax.patches if bar.get_facecolor() == to_rgba(TEAL)]
    assert sum(highlight) == pytest.approx(ref.cdf(high) - ref.cdf(low - 1))
    explorer._show_coordinates(low)
    assert "Masa" in explorer._tooltip.get_text()
    explorer.graph_mode.set("Acumulada")
    explorer.update_plot()
    x, y, _ = explorer._plot_data
    np.testing.assert_allclose(y, ref.cdf(x))
    assert {low - 1, low, high, high + 1}.issubset(x)
    span = explorer.ax.patches[0]
    assert (span.get_x(), span.get_width()) == (low, high + 1 - low)
    explorer.canvas.draw()


def test_concentrated_continuous_density_and_exact_cursor(desktop):
    explorer = desktop.explorer
    explorer.graph_mode.set("Densidad")
    low, high = .000052, .000154
    explorer.select_model("Beta", {"alpha": 2, "beta": 10000}, discrete=False,
                          event=("entre", low, high))
    ref = stats.beta(2, 10000)
    vertices = explorer.ax.collections[0].get_paths()[0].vertices
    x, y = vertices.T
    assert (x.min(), x.max()) == (low, high)
    area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
    assert area == pytest.approx(ref.cdf(high) - ref.cdf(low), abs=1e-5)
    assert explorer.ax.get_xlim()[1] < .01
    cursor = .0000937654
    explorer._show_coordinates(cursor)
    assert explorer._tooltip.xy == pytest.approx((cursor, ref.pdf(cursor)), rel=1e-12)
    explorer.canvas.draw()


def test_tick_labels_distinguish_nearby_coordinates():
    from fitec_plotting import DistinctTicks

    ticks = np.array([1e8, 1e8 + .001, 1e8 + .002])
    labels = DistinctTicks().format_ticks(ticks)
    assert len(set(labels)) == len(ticks)
    np.testing.assert_array_equal([float(label.replace(",", ".")) for label in labels], ticks)


def test_largest_congruential_residue_stays_below_one():
    m = 2**64
    uniform = congruential_rows(1, 0, m, m - 1, 1)[0][-1]
    assert uniform < 1
    assert decimal(uniform).startswith("0,999")


def test_normal_surface_slice(desktop):
    explorer = desktop.explorer
    explorer.select_model("Normal", {"mu": 0, "sigma": 1}, discrete=False, event=("entre", -1, 1))
    surface = explorer.surface
    surface.toggle()
    try:
        surface.slide(1.5)
        explorer.update_plot()
        x, sigma, density = surface.axes.lines[0].get_data_3d()
        np.testing.assert_allclose(sigma, 1.5)
        np.testing.assert_allclose(density, stats.norm.pdf(x, scale=1.5))
        assert explorer.ax in explorer.figure.axes
        explorer.parameter_vars["sigma"].set("0,0001")
        explorer.update_plot()
        x, sigma, density = surface.axes.lines[0].get_data_3d()
        assert np.count_nonzero(np.abs(x) < .0001) >= 9
        np.testing.assert_allclose(sigma, .0001)
        np.testing.assert_allclose(density, stats.norm.pdf(x, scale=.0001))
    finally:
        surface.close()


def test_galton_paths_and_exportable_sample(desktop):
    generator = desktop.generator
    generator.select_distribution("Binomial", {"n": 3, "p": .5})
    generator.count.set("20")
    generator.seed.set("42")
    board = generator.galton
    board.start(play=False)
    assert "75,00 %" in board.info.cget("text")
    for _ in range(80):
        board.tick()
        board.stop()
    expected = (np.random.default_rng(42).random((20, 3)) < .5).sum(axis=1)
    np.testing.assert_array_equal(generator.sample, expected.astype(str))
    np.testing.assert_array_equal(board.counts, np.bincount(expected, minlength=4))


def test_day_night_keeps_probability_and_sample(desktop):
    explorer, generator = desktop.explorer, desktop.generator
    explorer.graph_mode.set("Densidad")
    explorer.select_model("Binomial", {"n": 3, "p": .5}, discrete=True, event=("entre", 1, 2))
    desktop._transfer()
    generator.generate()
    sample = tuple(generator.sample)
    generator.select_distribution("Poisson", {"lambda_param": 4})
    button = desktop.menus.theme_button
    day_panel = ttk.Style(desktop.theme.root).lookup("Panel.TFrame", "background")
    day_axes = explorer.ax.get_facecolor()
    button.invoke()
    try:
        assert button.cget("text") == "Día"
        assert ttk.Style(desktop.theme.root).lookup("Panel.TFrame", "background") == NIGHT.panel
        assert explorer.ax.get_facecolor() == to_rgba(NIGHT.plot)
        assert explorer.result.cget("text") == "75,00 %"
        selected = [index for index, bar in enumerate(explorer.ax.patches)
                    if bar.get_facecolor() == to_rgba(NIGHT.focus)]
        assert selected == [1, 2]
        assert tuple(generator.sample) == sample
        theory = generator.sample_axes.lines[0]
        assert theory.get_color() == NIGHT.reference
        np.testing.assert_allclose(theory.get_ydata(), stats.binom(3, .5).pmf(theory.get_xdata()))
    finally:
        button.invoke()
    assert button.cget("text") == "Noche"
    assert explorer.ax.get_facecolor() == day_axes
    assert ttk.Style(desktop.theme.root).lookup("Panel.TFrame", "background") == day_panel


def test_zoom_resizes_interface_and_workspace_in_both_themes(desktop):
    root = desktop.theme.root
    menus = desktop.menus
    style = ttk.Style(root)
    font_size = tkfont.nametofont("TkDefaultFont", root=root).cget("size")
    icon_size = desktop.icons["zoom"].width()
    row_height = int(style.lookup("Sample.Treeview", "rowheight"))
    probability_text = desktop.explorer.result.cget("text")
    graph_limits = desktop.explorer.ax.get_xlim()
    try:
        menus.zoom_menu.invoke(3)
        assert menus.zoom_button.cget("text") == "125 %"
        assert tkfont.nametofont("TkDefaultFont", root=root).cget("size") > font_size
        assert desktop.icons["zoom"].width() > icon_size
        assert int(style.lookup("Sample.Treeview", "rowheight")) > row_height

        menus.open_fit_workspace()
        menus.fit_workspace.root.withdraw()
        zoomed_grid_row = int(style.lookup("Data.Treeview", "rowheight"))
        desktop.toggle_theme()
        assert int(style.lookup("Data.Treeview", "rowheight")) == zoomed_grid_row
        assert menus.zoom_button.cget("image")[0] == str(menus.night_icons["zoom"])

        menus.zoom_menu.invoke(0)
        assert menus.zoom_button.cget("text") == "90 %"
        assert desktop.icons["zoom"].width() < icon_size
        assert int(style.lookup("Data.Treeview", "rowheight")) < zoomed_grid_row
        assert desktop.explorer.result.cget("text") == probability_text
        assert desktop.explorer.ax.get_xlim() == graph_limits
    finally:
        menus.zoom_menu.invoke(1)
        if desktop.theme.night:
            desktop.toggle_theme()
        if menus.fit_workspace is not None:
            menus.fit_workspace.close()
            menus.fit_workspace = None
    assert desktop.icons["zoom"].width() == icon_size
    assert int(style.lookup("Sample.Treeview", "rowheight")) == row_height
