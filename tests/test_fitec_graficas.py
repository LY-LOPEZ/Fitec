"""Audit the desktop explorer and generator against explicit course parameterizations."""

import tkinter as tk

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from scipy import stats

from desktop_app import CONTINUOUS, DISCRETE, TEAL, FitecApp, format_probability, parse_observations, parse_parameter, probability
from fitec_tools import FitecTools
from fitec_tutor import congruential_rows, decimal


REFERENCES = {
    "Normal": stats.norm(), "Uniforme": stats.uniform(scale=10),
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
    assert grid[0] <= model.median() <= grid[-1]
    pct, decimal = format_probability(np.nextafter(1., 0.))
    assert pct.startswith(">") and decimal != "= 1"


def test_largest_congruential_residue_stays_below_one():
    m = 2**64
    uniform = congruential_rows(1, 0, m, m - 1, 1)[0][-1]
    assert uniform < 1
    assert decimal(uniform).startswith("0,999")
