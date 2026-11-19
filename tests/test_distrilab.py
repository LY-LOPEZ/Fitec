import math
from urllib.error import HTTPError

import numpy as np
import pytest
from scipy import stats

from desktop_app import (
    CONTINUOUS,
    DISCRETE,
    FitecApp,
    format_probability,
    parse_observations,
    parse_parameter,
    probability,
)
from fitec_analysis import fit_models, ks_distance
from fitec_tutor import congruential_rows, local_suggestion, online_explanation, probability_steps
from src.distributions import (
    BinomialDistribution,
    ErlangDistribution,
    ExponentialDistribution,
    FisherFDistribution,
    GeometricDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    ZeroGeometricDistribution,
)


def test_exponential_probability_up_to_five():
    result = probability(ExponentialDistribution(lambda_param=0.2), "hasta", 5)
    assert result == pytest.approx(1.0 - math.exp(-1.0))


def test_continuous_all_inequalities_and_upper_tail_precision():
    dist = NormalDistribution(mu=0.0, sigma=1.0)
    assert probability(dist, "hasta", 0.0) == pytest.approx(0.5)
    assert probability(dist, "menos", 0.0) == pytest.approx(0.5)
    assert probability(dist, "encima", 0.0) == pytest.approx(0.5)
    assert probability(dist, "al_menos", 0.0) == pytest.approx(0.5)

    # Extreme upper tail where 1 - cdf would lose precision without sf
    expected_tail = float(stats.norm.sf(7.0))
    assert probability(dist, "encima", 7.0) == pytest.approx(expected_tail, rel=1e-12)
    expected_band = float(stats.norm.sf(6.5) - stats.norm.sf(7.5))
    assert probability(dist, "entre", 6.5, 7.5) == pytest.approx(expected_band, rel=1e-12)


def test_discrete_interval_counts_both_endpoints():
    dist = BinomialDistribution(n=3, p=0.5)
    assert probability(dist, "entre", 1, 2) == pytest.approx(0.75)
    assert probability(dist, "entre", 1.2, 1.8) == pytest.approx(0.0)
    assert probability(dist, "entre", 2, 2) == pytest.approx(0.375)


def test_discrete_strict_vs_inclusive_inequalities():
    dist = BinomialDistribution(n=4, p=0.5)
    # PMF at k=0..4 is 1/16, 4/16, 6/16, 4/16, 1/16
    assert probability(dist, "exacta", 2) == pytest.approx(6 / 16)
    assert probability(dist, "menos", 2) == pytest.approx(5 / 16)
    assert probability(dist, "hasta", 2) == pytest.approx(11 / 16)
    assert probability(dist, "encima", 2) == pytest.approx(5 / 16)
    assert probability(dist, "al_menos", 2) == pytest.approx(11 / 16)

    # Partition identities
    for k in range(5):
        total = (
            probability(dist, "menos", k)
            + probability(dist, "exacta", k)
            + probability(dist, "encima", k)
        )
        assert total == pytest.approx(1.0)
        assert probability(dist, "hasta", k) + probability(dist, "encima", k) == pytest.approx(1.0)
        assert probability(dist, "menos", k) + probability(dist, "al_menos", k) == pytest.approx(
            1.0
        )


def test_discrete_upper_tail_interval_precision():
    dist = PoissonDistribution(lambda_param=1.0)
    expected = float(stats.poisson.pmf(20, mu=1.0))
    assert probability(dist, "entre", 20, 20) == pytest.approx(expected, rel=1e-12)
    expected_pair = float(stats.poisson.sf(17, mu=1.0) - stats.poisson.sf(20, mu=1.0))
    assert probability(dist, "entre", 18, 20) == pytest.approx(expected_pair, rel=1e-12)


def test_course_geometric_starts_at_zero_and_matches_nbinom_one():
    p = 0.3
    q = 1.0 - p
    geom_zero = DISCRETE["Geométrica"][0](p=p)
    nbinom_one = NegativeBinomialDistribution(r=1, p=p)
    geom_trials = DISCRETE["Geométrica (ensayos)"][0](p=p)

    assert isinstance(geom_zero, ZeroGeometricDistribution)
    assert isinstance(geom_trials, GeometricDistribution)
    assert geom_zero.mean() == pytest.approx(q / p)
    assert geom_zero.var() == pytest.approx(q / p**2)
    assert probability(geom_zero, "exacta", 0) == pytest.approx(p)
    assert probability(geom_zero, "exacta", 4) == pytest.approx(p * q**4)
    assert probability(geom_zero, "hasta", 4) == pytest.approx(1.0 - q**5)
    assert probability(geom_zero, "hasta", 4) == pytest.approx(probability(nbinom_one, "hasta", 4))
    assert probability(geom_trials, "exacta", 1) == pytest.approx(p)


def test_erlang_and_fisher_f_moments():
    erlang = ErlangDistribution(k=3, lambda_param=0.5)
    assert erlang.mean() == pytest.approx(3 / 0.5)
    assert erlang.var() == pytest.approx(3 / 0.25)
    with pytest.raises(ValueError):
        ErlangDistribution(k=2.5, lambda_param=1.0)  # type: ignore[arg-type]

    m, n = 5.0, 10.0
    fisher = FisherFDistribution(dfn=m, dfd=n)
    assert fisher.mean() == pytest.approx(n / (n - 2))
    expected_var = 2 * n**2 * (m + n - 2) / (m * (n - 2) ** 2 * (n - 4))
    assert fisher.var() == pytest.approx(expected_var)


def test_empirical_probabilities_and_observations_parsing():
    values = parse_observations("0, 1, 2, 2, 3, 4")
    assert np.array_equal(values, np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0]))
    assert FitecApp._empirical_probability(values, "exacta", 2.0, None) == pytest.approx(2 / 6)
    assert FitecApp._empirical_probability(values, "menos", 2.0, None) == pytest.approx(2 / 6)
    assert FitecApp._empirical_probability(values, "hasta", 2.0, None) == pytest.approx(4 / 6)
    assert FitecApp._empirical_probability(values, "encima", 2.0, None) == pytest.approx(2 / 6)
    assert FitecApp._empirical_probability(values, "al_menos", 2.0, None) == pytest.approx(4 / 6)
    assert FitecApp._empirical_probability(values, "entre", 1.0, 3.0, ) == pytest.approx(4 / 6)

    decimal_commas = parse_observations("1,5; 2,5; 3,25")
    assert np.allclose(decimal_commas, [1.5, 2.5, 3.25])
    dot_csv = parse_observations("1.5,2.5")
    assert np.allclose(dot_csv, [1.5, 2.5])


def test_format_probability_never_rounds_interior_to_zero_or_hundred():
    assert format_probability(0.0) == ("0,00 %", "= 0")
    assert format_probability(1.0) == ("100,00 %", "= 1")
    pct_low, _ = format_probability(1e-5)
    assert pct_low != "0,00 %"
    pct_high, _ = format_probability(0.99999)
    assert pct_high != "100,00 %"


def test_tutor_local_suggestion_inequalities_and_units():
    # "mayor o igual a 5" must be parsed as al_menos, NOT exacta
    s1 = local_suggestion(
        "Llegan 4 clientes por hora según Poisson. "
        "¿Cuál es la probabilidad de que lleguen mayor o igual a 5 clientes?"
    )
    assert s1.name == "Poisson"
    assert s1.parameters == {"lambda_param": pytest.approx(4.0)}
    assert s1.event == ("al_menos", 5.0, None)

    # Time-unit conversion in Exponential (12 per hour -> 0.2 per minute)
    s2 = local_suggestion(
        "Llegan 12 clientes por hora de forma exponencial. "
        "¿Cuál es la probabilidad de esperar menos de 10 minutos?"
    )
    assert s2.name == "Exponencial"
    assert s2.parameters["lambda_param"] == pytest.approx(0.2)
    assert s2.event == ("menos", 10.0, None)

    # Time-interval scaling in Poisson (6 per hour in 30 minutes -> lambda = 3)
    s3 = local_suggestion(
        "Se reciben 6 llamadas por hora según Poisson. "
        "¿Cuál es la probabilidad de recibir exactamente 2 llamadas en 30 minutos?"
    )
    assert s3.name == "Poisson"
    assert s3.parameters["lambda_param"] == pytest.approx(3.0)
    assert s3.event == ("exacta", 2.0, None)

    # Binomial with p = 30%
    s4 = local_suggestion(
        "Sea una Binomial con n = 10 y p = 30%. Calcule P(X <= 3)."
    )
    assert s4.name == "Binomial"
    assert s4.parameters == {"n": 10, "p": pytest.approx(0.3)}
    assert s4.event == ("hasta", 3.0, None)

    # Normal given variance instead of std dev
    s5 = local_suggestion("Una variable Normal tiene media 50 y varianza 16. Hallar P(X > 54).")
    assert s5.name == "Normal"
    assert s5.parameters == {"mu": pytest.approx(50.0), "sigma": pytest.approx(4.0)}
    assert s5.event == ("encima", 54.0, None)


def test_congruential_rows_mixed_and_multiplicative():
    # LibroSimulacionCap2.pdf p. 9, Ejemplo 5
    rows = congruential_rows(a=5, c=7, m=8, seed=4, count=8)
    uniforms = [row[3] for row in rows]
    assert uniforms == pytest.approx([0.375, 0.75, 0.625, 0.0, 0.875, 0.25, 0.125, 0.5])

    # LibroSimulacionCap2.pdf p. 10, Ejemplo 6 (multiplicativo c=0)
    mult_rows = congruential_rows(a=3, c=0, m=100, seed=17, count=3)
    assert [row[3] for row in mult_rows] == pytest.approx([0.51, 0.53, 0.59])


def test_probability_steps_covers_all_modes():
    dist = BinomialDistribution(n=5, p=0.4)
    for kind in ("hasta", "menos", "encima", "al_menos", "exacta"):
        text = probability_steps(dist, kind, 2, None)
        assert text
    assert probability_steps(dist, "entre", 0, 2)
    assert probability_steps(dist, "entre", 2, 4)


def test_fit_models_maps_cleanly_to_explorer_catalog():
    rng = np.random.default_rng(123)
    cont_sample = rng.exponential(scale=5.0, size=60)
    cont_fits = fit_models(cont_sample, discrete=False, repetitions=20)
    for fit in cont_fits:
        name = fit.candidate.explorer_name
        assert name in CONTINUOUS
        cls, _, _ = CONTINUOUS[name]
        params = {
            k: parse_parameter(name, k, str(v))
            for k, v in fit.candidate.explorer_parameters(fit.parameters).items()
        }
        cls(**params)

    disc_sample = rng.geometric(p=0.35, size=60) - 1
    disc_fits = fit_models(disc_sample, discrete=True, repetitions=20)
    for fit in disc_fits:
        name = fit.candidate.explorer_name
        assert name in DISCRETE
        cls, _, _ = DISCRETE[name]
        params = {
            k: parse_parameter(name, k, str(v))
            for k, v in fit.candidate.explorer_parameters(fit.parameters).items()
        }
        cls(**params)
    assert ks_distance(np.array([0, 1, 2]), stats.poisson(1.0), discrete=True) > 0


def test_binomial_fit_uses_known_trials():
    values = np.random.RandomState(42).binomial(10, 0.5, 100)
    fit = fit_models(values, discrete=True, trials=10, repetitions=10)[0]
    assert fit.model == "binom"
    assert fit.parameters == pytest.approx((10, values.mean() / 10))
    assert fit.aic == pytest.approx(2 - 2 * fit.frozen().logpmf(values).sum())
    assert fit.repetitions == 10
    for invalid in (None, 0, 2, 3.5):
        with pytest.raises(ValueError, match="ensayos"):
            fit_models(values, discrete=True, trials=invalid, model_names=["binom"])


@pytest.mark.parametrize(("status", "detail"), [
    (401, "Configurar IA"),
    (403, "permisos"),
    (404, "modelo configurado"),
    (429, "saldo"),
    (500, "servicio en línea"),
])
def test_online_assistant_explains_service_errors(monkeypatch, status, detail):
    def reject(request, timeout):
        raise HTTPError(request.full_url, status, "Detalle privado del proveedor", None, None)

    monkeypatch.setattr("fitec_tutor.urlopen", reject)
    with pytest.raises(RuntimeError, match="^FITEC") as error:
        online_explanation("clave-simulada", "Una prueba local de FITEC.")
    message = str(error.value)
    assert str(status) in message
    assert detail in message
    assert "Detalle privado del proveedor" not in message
