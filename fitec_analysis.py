"""Fit candidate input distributions and compare them with observed data."""

from dataclasses import dataclass
import math

import numpy as np
from scipy import optimize, special, stats

from fitec_labels import labeled_parameters, model_display_name
from src.fitting.distribution_fitter import DistributionFitter


BOOTSTRAP_REPETITIONS = 199
MAX_FIT_ROWS = 20_000
GOODNESS_TESTS = {
    "KS": ("ks", "p_value", "repetitions"),
    "Anderson–Darling": ("ad", "ad_p", "ad_repetitions"),
    "Chi-cuadrado": ("chi_square", "chi_square_p", "chi_square_repetitions"),
}


@dataclass(frozen=True)
class Candidate:
    label: str
    scipy_name: str
    parameter_count: int
    minimum: int | None = None
    discrete: bool = False
    identifier: str | None = None

    @property
    def distribution(self):
        return getattr(stats, self.scipy_name)

    @property
    def key(self):
        return self.identifier or self.scipy_name

    def estimate(self, values, trials=None):
        if self.minimum is not None and np.min(values) < self.minimum:
            raise ValueError("La muestra está fuera del soporte.")
        if self.discrete:
            mean = float(np.mean(values))
            if self.scipy_name == "binom":
                return trials, mean / trials
            if self.scipy_name == "poisson":
                return (mean,)
            if self.scipy_name == "geom":
                return (1 / mean,)
            if self.scipy_name == "randint":
                return int(np.min(values)), int(np.max(values)) + 1
            if self.key == "nbinom_general":
                return _fit_negative_binomial(values, mean)
            # Failures before the first success (support starts at zero).
            return (1, 1 / (1 + mean))
        constraints = {} if self.minimum is None else {"floc": 0}
        fitter = DistributionFitter(values)
        if self.scipy_name == "erlang":
            shape, _, _ = fitter.estimate_parameters("gamma", floc=0)
            mean = float(np.mean(values))
            shapes = {max(1, math.floor(shape)), max(1, math.ceil(shape))}
            shape = max(sorted(shapes), key=lambda k: np.sum(stats.gamma.logpdf(values, k, scale=mean / k)))
            return shape, 0, mean / shape
        if self.scipy_name in {"beta", "chi2", "t", "f"}:
            constraints = {"floc": 0, "fscale": 1}
        parameters = fitter.estimate_parameters(self.scipy_name, **constraints)
        if self.scipy_name == "chi2":
            degrees = {max(1, math.floor(parameters[0])), max(1, math.ceil(parameters[0]))}
            df = max(sorted(degrees), key=lambda k: np.sum(stats.chi2.logpdf(values, k)))
            return df, 0, 1
        return parameters

    def explorer_parameters(self, parameters):
        match self.scipy_name:
            case "norm":
                return {"mu": parameters[0], "sigma": parameters[1]}
            case "uniform":
                return {"a": parameters[0], "b": parameters[0] + parameters[1]}
            case "triang":
                shape, lower, width = parameters
                return {"a": lower, "mode": lower + shape * width, "b": lower + width}
            case "expon":
                return {"lambda_param": 1 / parameters[1]}
            case "gamma" | "weibull_min":
                return {"shape": parameters[0], "scale": parameters[2]}
            case "lognorm":
                return {"mu": math.log(parameters[2]), "sigma": parameters[0]}
            case "erlang":
                return {"k": int(parameters[0]), "lambda_param": 1 / parameters[2]}
            case "chi2" | "t":
                return {"df": parameters[0]}
            case "f":
                return {"dfn": parameters[0], "dfd": parameters[1]}
            case "beta":
                return {"alpha": parameters[0], "beta": parameters[1]}
            case "cauchy":
                return {"x0": parameters[0], "gamma": parameters[1]}
            case "binom":
                return {"n": int(parameters[0]), "p": parameters[1]}
            case "poisson":
                return {"lambda_param": parameters[0]}
            case "geom":
                return {"p": parameters[0]}
            case "nbinom":
                if self.key == "nbinom_general":
                    return {"r": int(parameters[0]), "p": parameters[1]}
                return {"p": parameters[1]}
            case "randint":
                return {"low": int(parameters[0]), "high": int(parameters[1]) - 1}
        raise ValueError("Modelo desconocido.")

    @property
    def explorer_name(self):
        if self.key == "nbinom":
            return "Geométrica"
        if self.scipy_name == "geom":
            return "Geométrica (ensayos)"
        return self.label

    @property
    def display_name(self):
        return model_display_name(self.explorer_name)

    def display_parameters(self, parameters):
        return labeled_parameters(self.explorer_name, self.explorer_parameters(parameters))

    def parameter_summary(self, parameters):
        return " · ".join(f"{label} = {value:.5g}".replace(".", ",")
                          for label, value in self.display_parameters(parameters).items())


CONTINUOUS_MODELS = (
    Candidate("Normal", "norm", 2),
    Candidate("Uniforme", "uniform", 2),
    Candidate("Exponencial", "expon", 1, minimum=0),
    Candidate("Gamma", "gamma", 2, minimum=0),
    Candidate("Weibull", "weibull_min", 2, minimum=0),
    Candidate("Lognormal", "lognorm", 2, minimum=0),
    Candidate("Erlang", "erlang", 2, minimum=0),
    Candidate("Chi-cuadrado", "chi2", 1, minimum=0),
    Candidate("t de Student", "t", 1),
    Candidate("F de Fisher", "f", 2, minimum=0),
    Candidate("Beta", "beta", 2, minimum=0),
    Candidate("Cauchy", "cauchy", 2),
    Candidate("Triangular", "triang", 3),
)
DISCRETE_MODELS = (
    Candidate("Binomial", "binom", 1, minimum=0, discrete=True),
    Candidate("Poisson", "poisson", 1, minimum=0, discrete=True),
    Candidate("Geométrica", "geom", 1, minimum=1, discrete=True),
    Candidate("Geométrica (fallos)", "nbinom", 1, minimum=0, discrete=True),
    Candidate("Binomial negativa", "nbinom", 2, minimum=0, discrete=True, identifier="nbinom_general"),
    Candidate("Uniforme discreta", "randint", 2, minimum=0, discrete=True),
)
MODELS = {model.key: model for model in (*CONTINUOUS_MODELS, *DISCRETE_MODELS)}


def _fit_negative_binomial(values, mean):
    variance = float(np.var(values))
    if not math.isfinite(variance) or variance <= mean:
        raise ValueError("La binomial negativa necesita varianza mayor que la media para estimar su dispersión.")
    points, frequencies = np.unique(values, return_counts=True)

    def score(shape):
        difference = special.digamma(points + shape) - special.digamma(shape)
        return np.dot(frequencies, difference) - len(values) * np.log1p(mean / shape)

    if score(1) <= 0:
        return 1, 1 / (1 + mean)
    # Above this relative scale the profile score loses precision near its Poisson limit.
    limit = max(2.0, mean / np.sqrt(np.finfo(float).eps))
    high = 2.0
    while high < limit and score(high) > 0:
        high = min(2 * high, limit)
    if score(high) > 0:
        raise ValueError("No se pudo distinguir la dispersión del límite Poisson.")
    shape = optimize.brentq(score, 1, high)
    shapes = sorted({max(1, math.floor(shape)), max(1, math.ceil(shape))})
    n = max(shapes, key=lambda r: np.dot(frequencies, stats.nbinom.logpmf(points, r, r / (r + mean))))
    return n, n / (n + mean)


@dataclass(frozen=True)
class FitResult:
    model: str
    parameters: tuple[float, ...]
    aic: float
    bic: float
    ks: float
    p_value: float | None
    repetitions: int
    ad: float | None = None
    ad_p: float | None = None
    ad_repetitions: int = 0
    chi_square: float | None = None
    chi_square_p: float | None = None
    chi_square_repetitions: int = 0

    @property
    def candidate(self):
        return MODELS[self.model]

    def frozen(self):
        return self.candidate.distribution(*self.parameters)

    def goodness(self, test):
        statistic, probability, repetitions = GOODNESS_TESTS[test]
        return getattr(self, statistic), getattr(self, probability), getattr(self, repetitions)


def ks_distance(values, distribution, discrete):
    """Largest vertical distance before or after an empirical CDF jump."""
    x, counts = np.unique(values, return_counts=True)
    after = np.cumsum(counts) / len(values)
    before = after - counts / len(values)
    theoretical_after = distribution.cdf(x)
    theoretical_before = distribution.cdf(x - 1) if discrete else theoretical_after
    return float(max(np.max(np.abs(after - theoretical_after)),
                     np.max(np.abs(before - theoretical_before))))


def anderson_darling(values, distribution, *, fitted_uniform=False):
    """Continuous AD statistic for observations strictly inside the support."""
    ordered = np.sort(values)
    if fitted_uniform:
        # Given the two sample extremes, the remaining observations are uniform inside them.
        ordered = ordered[1:-1]
    size = len(ordered)
    tails = distribution.logcdf(ordered) + distribution.logsf(ordered[::-1])
    if not np.all(np.isfinite(tails)):
        return None
    return max(0.0, float(-size - np.dot(2 * np.arange(1, size + 1) - 1, tails) / size))


def pearson_chi_square(values, distribution, parameter_count):
    """Group by fitted quantiles and join bins with small expected counts."""
    size = len(values)
    group_count = min(20, size // 5)
    if group_count <= parameter_count + 1:
        return None
    cutoffs = np.unique(distribution.ppf(np.arange(1, group_count) / group_count))
    boundaries = []
    probabilities = [0.0]
    minimum_mass = (5.0 - 1e-10) / size
    for cutoff, probability in zip(cutoffs, distribution.cdf(cutoffs)):
        if probability - probabilities[-1] >= minimum_mass and 1 - probability >= minimum_mass:
            boundaries.append(cutoff)
            probabilities.append(probability)
    if len(probabilities) <= parameter_count + 1:
        return None
    expected = size * np.diff([*probabilities, 1.0])
    indices = np.searchsorted(boundaries, values, side="left")
    counts = np.bincount(indices, minlength=len(expected))
    return float(np.sum((counts - expected) ** 2 / expected))


def _statistics(candidate, values, distribution, tests):
    results = {}
    for test in tests:
        if test == "KS":
            results[test] = ks_distance(values, distribution, candidate.discrete)
        elif test == "Anderson–Darling" and not candidate.discrete:
            results[test] = anderson_darling(values, distribution, fitted_uniform=candidate.scipy_name == "uniform")
        elif test == "Chi-cuadrado":
            results[test] = pearson_chi_square(values, distribution, candidate.parameter_count)
    return {test: value for test, value in results.items() if value is not None and math.isfinite(value)}


def _bootstrap_tests(candidate, distribution, observed, size, rng, repetitions, trials=None):
    exceedances = dict.fromkeys(observed, 0)
    valid = dict.fromkeys(observed, 0)
    for _ in range(repetitions):
        simulated = distribution.rvs(size=size, random_state=rng)
        try:
            parameters = candidate.estimate(simulated, trials)
            simulated_fit = candidate.distribution(*parameters)
            measured = _statistics(candidate, simulated, simulated_fit, observed)
        except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError):
            continue
        for test, statistic in measured.items():
            valid[test] += 1
            exceedances[test] += statistic >= observed[test]
    return {test: (statistic, (1 + exceedances[test]) / (1 + valid[test])
                   if valid[test] >= max(1, math.ceil(repetitions * .9)) else None, valid[test])
            for test, statistic in observed.items()}


def fit_models(values, discrete, repetitions=BOOTSTRAP_REPETITIONS, *, model_names=None, trials=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Usa una columna de números finitos.")
    if len(values) < 5 or np.min(values) == np.max(values):
        raise ValueError("Para ajustar, usa al menos cinco observaciones y dos valores distintos.")
    if len(values) > MAX_FIT_ROWS:
        raise ValueError(f"El ajuste admite hasta {MAX_FIT_ROWS:,} observaciones por columna.")
    if discrete and (np.any(values < 0) or np.any(values != np.floor(values))):
        raise ValueError("Para ajustar datos discretos, usa enteros no negativos.")
    candidates = DISCRETE_MODELS if discrete else CONTINUOUS_MODELS
    if model_names is not None:
        candidates = [candidate for candidate in candidates if candidate.key in model_names]
        if not candidates:
            kind = "datos discretos" if discrete else "datos continuos"
            raise ValueError(f"Marca al menos una distribución para {kind} en Configurar ajuste.")
    if discrete and any(c.scipy_name == "binom" for c in candidates):
        if trials is not None and (type(trials) is not int or trials < 1):
            raise ValueError("El número de ensayos debe ser entero positivo.")
        if trials is None or np.max(values) > trials:
            candidates = [c for c in candidates if c.scipy_name != "binom"]
            if not candidates:
                raise ValueError("Para la binomial, indica un número de ensayos igual o mayor al máximo observado.")
    rng = np.random.default_rng(42)
    results = []
    for candidate in candidates:
        if candidate.scipy_name in {"gamma", "weibull_min", "lognorm", "erlang", "chi2", "f", "beta"} and np.min(values) <= 0:
            continue
        if candidate.scipy_name == "beta" and np.max(values) >= 1:
            continue
        try:
            parameters = tuple(float(value) for value in candidate.estimate(values, trials))
            if not all(map(math.isfinite, parameters)):
                continue
            distribution = candidate.distribution(*parameters)
            log_prob = distribution.logpmf if discrete else distribution.logpdf
            likelihood = float(np.sum(log_prob(values)))
            if not math.isfinite(likelihood):
                continue
            measured = _statistics(candidate, values, distribution, GOODNESS_TESTS)
            if "KS" not in measured:
                continue
            diagnostics = _bootstrap_tests(candidate, distribution, measured, len(values), rng, repetitions, trials)
        except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError):
            continue
        k = candidate.parameter_count
        scores = {field: value for test, result in diagnostics.items()
                  for field, value in zip(GOODNESS_TESTS[test], result)}
        results.append(FitResult(candidate.key, parameters, 2 * k - 2 * likelihood,
                                 k * math.log(len(values)) - 2 * likelihood, **scores))
    if not results:
        raise ValueError("Ninguno de los modelos disponibles se pudo ajustar a la columna.")
    return sorted(results, key=lambda fit: fit.aic)


def fit_columns(samples, kind, *, model_names=None, trials=None):
    """Adjust each captured column independently, preserving recoverable errors."""
    fitted = {}
    errors = {}
    for column, values in samples.items():
        counts = bool(np.all(values >= 0) and np.all(values == np.floor(values)))
        discrete = kind in ("Discretos", "Recuentos") or (kind == "Automático" and counts)
        try:
            fitted[column] = fit_models(values, discrete, model_names=model_names, trials=trials)
        except (ValueError, RuntimeError, OverflowError) as exc:
            errors[column] = str(exc)
    return fitted, errors
