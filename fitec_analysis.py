"""Fit candidate input distributions and compare them with observed data."""

from dataclasses import dataclass
import math

import numpy as np
from scipy import stats

from src.fitting.distribution_fitter import DistributionFitter


BOOTSTRAP_REPETITIONS = 199
MAX_FIT_ROWS = 20_000


@dataclass(frozen=True)
class Candidate:
    label: str
    scipy_name: str
    parameter_count: int
    minimum: int | None = None
    discrete: bool = False

    @property
    def distribution(self):
        return getattr(stats, self.scipy_name)

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
            # Failures before the first success (support starts at zero).
            return (1, 1 / (1 + mean))
        constraints = {} if self.minimum is None else {"floc": 0}
        return DistributionFitter(values).estimate_parameters(self.scipy_name, **constraints)

    def explorer_parameters(self, parameters):
        match self.scipy_name:
            case "norm":
                return {"mu": parameters[0], "sigma": parameters[1]}
            case "uniform":
                return {"a": parameters[0], "b": parameters[0] + parameters[1]}
            case "expon":
                return {"lambda_param": 1 / parameters[1]}
            case "gamma" | "weibull_min":
                return {"shape": parameters[0], "scale": parameters[2]}
            case "lognorm":
                return {"mu": math.log(parameters[2]), "sigma": parameters[0]}
            case "binom":
                return {"n": int(parameters[0]), "p": parameters[1]}
            case "poisson":
                return {"lambda_param": parameters[0]}
            case "geom":
                return {"p": parameters[0]}
            case "nbinom":
                return {"p": parameters[1]}
        raise ValueError("Modelo desconocido.")

    @property
    def explorer_name(self):
        if self.scipy_name == "nbinom":
            return "Geométrica"
        if self.scipy_name == "geom":
            return "Geométrica (ensayos)"
        return self.label


CONTINUOUS_MODELS = (
    Candidate("Normal", "norm", 2),
    Candidate("Uniforme", "uniform", 2),
    Candidate("Exponencial", "expon", 1, minimum=0),
    Candidate("Gamma", "gamma", 2, minimum=0),
    Candidate("Weibull", "weibull_min", 2, minimum=0),
    Candidate("Lognormal", "lognorm", 2, minimum=0),
)
DISCRETE_MODELS = (
    Candidate("Binomial", "binom", 1, minimum=0, discrete=True),
    Candidate("Poisson", "poisson", 1, minimum=0, discrete=True),
    Candidate("Geométrica", "geom", 1, minimum=1, discrete=True),
    Candidate("Geométrica (fallos)", "nbinom", 1, minimum=0, discrete=True),
)
MODELS = {model.scipy_name: model for model in (*CONTINUOUS_MODELS, *DISCRETE_MODELS)}


@dataclass(frozen=True)
class FitResult:
    model: str
    parameters: tuple[float, ...]
    aic: float
    bic: float
    ks: float
    p_value: float | None
    repetitions: int

    @property
    def candidate(self):
        return MODELS[self.model]

    def frozen(self):
        return self.candidate.distribution(*self.parameters)


def ks_distance(values, distribution, discrete):
    """Largest vertical distance before or after an empirical CDF jump."""
    x, counts = np.unique(values, return_counts=True)
    after = np.cumsum(counts) / len(values)
    before = after - counts / len(values)
    theoretical_after = distribution.cdf(x)
    theoretical_before = distribution.cdf(x - 1) if discrete else theoretical_after
    return float(max(np.max(np.abs(after - theoretical_after)),
                     np.max(np.abs(before - theoretical_before))))


def _bootstrap_ks(candidate, distribution, observed, size, rng, repetitions, trials=None):
    exceedances = 0
    valid = 0
    for _ in range(repetitions):
        simulated = distribution.rvs(size=size, random_state=rng)
        try:
            parameters = candidate.estimate(simulated, trials)
            distance = ks_distance(simulated, candidate.distribution(*parameters), candidate.discrete)
        except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError):
            continue
        if math.isfinite(distance):
            valid += 1
            exceedances += distance >= observed
    if valid < math.ceil(repetitions * .9):
        return None, valid
    return (1 + exceedances) / (1 + valid), valid


def fit_models(values, discrete, repetitions=BOOTSTRAP_REPETITIONS, *, model_names=None, trials=None):
    values = np.asarray(values, dtype=float)
    if len(values) < 5 or np.ptp(values) == 0:
        raise ValueError("Para ajustar, usa al menos cinco observaciones y dos valores distintos.")
    if len(values) > MAX_FIT_ROWS:
        raise ValueError(f"El ajuste admite hasta {MAX_FIT_ROWS:,} observaciones por columna.")
    if discrete and (np.any(values < 0) or np.any(values != np.floor(values))):
        raise ValueError("Para ajustar recuentos, usa enteros no negativos.")
    candidates = DISCRETE_MODELS if discrete else CONTINUOUS_MODELS
    if model_names is not None:
        candidates = [candidate for candidate in candidates if candidate.scipy_name in model_names]
        if not candidates:
            kind = "recuentos" if discrete else "datos continuos"
            raise ValueError(f"Marca al menos una distribución para {kind} en Configurar ajuste.")
    if discrete and any(c.scipy_name == "binom" for c in candidates):
        if trials is None:
            candidates = [c for c in candidates if c.scipy_name != "binom"]
            if not candidates:
                raise ValueError("Indica el número conocido de ensayos para ajustar la binomial.")
        elif type(trials) is not int or trials < 1 or np.max(values) > trials:
            raise ValueError("El número de ensayos debe ser entero positivo y cubrir el mayor recuento.")
    rng = np.random.default_rng(42)
    results = []
    for candidate in candidates:
        if candidate.scipy_name in {"gamma", "weibull_min", "lognorm"} and np.min(values) <= 0:
            continue
        try:
            parameters = tuple(float(value) for value in candidate.estimate(values, trials))
            distribution = candidate.distribution(*parameters)
            log_prob = distribution.logpmf if discrete else distribution.logpdf
            likelihood = float(np.sum(log_prob(values)))
            if not math.isfinite(likelihood):
                continue
            distance = ks_distance(values, distribution, discrete)
            if not math.isfinite(distance):
                continue
            p_value, valid = _bootstrap_ks(candidate, distribution, distance, len(values), rng, repetitions, trials)
        except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError):
            continue
        k = candidate.parameter_count
        results.append(FitResult(candidate.scipy_name, parameters, 2 * k - 2 * likelihood,
                                 k * math.log(len(values)) - 2 * likelihood, distance, p_value, valid))
    if not results:
        raise ValueError("Ninguno de los modelos disponibles se pudo ajustar a la columna.")
    return sorted(results, key=lambda fit: fit.aic)
