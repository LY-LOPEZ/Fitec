"""Continuous probability distributions."""

from typing import Any

import numpy as np
from scipy import stats

from .base import Distribution


class NormalDistribution(Distribution):
    """Normal (Gaussian) distribution."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        """
        Initialize Normal distribution.

        Args:
            mu: Mean parameter
            sigma: Standard deviation parameter (must be > 0)
        """
        super().__init__("Normal", is_discrete=False)
        self.mu = mu
        self.sigma = sigma
        self.set_parameters(mu=mu, sigma=sigma)

    def _create_distribution(self, **params):
        """Create scipy normal distribution."""
        return stats.norm(loc=params["mu"], scale=params["sigma"])

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"mu": self.mu, "sigma": self.sigma}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.mu = params.get("mu", self.mu)
        self.sigma = params.get("sigma", self.sigma)

        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

        self._dist = self._create_distribution(mu=self.mu, sigma=self.sigma)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "mu": (-100.0, 100.0),
            "sigma": (0.1, 50.0),
        }


class ExponentialDistribution(Distribution):
    """Exponential distribution."""

    def __init__(self, lambda_param: float = 1.0):
        """
        Initialize Exponential distribution.

        Args:
            lambda_param: Rate parameter (must be > 0)
        """
        super().__init__("Exponential", is_discrete=False)
        self.lambda_param = lambda_param
        self.set_parameters(lambda_param=lambda_param)

    def _create_distribution(self, **params):
        """Create scipy exponential distribution."""
        return stats.expon(scale=1.0 / params["lambda_param"])

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"lambda": self.lambda_param}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.lambda_param = params.get("lambda_param", params.get("lambda", self.lambda_param))

        if self.lambda_param <= 0:
            raise ValueError("lambda must be positive")

        self._dist = self._create_distribution(lambda_param=self.lambda_param)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {"lambda": (0.1, 10.0)}


class UniformDistribution(Distribution):
    """Continuous Uniform distribution."""

    def __init__(self, a: float = 0.0, b: float = 1.0):
        """
        Initialize Uniform distribution.

        Args:
            a: Lower bound
            b: Upper bound (must be > a)
        """
        super().__init__("Uniform", is_discrete=False)
        self.a = a
        self.b = b
        self.set_parameters(a=a, b=b)

    def _create_distribution(self, **params):
        """Create scipy uniform distribution."""
        return stats.uniform(loc=params["a"], scale=params["b"] - params["a"])

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"a": self.a, "b": self.b}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.a = params.get("a", self.a)
        self.b = params.get("b", self.b)

        if self.a >= self.b:
            raise ValueError("a must be less than b")

        self._dist = self._create_distribution(a=self.a, b=self.b)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "a": (-10.0, 10.0),
            "b": (-10.0, 10.0),
        }


class TriangularDistribution(Distribution):
    """Continuous distribution described by a minimum, most likely value, and maximum."""

    def __init__(self, a: float = 0.0, mode: float = 4.0, b: float = 10.0):
        super().__init__("Triangular", is_discrete=False)
        self._parameters = {}
        self.set_parameters(a=a, mode=mode, b=b)

    def _create_distribution(self, **params):
        a, mode, b = params["a"], params["mode"], params["b"]
        return stats.triang((mode - a) / (b - a), loc=a, scale=b - a)

    def get_parameters(self) -> dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, **params):
        updated = {**self._parameters, **params}
        a, mode, b = updated["a"], updated["mode"], updated["b"]
        if not np.isfinite(b - a) or not a < b:
            raise ValueError("El mínimo debe ser menor que el máximo y el intervalo debe ser finito.")
        if not a <= mode <= b:
            raise ValueError("El valor más probable debe estar entre el mínimo y el máximo.")
        self._dist = self._create_distribution(**updated)
        self._parameters = updated

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {}


class BetaDistribution(Distribution):
    """Beta distribution."""

    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        """
        Initialize Beta distribution.

        Args:
            alpha: Shape parameter (must be > 0)
            beta: Shape parameter (must be > 0)
        """
        super().__init__("Beta", is_discrete=False)
        self.alpha = alpha
        self.beta_param = beta
        self.set_parameters(alpha=alpha, beta=beta)

    def _create_distribution(self, **params):
        """Create scipy beta distribution."""
        return stats.beta(a=params["alpha"], b=params["beta"])

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"alpha": self.alpha, "beta": self.beta_param}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.alpha = params.get("alpha", self.alpha)
        self.beta_param = params.get("beta", self.beta_param)

        if self.alpha <= 0 or self.beta_param <= 0:
            raise ValueError("alpha and beta must be positive")

        self._dist = self._create_distribution(alpha=self.alpha, beta=self.beta_param)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "alpha": (0.1, 10.0),
            "beta": (0.1, 10.0),
        }


class GammaDistribution(Distribution):
    """Gamma distribution."""

    def __init__(self, shape: float = 2.0, scale: float = 2.0):
        """
        Initialize Gamma distribution.

        Args:
            shape: Shape parameter (k, must be > 0)
            scale: Scale parameter (theta, must be > 0)
        """
        super().__init__("Gamma", is_discrete=False)
        self.shape = shape
        self.scale = scale
        self.set_parameters(shape=shape, scale=scale)

    def _create_distribution(self, **params):
        """Create scipy gamma distribution."""
        return stats.gamma(a=params["shape"], scale=params["scale"])

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"shape": self.shape, "scale": self.scale}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.shape = params.get("shape", self.shape)
        self.scale = params.get("scale", self.scale)

        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("shape and scale must be positive")

        self._dist = self._create_distribution(shape=self.shape, scale=self.scale)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "shape": (0.1, 10.0),
            "scale": (0.1, 10.0),
        }


class ErlangDistribution(Distribution):
    """Erlang distribution with integer shape and rate parameters."""

    def __init__(self, k: int = 2, lambda_param: float = 1.0):
        super().__init__("Erlang", is_discrete=False)
        if not float(k).is_integer():
            raise ValueError("k debe ser un entero positivo")
        self.k = int(k)
        self.lambda_param = lambda_param
        self.set_parameters(k=k, lambda_param=lambda_param)

    def _create_distribution(self, **params):
        return stats.gamma(a=params["k"], scale=1.0 / params["lambda_param"])

    def get_parameters(self) -> dict[str, Any]:
        return {"k": self.k, "lambda": self.lambda_param}

    def set_parameters(self, **params):
        raw_k = params.get("k", self.k)
        if not float(raw_k).is_integer():
            raise ValueError("k debe ser un entero positivo")
        self.k = int(raw_k)
        self.lambda_param = params.get(
            "lambda_param", params.get("lambda", self.lambda_param)
        )
        if self.k <= 0:
            raise ValueError("k debe ser un entero positivo")
        if self.lambda_param <= 0:
            raise ValueError("lambda debe ser positiva")
        self._dist = self._create_distribution(k=self.k, lambda_param=self.lambda_param)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {"k": (1, 30), "lambda": (0.01, 10.0)}


class FisherFDistribution(Distribution):
    """Fisher-Snedecor F distribution."""

    def __init__(self, dfn: float = 5.0, dfd: float = 10.0):
        super().__init__("F de Fisher", is_discrete=False)
        self.dfn = dfn
        self.dfd = dfd
        self.set_parameters(dfn=dfn, dfd=dfd)

    def _create_distribution(self, **params):
        return stats.f(dfn=params["dfn"], dfd=params["dfd"])

    def get_parameters(self) -> dict[str, Any]:
        return {"dfn": self.dfn, "dfd": self.dfd}

    def set_parameters(self, **params):
        self.dfn = params.get("dfn", self.dfn)
        self.dfd = params.get("dfd", self.dfd)
        if self.dfn <= 0 or self.dfd <= 0:
            raise ValueError("Los grados de libertad deben ser positivos")
        self._dist = self._create_distribution(dfn=self.dfn, dfd=self.dfd)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {"dfn": (0.1, 100.0), "dfd": (0.1, 100.0)}


class ChiSquareDistribution(Distribution):
    """Chi-square distribution."""

    def __init__(self, df: int = 3):
        """
        Initialize Chi-square distribution.

        Args:
            df: Degrees of freedom (must be > 0)
        """
        super().__init__("Chi-Square", is_discrete=False)
        self.df = df
        self.set_parameters(df=df)

    def _create_distribution(self, **params):
        """Create scipy chi-square distribution."""
        return stats.chi2(df=params["df"])

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"df": self.df}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.df = params.get("df", self.df)

        if self.df <= 0:
            raise ValueError("df must be positive")

        self._dist = self._create_distribution(df=self.df)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {"df": (1, 30)}


class StudentTDistribution(Distribution):
    """Student's t-distribution."""

    def __init__(self, df: float = 10.0):
        """
        Initialize Student's t-distribution.

        Args:
            df: Degrees of freedom (must be > 0)
        """
        super().__init__("Student-t", is_discrete=False)
        self.df = df
        self.set_parameters(df=df)

    def _create_distribution(self, **params):
        """Create scipy t distribution."""
        return stats.t(df=params["df"])

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"df": self.df}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.df = params.get("df", self.df)

        if self.df <= 0:
            raise ValueError("df must be positive")

        self._dist = self._create_distribution(df=self.df)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {"df": (1.0, 30.0)}


class WeibullDistribution(Distribution):
    """Weibull distribution."""

    def __init__(self, shape: float = 1.5, scale: float = 1.0):
        """
        Initialize Weibull distribution.

        Args:
            shape: Shape parameter (k, must be > 0)
            scale: Scale parameter (lambda, must be > 0)
        """
        super().__init__("Weibull", is_discrete=False)
        self.shape = shape
        self.scale = scale
        self.set_parameters(shape=shape, scale=scale)

    def _create_distribution(self, **params):
        """Create scipy weibull distribution."""
        return stats.weibull_min(c=params["shape"], scale=params["scale"])

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"shape": self.shape, "scale": self.scale}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.shape = params.get("shape", self.shape)
        self.scale = params.get("scale", self.scale)

        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("shape and scale must be positive")

        self._dist = self._create_distribution(shape=self.shape, scale=self.scale)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "shape": (0.1, 5.0),
            "scale": (0.1, 5.0),
        }


class LognormalDistribution(Distribution):
    """Lognormal distribution."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        """
        Initialize Lognormal distribution.

        Args:
            mu: Mean of underlying normal distribution
            sigma: Standard deviation of underlying normal distribution (must be > 0)
        """
        super().__init__("Lognormal", is_discrete=False)
        self.mu = mu
        self.sigma = sigma
        self.set_parameters(mu=mu, sigma=sigma)

    def _create_distribution(self, **params):
        """Create scipy lognormal distribution."""
        return stats.lognorm(s=params["sigma"], scale=np.exp(params["mu"]))

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"mu": self.mu, "sigma": self.sigma}

    def set_parameters(self, **params):
        """Set distribution parameters."""

        self.mu = params.get("mu", self.mu)
        self.sigma = params.get("sigma", self.sigma)

        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

        self._dist = self._create_distribution(mu=self.mu, sigma=self.sigma)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "mu": (-5.0, 5.0),
            "sigma": (0.1, 5.0),
        }


class CauchyDistribution(Distribution):
    """Cauchy distribution."""

    def __init__(self, x0: float = 0.0, gamma: float = 1.0):
        """
        Initialize Cauchy distribution.

        Args:
            x0: Location parameter
            gamma: Scale parameter (must be > 0)
        """
        super().__init__("Cauchy", is_discrete=False)
        self.x0 = x0
        self.gamma = gamma
        self.set_parameters(x0=x0, gamma=gamma)

    def _create_distribution(self, **params):
        """Create scipy cauchy distribution."""
        return stats.cauchy(loc=params["x0"], scale=params["gamma"])

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"x0": self.x0, "gamma": self.gamma}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.x0 = params.get("x0", self.x0)
        self.gamma = params.get("gamma", self.gamma)

        if self.gamma <= 0:
            raise ValueError("gamma must be positive")

        self._dist = self._create_distribution(x0=self.x0, gamma=self.gamma)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "x0": (-10.0, 10.0),
            "gamma": (0.1, 5.0),
        }
