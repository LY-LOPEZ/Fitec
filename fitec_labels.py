"""Spanish presentation of distributions and their parameters."""

import math


MODEL_NAMES = {"Chi-cuadrado": "Ji-cuadrado", "Geométrica": "Geométrica (fracasos)"}

PARAMETER_LABELS = {
    "Uniforme": {"a": "a · Límite inferior", "b": "b · Límite superior"},
    "Exponencial": {"lambda_param": "λ · Tasa de ocurrencia"},
    "Erlang": {"k": "k · Número de etapas", "lambda_param": "λ · Tasa por etapa"},
    "Normal": {"mu": "μ · Media", "sigma": "σ · Desviación estándar"},
    "Chi-cuadrado": {"df": "m · Grados de libertad"},
    "t de Student": {"df": "m · Grados de libertad"},
    "F de Fisher": {"dfn": "m · Grados de libertad del numerador", "dfd": "n · Grados de libertad del denominador"},
    "Beta": {"alpha": "α · Forma", "beta": "β · Forma"},
    "Gamma": {"shape": "k · Forma", "scale": "λ · Tasa"},
    "Weibull": {"shape": "Forma", "scale": "Escala"},
    "Lognormal": {"mu": "μ · Media de ln(X)", "sigma": "σ · Desviación estándar de ln(X)"},
    "Cauchy": {"x0": "x₀ · Ubicación", "gamma": "γ · Escala"},
    "Geométrica": {"p": "p · Probabilidad de éxito"},
    "Geométrica (ensayos)": {"p": "p · Probabilidad de éxito"},
    "Binomial negativa": {"r": "k · Número de éxitos requeridos", "p": "p · Probabilidad de éxito"},
    "Binomial": {"n": "n · Número de ensayos", "p": "p · Probabilidad de éxito"},
    "Poisson": {"lambda_param": "λ · Número esperado de eventos en el intervalo"},
    "Uniforme discreta": {"low": "a · Límite inferior", "high": "b · Límite superior"},
    "Hipergeométrica": {"M": "M · Tamaño de la población", "n": "K · Éxitos en la población", "N": "n · Extracciones"},
    "Triangular": {"a": "Mínimo", "mode": "Valor más probable", "b": "Máximo"},
}


def model_display_name(name):
    return MODEL_NAMES.get(name, name)


def uses_rate(name, key):
    return name == "Gamma" and key == "scale"


def parameter_display_value(name, key, value):
    return 1 / value if uses_rate(name, key) and value > 0 else value


def translate_parameter_text(name, key, text):
    """Translate between stored scale and the visible rate, preserving invalid drafts."""
    if uses_rate(name, key):
        try:
            value = float(str(text).strip().replace(",", "."))
        except ValueError:
            return text
        if value > 0 and math.isfinite(value):
            return repr(1 / value).replace(".", ",")
    return text


def labeled_parameters(name, parameters):
    return {PARAMETER_LABELS[name][key]: parameter_display_value(name, key, value)
            for key, value in parameters.items()}
