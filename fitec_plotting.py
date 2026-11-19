"""Sampling and probability helpers for FITEC figures."""

import numpy as np
from matplotlib.ticker import Formatter


def _format_tick(value, precision):
    return f"{value:.{precision}g}".replace(".", ",")


class DistinctTicks(Formatter):
    """Keep compact labels while distinguishing nearby tick values."""

    def __init__(self):
        self.precision = 4

    def set_locs(self, locs):
        super().set_locs(locs)
        values = np.unique(locs)
        for self.precision in range(4, 18):
            labels = {_format_tick(value, self.precision) for value in values}
            if len(labels) == len(values):
                break

    def __call__(self, value, pos=None):
        return _format_tick(value, self.precision)


def curve_grid(low, high, ppf, *, limits=()):
    """Resolve both the visible domain and regions where probability is concentrated."""
    quantiles = np.asarray(ppf(np.linspace(.001, .999, 257)))
    visible = quantiles[(quantiles >= low) & (quantiles <= high)]
    boundaries = np.clip(limits, low, high)
    interval = np.linspace(*boundaries, 129) if len(boundaries) == 2 else boundaries
    return np.unique(np.r_[np.linspace(low, high, 600), visible, interval])


def discrete_grid(low, high):
    """Use individual integers when possible and bounded sampling for wide supports."""
    if high - low <= 300:
        return np.arange(low, high + 1)
    stride = (high - low + 299) // 300
    return np.r_[np.arange(low, high, stride), high]


def interval_masses(model, starts, ends):
    """Probability of inclusive integer intervals, with stable upper-tail differences."""
    before = starts - 1
    cumulative = model.cdf(before)
    return np.clip(np.where(cumulative > .5, model.sf(before) - model.sf(ends),
                            model.cdf(ends) - cumulative), 0, 1)
