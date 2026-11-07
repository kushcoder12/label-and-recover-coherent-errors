"""Estimators for the coherent-error angle from labeled twirl data.

These are the same estimators as ``hardware/analyze.py`` (the paper's hardware
analysis), restated here so the PennyLane layer is self-contained:

- per instance i: label sum m_i = sum_l s_l and measured y_i = <Y>_i
- model: y_i = v(L) sin(theta * m_i), with v(L) from the visibility reference
- fit: least squares over theta (grid scan + parabolic refinement)
- at L = 1 this reduces to theta_hat = arcsin(E[s y] / v)
"""

import numpy as np


def label_correlation(m, y):
    """C = mean_i(m_i * y_i): the label-outcome correlation (grows with depth)."""
    return float(np.mean(np.asarray(m, float) * np.asarray(y, float)))


def marginal_mean(y):
    """mean_i(y_i): what standard RC keeps after averaging the labels away."""
    return float(np.mean(np.asarray(y, float)))


def arcsin_estimate(s, y, v=1.0):
    """Closed-form single-layer estimator theta_hat = arcsin(E[s y] / v)."""
    c = np.clip(label_correlation(s, y) / v, -1.0, 1.0)
    return float(np.arcsin(c))


def fit_theta(m, y, v, grid=None):
    """Least-squares fit of theta in y_i = v sin(theta * m_i)."""
    m = np.asarray(m, float)
    y = np.asarray(y, float)
    if grid is None:
        grid = np.linspace(-0.6, 0.6, 4001)
    rss = np.sum((v * np.sin(np.outer(grid, m)) - y[None, :]) ** 2, axis=1)
    k = int(np.argmin(rss))
    theta = grid[k]
    if 0 < k < len(grid) - 1:
        y0, y1, y2 = rss[k - 1], rss[k], rss[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-30:
            theta = grid[k] + 0.5 * (y0 - y2) / denom * (grid[k] - grid[k - 1])
    return float(theta)


def bootstrap_theta(m, y, v, num_boot=500, seed=0):
    """Point estimate and 95% bootstrap interval, resampling twirl instances."""
    m = np.asarray(m, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(m)
    boots = []
    for _ in range(num_boot):
        idx = rng.integers(n, size=n)
        boots.append(fit_theta(m[idx], y[idx], v))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return fit_theta(m, y, v), (float(lo), float(hi))
