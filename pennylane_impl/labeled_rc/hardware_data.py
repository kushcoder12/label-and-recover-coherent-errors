"""Re-analysis of the paper's IBM hardware records with this package's estimator.

The records are the per-circuit JSON written by ``hardware/retrieve.py``
(``hardware/online_data/raw_results.json``). Each record has ``kind`` ("signal" or
"viscal"), ``L``, ``theta_star``, ``sum_signs`` and ``counts`` ({"0": n0, "1": n1}).
No new hardware runs are needed.
"""

import json

import numpy as np

from .estimator import bootstrap_theta, label_correlation, marginal_mean


def load_records(path):
    """Load a list of per-circuit records from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["records"] if isinstance(data, dict) and "records" in data else data


def expval_y(counts):
    """<Y> from counts of the Y-basis measurement (outcome 0 <-> +1)."""
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    return (n0 - n1) / (n0 + n1) if (n0 + n1) else 0.0


def recover_from_records(records, num_boot=500, seed=0):
    """Per (theta_star, L): visibility, theta_hat, 95% CI, label correlation, marginal mean."""
    viscal, signal = {}, {}
    for r in records:
        y = expval_y(r["counts"])
        if r["kind"] == "viscal":
            viscal.setdefault(int(r["L"]), []).append(y)
        elif r["kind"] == "signal":
            key = (round(float(r["theta_star"]), 6), int(r["L"]))
            signal.setdefault(key, []).append((int(r["sum_signs"]), y))

    out = {}
    for (theta_star, L), pairs in sorted(signal.items()):
        v = float(np.clip(np.mean(viscal[L]), 1e-6, 1.0)) if L in viscal else 1.0
        m = np.array([p[0] for p in pairs], float)
        y = np.array([p[1] for p in pairs], float)
        theta_hat, ci = bootstrap_theta(m, y, v, num_boot=num_boot, seed=seed)
        out[(theta_star, L)] = {
            "visibility": v,
            "theta_hat": theta_hat,
            "ci95": ci,
            "label_correlation": label_correlation(m, y),
            "marginal_mean": marginal_mean(y),
            "instances": len(pairs),
        }
    return out
