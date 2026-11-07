"""Re-analysis code on synthetic records in the exact hardware/raw_results.json schema."""

import json
import os

import numpy as np
import pytest

from labeled_rc.hardware_data import expval_y, load_records, recover_from_records


def _synthetic_records(theta_star=0.1, depths=(1, 4), instances=40, shots=3000, v=0.95, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for L in depths:
        for _ in range(instances):
            s = rng.choice([-1, 1], size=L)
            y = v * np.sin(theta_star * s.sum())
            n0 = int(rng.binomial(shots, (1 + y) / 2))
            records.append(
                {
                    "kind": "signal",
                    "L": L,
                    "theta_star": theta_star,
                    "sum_signs": int(s.sum()),
                    "counts": {"0": n0, "1": shots - n0},
                }
            )
        for _ in range(20):
            n0 = int(rng.binomial(shots, (1 + v) / 2))
            records.append(
                {
                    "kind": "viscal",
                    "L": L,
                    "theta_star": float("nan"),
                    "sum_signs": L,
                    "counts": {"0": n0, "1": shots - n0},
                }
            )
    return records


def test_expval_y():
    assert expval_y({"0": 75, "1": 25}) == 0.5
    assert expval_y({"1": 10}) == -1.0
    assert expval_y({}) == 0.0


def test_recovery_from_records(tmp_path):
    path = tmp_path / "raw_results.json"
    path.write_text(json.dumps(_synthetic_records()))
    result = recover_from_records(load_records(path), num_boot=300)
    for (theta_star, L), r in result.items():
        lo, hi = r["ci95"]
        assert lo <= theta_star <= hi
        assert r["theta_hat"] == pytest.approx(theta_star, abs=0.01)
        assert r["visibility"] == pytest.approx(0.95, abs=0.01)
        assert r["instances"] == 40


HARDWARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hardware")
RAW = os.path.join(HARDWARE_DIR, "online_data", "raw_results.json")
ANALYSIS = os.path.join(HARDWARE_DIR, "online_data", "analysis.json")


@pytest.mark.skipif(
    not (os.path.exists(RAW) and os.path.exists(ANALYSIS)), reason="ibm_marrakesh records not present"
)
def test_reanalysis_of_ibm_marrakesh_run_matches_paper_analysis():
    """Point estimates, visibilities and correlations agree with hardware/analyze.py's output."""
    ours = recover_from_records(load_records(RAW), num_boot=50)
    with open(ANALYSIS, encoding="utf-8") as fh:
        paper = json.load(fh)["results"]
    assert len(ours) == len(paper) == 30  # 5 injected angles x 6 depths
    for row in paper.values():
        r = ours[(round(row["theta_star"], 6), row["L"])]
        assert r["theta_hat"] == pytest.approx(row["theta_hat"], abs=1e-10)
        assert r["visibility"] == pytest.approx(row["v"], abs=1e-12)
        assert r["label_correlation"] == pytest.approx(row["C"], abs=1e-12)
        assert r["marginal_mean"] == pytest.approx(row["marginal"], abs=1e-12)
        assert r["instances"] == row["n_instances"]
