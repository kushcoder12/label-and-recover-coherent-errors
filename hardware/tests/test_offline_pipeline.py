"""hardware/tests/test_offline_pipeline.py — offline correctness of the owned-label pipeline.

Runs without qiskit / network. Verifies:
  1. spec generation owns exactly L labels per circuit; spec-integrity passes.
  2. the ideal (noiseless) statevector reproduces <Y> = sin(theta* * sum_signs).
  3. the noisy local sim shrinks but does not bias the signal (visibility 0<v<1).
  4. the estimator recovers theta* on the diagonal and passes the theta*=0 null control.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))       # hardware/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root (for core)

import config
import build_circuits as bc
import local_sim as ls
import analyze as an


def test_spec_owns_labels():
    specs = bc.generate_all_specs()
    assert len(specs) == config.total_signal_circuits() + config.total_viscal_circuits()
    for s in specs:
        assert len(s.sign_vector) == s.L
        assert len(s.layer_angles) == s.L
        assert bc.spec_integrity_ok(s)
    print(f"  [1] {len(specs)} specs, all own exactly L labels; integrity OK.")


def test_ideal_expectation():
    """Noiseless <Y> == sin(theta* * sum_signs) for several instances."""
    rng = np.random.default_rng(0)
    maxerr = 0.0
    for _ in range(200):
        theta = rng.uniform(-0.4, 0.4)
        L = int(rng.integers(1, 20))
        s = rng.choice([-1, 1], size=L)
        spec = bc.CircuitSpec(cid="t", kind="signal", theta_star=theta, L=L,
                              sign_vector=[int(x) for x in s],
                              layer_angles=[float(theta * x) for x in s],
                              sum_signs=int(s.sum()))
        # noiseless: infinite T1/T2, zero errors
        prof = ls.make_offline_profile(L, params={
            "T1": math.inf, "T2": math.inf, "dt_fixed": 1.0, "idle_per_cycle": 0.0,
            "r_1q": 0.0, "e01": 0.0, "e10": 0.0})
        _, pmeas = ls.simulate_counts(spec, shots=1, profile=prof, rng=rng)
        yexp = pmeas[0] - pmeas[1]           # <Y> = P0 - P1 (exact prob, no sampling)
        maxerr = max(maxerr, abs(yexp - math.sin(theta * s.sum())))
    assert maxerr < 1e-12, maxerr
    print(f"  [2] ideal <Y> == sin(theta*·Σs): max err {maxerr:.2e}")


def test_noise_shrinks_unbiased():
    """Noisy visibility 0<v<1 and the sign of the signal is preserved."""
    L = 8
    theta = 0.2
    s = np.ones(L, dtype=int)
    spec = bc.CircuitSpec(cid="t", kind="signal", theta_star=theta, L=L,
                          sign_vector=[1] * L, layer_angles=[theta] * L, sum_signs=L)
    prof = ls.make_offline_profile(L)
    _, pmeas = ls.simulate_counts(spec, shots=1, profile=prof,
                                  rng=np.random.default_rng(0))
    y = pmeas[0] - pmeas[1]
    ideal = math.sin(theta * L)
    v = y / ideal
    assert 0.0 < v < 1.0, v
    print(f"  [3] noisy visibility v={v:.3f} (0<v<1), signal sign preserved.")


def test_end_to_end_recovery():
    """Full offline pipeline recovers theta* and passes the null control."""
    # smaller grid for speed
    specs = (bc.generate_signal_specs(thetas=[0.0, 0.1, 0.2, 0.3],
                                      depths=[2, 4, 8], instances=40)
             + bc.generate_viscal_specs(depths=[2, 4, 8], repeats=20))
    records = ls.run_offline(specs, shots=4000, seed=7)
    analysis = an.analyze(records, B=300, seed=7)

    # null control: theta*=0 CI contains 0 at every depth
    for key, r in analysis["results"].items():
        if abs(r["theta_star"]) < 1e-9:
            lo, hi = r["theta_ci"]
            assert lo <= 0.0 <= hi, (key, lo, hi)

    # diagonal: recovered within tolerance at L=4,8
    worst = 0.0
    for key, r in analysis["results"].items():
        if r["theta_star"] > 0 and r["L"] in (4, 8):
            worst = max(worst, abs(r["theta_hat"] - r["theta_star"]))
    assert worst < 0.03, worst
    print(f"  [4] end-to-end recovery: max|theta_hat-theta*| (L=4,8) = {worst:.4f}; null OK.")


if __name__ == "__main__":
    test_spec_owns_labels()
    test_ideal_expectation()
    test_noise_shrinks_unbiased()
    test_end_to_end_recovery()
    print("\nALL OFFLINE TESTS PASSED.")
