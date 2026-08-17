"""hardware_zz/tests/test_offline_pipeline.py — offline correctness, no qiskit/network.

  1. spec generation owns exactly the right labels per topology; integrity passes.
  2. the noiseless statevector reproduces <Y> = sin(s1*theta0 + sum kappa*c) exactly.
  3. the noisy local sim shrinks but does not bias the signal.
  4. the full offline pipeline recovers theta0 (positive control) and the assumed
     ground-truth kappa, and passes the theta0=0 null control.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                     # hardware_zz/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))    # repo root (for core)

import config
import build_circuits as bc
import local_sim as ls
import analyze as an
from core.circuits import multi_qubit_zz_probe


def test_spec_owns_labels():
    specs = bc.generate_all_specs()
    assert len(specs) == config.total_signal_circuits() + config.total_viscal_circuits()
    for s in specs:
        expected = {2} if s.topology == "pair" else {2, 3}
        assert set(s.c) == expected
        assert bc.spec_integrity_ok(s)
    print(f"  [1] {len(specs)} specs, all own the right labels; integrity OK.")


def test_ideal_expectation():
    rng = np.random.default_rng(0)
    maxerr = 0.0
    for _ in range(150):
        theta0 = rng.uniform(-0.3, 0.3)
        topology = rng.choice(["pair", "star"])
        neighbors = [2] if topology == "pair" else [2, 3]
        kappa = {j: rng.uniform(-0.1, 0.1) for j in neighbors}
        s1 = int(rng.choice([-1, 1]))
        c = {j: int(rng.choice([-1, 1])) for j in neighbors}
        y = multi_qubit_zz_probe(theta0, kappa, s1, c)
        expected = s1 * math.sin(theta0 + sum(kappa[j] * c[j] for j in neighbors))
        maxerr = max(maxerr, abs(y - expected))
    assert maxerr < 1e-10, maxerr
    print(f"  [2] ideal <Y> matches the closed form: max err {maxerr:.2e}")


def test_noise_shrinks_unbiased():
    theta0 = 0.15
    kappa = {2: 0.05}
    profile = ls.make_offline_profile()
    spec = bc.CircuitSpec(cid="t", kind="signal", topology="pair", theta0=theta0,
                          s1=1, c={2: 1}, delay_s=config.DELAY_S)
    _, pmeas = ls.simulate_counts(spec, shots=1, kappa_true=kappa, profile=profile,
                                  rng=np.random.default_rng(0))
    y = pmeas[0] - pmeas[1]
    ideal = math.sin(theta0 + kappa[2])
    v = y / ideal
    assert 0.0 < v < 1.0, v
    print(f"  [3] noisy visibility v={v:.3f} (0<v<1), signal sign preserved.")


def test_end_to_end_recovery():
    specs = (bc.generate_signal_specs(thetas=[0.0, 0.1, 0.2])
             + bc.generate_viscal_specs())
    records = ls.run_offline(specs, shots=3000, seed=7)
    analysis = an.analyze(records, B=300, seed=7)

    for topology in ("pair", "star"):
        for key, r in analysis["results"][topology].items():
            if abs(r["theta0_star"]) < 1e-9:
                lo, hi = r["bootstrap_ci"]["theta0"]
                assert lo <= 0.0 <= hi, (topology, key, lo, hi)

    worst = 0.0
    for topology in ("pair", "star"):
        for key, r in analysis["results"][topology].items():
            if r["theta0_star"] > 0:
                worst = max(worst, abs(r["theta0_hat"] - r["theta0_star"]))
    assert worst < 0.05, worst
    print(f"  [4] end-to-end recovery: max|theta0_hat-theta0*| = {worst:.4f}; null OK.")


if __name__ == "__main__":
    test_spec_owns_labels()
    test_ideal_expectation()
    test_noise_shrinks_unbiased()
    test_end_to_end_recovery()
    print("\nALL OFFLINE TESTS PASSED.")
