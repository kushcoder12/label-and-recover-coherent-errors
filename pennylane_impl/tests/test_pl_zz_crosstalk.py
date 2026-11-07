"""Static ZZ crosstalk in PennyLane: probe, sampling, and recovery against core/multiqubit_zz."""

from itertools import product

import numpy as np
import pennylane as qp
import pytest

from core.circuits import multi_qubit_zz_probe
from core.multiqubit_zz import joint_correlation_estimator
from labeled_rc.crosstalk import recover_zz, sample_zz_records, zz_probe_qnode, zz_visibility


@pytest.mark.parametrize("kappas", [[0.05], [0.03, -0.07], [0.02, 0.06, -0.04]])
def test_probe_matches_core_and_closed_form(kappas):
    Q, theta0 = len(kappas), 0.11
    probe = zz_probe_qnode(qp.device("default.qubit", wires=Q + 1))
    for s in (1, -1):
        for cs in product((1, -1), repeat=Q):
            pl = float(probe(theta0, kappas, s, list(cs)))
            core = multi_qubit_zz_probe(
                theta0, {j + 2: k for j, k in enumerate(kappas)}, s, {j + 2: c for j, c in enumerate(cs)}
            )
            closed = s * np.sin(theta0 + np.dot(kappas, cs))
            assert pl == pytest.approx(core, abs=1e-12)
            assert pl == pytest.approx(closed, abs=1e-12)


def test_probe_rejects_mismatched_labels():
    probe = zz_probe_qnode(qp.device("default.qubit", wires=3))
    with pytest.raises(ValueError):
        probe(0.1, [0.05, 0.02], 1, [1])


def test_sampled_recovery_one_neighbor_matches_core_estimator():
    theta0, kappa = 0.10, 0.05
    dev = qp.device("default.qubit", wires=2, seed=11)
    records = sample_zz_records(dev, theta0, [kappa], shots=40000, seed=1)
    labels = {(r[0], r[1][2]) for r in records}
    assert labels == set(product((1, -1), repeat=2))  # every label configuration was drawn

    ours = recover_zz(records, num_boot=300, seed=2)
    core = joint_correlation_estimator(records, target_qubit=1, neighbor_qubits=[2], B=50, seed=2)
    assert ours["theta0_hat"] == pytest.approx(core["theta0_hat"], abs=1e-12)
    assert ours["kappa_hat"][2] == pytest.approx(core["kappa_hat"][2], abs=1e-12)

    lo, hi = ours["ci95"]["theta0"]
    assert lo <= theta0 <= hi
    lo, hi = ours["ci95"]["kappa"][2]
    assert lo <= kappa <= hi


def test_sampled_recovery_two_neighbors_matches_core_estimator():
    theta0, kappas = 0.08, [0.04, -0.03]
    dev = qp.device("default.qubit", wires=3, seed=12)
    records = sample_zz_records(dev, theta0, kappas, shots=60000, seed=3)
    ours = recover_zz(records)
    core = joint_correlation_estimator(records, 1, [2, 3], B=20, seed=0)
    assert ours["theta0_hat"] == pytest.approx(core["theta0_hat"], abs=1e-12)
    for j in (2, 3):
        assert ours["kappa_hat"][j] == pytest.approx(core["kappa_hat"][j], abs=1e-12)
    assert ours["theta0_hat"] == pytest.approx(theta0, abs=0.02)
    assert ours["kappa_hat"][2] == pytest.approx(kappas[0], abs=0.02)
    assert ours["kappa_hat"][3] == pytest.approx(kappas[1], abs=0.02)


# Noise after the error block on the target (phase damping + depolarizing, both X-covariant,
# so the s = -1 dressing commutes through them) and a symmetric readout bit flip.
def _block_noise(op, **kwargs):
    qp.PhaseDamping(0.03, wires=op.wires[0])
    qp.DepolarizingChannel(0.01, wires=op.wires[0])


def _readout(op, **kwargs):
    qp.BitFlip(0.02, wires=op.wires)


NOISE = qp.NoiseModel(
    {qp.noise.op_eq(qp.IsingZZ): _block_noise},
    meas_map={qp.noise.meas_eq(qp.expval) | qp.noise.meas_eq(qp.sample): _readout},
)


def test_noise_only_rescales_by_visibility():
    dev = qp.device("default.mixed", wires=2)
    v = zz_visibility(dev, 1, noise_model=NOISE)
    expected_v = np.sqrt(1 - 0.03) * (1 - 4 * 0.01 / 3) * (1 - 2 * 0.02)
    assert v == pytest.approx(expected_v, abs=1e-12)
    probe = qp.add_noise(zz_probe_qnode(dev), NOISE)
    for s, c in product((1, -1), repeat=2):
        assert float(probe(0.1, [0.05], s, [c])) == pytest.approx(
            v * s * np.sin(0.1 + 0.05 * c), abs=1e-12
        )


def test_noisy_sampled_recovery_with_visibility_debiasing():
    theta0, kappa = 0.10, 0.05
    dev = qp.device("default.mixed", wires=2, seed=13)
    v = zz_visibility(dev, 1, noise_model=NOISE, shots=4000, repeats=10)
    records = sample_zz_records(dev, theta0, [kappa], shots=40000, seed=4, noise_model=NOISE)

    debiased = recover_zz(records, v=v, num_boot=300, seed=5)
    lo, hi = debiased["ci95"]["theta0"]
    assert lo <= theta0 <= hi
    lo, hi = debiased["ci95"]["kappa"][2]
    assert lo <= kappa <= hi

    # without the visibility correction the estimate is biased toward zero
    naive = recover_zz(records, v=1.0)
    assert abs(naive["theta0_hat"]) < abs(debiased["theta0_hat"])
