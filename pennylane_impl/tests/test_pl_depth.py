"""Depth theorem with qp.gradients.quantum_fisher, against Oracles O2/O3 and core/."""

import numpy as np
import pytest

from experiments.exp2_depth_scaling import coherent_FQ, labeled_EFQ_enumerate
from labeled_rc.depth import (
    coherent_closed_form,
    coherent_qfi,
    labeled_closed_form,
    labeled_qfi,
    trotter_qnode,
)

# Oracle O2 (alpha = 1 rad): L -> (labeled E[F_Q], coherent F_Q), from exp2_depth_scaling.py
O2 = {
    1: (0.708073, 0.708073),
    2: (1.534895, 3.065190),
    3: (1.554810, 3.579242),
    4: (2.127560, 1.288420),
    5: (3.047096, 0.031033),
    6: (3.125169, 0.010661),
}
# Oracle O3 (alpha = pi/2): labeled = ceil(L/2); coherent = 1, 1, 0, 0, 1, 1
O3 = {1: (1, 1), 2: (1, 1), 3: (2, 0), 4: (2, 0), 5: (3, 1), 6: (3, 1)}


@pytest.mark.parametrize("L", sorted(O2))
def test_oracle_O2(L):
    lab, coh = O2[L]
    assert labeled_qfi(L, 1.0) == pytest.approx(lab, abs=1e-6)
    assert coherent_qfi(L, 1.0) == pytest.approx(coh, abs=1e-6)


@pytest.mark.parametrize("L", sorted(O3))
def test_oracle_O3(L):
    lab, coh = O3[L]
    assert labeled_qfi(L, np.pi / 2) == pytest.approx(lab, abs=1e-9)
    assert coherent_qfi(L, np.pi / 2) == pytest.approx(coh, abs=1e-9)


@pytest.mark.parametrize("alpha", [0.3, 1.0, 2.2])
def test_closed_forms(alpha):
    for L in range(1, 8):
        assert labeled_qfi(L, alpha) == pytest.approx(labeled_closed_form(L, alpha), abs=1e-10)
        assert coherent_qfi(L, alpha) == pytest.approx(coherent_closed_form(L, alpha), abs=1e-10)


@pytest.mark.parametrize("p_plus", [0.5, 0.7, 0.9])
def test_matches_core_engine_including_partial_twirl(p_plus):
    for L in (2, 4, 6):
        assert labeled_qfi(L, 0.8, p_plus=p_plus) == pytest.approx(
            labeled_EFQ_enumerate(L, 0.8, p=p_plus), abs=1e-9
        )
        assert coherent_qfi(L, 0.8) == pytest.approx(coherent_FQ(L, 0.8), abs=1e-9)


def test_labeled_grows_while_coherent_stays_bounded():
    alpha = 1.0
    labeled = [labeled_closed_form(L, alpha) for L in (10, 20, 40)]
    coherent = [coherent_closed_form(L, alpha) for L in range(1, 41)]
    # labeled ~ L/2; coherent bounded by (1 / sin(alpha / 2))^2 for every L
    assert labeled[2] / labeled[0] == pytest.approx(4.0, rel=0.1)
    assert max(coherent) <= 1 / np.sin(alpha / 2) ** 2 + 1e-12
    # spot-check the PennyLane QFI at a depth beyond the oracle tables
    assert labeled_qfi(10, alpha) == pytest.approx(labeled[0], abs=1e-9)


def test_rejects_bad_labels():
    import pennylane as qp

    with pytest.raises(ValueError):
        trotter_qnode(qp.device("default.qubit", wires=1), 1.0, [1, 0])
