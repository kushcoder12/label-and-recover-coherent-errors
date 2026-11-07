"""Static ZZ crosstalk recovered from owned labels, in PennyLane.

PennyLane version of ``core/multiqubit_zz.py`` (experiments 11-13). The target qubit (wire 0)
starts in |+>; each neighbour j (wire j, j = 1..Q) is prepared in |0> (label c_j = +1) or |1>
(c_j = -1). The coherent error block is

    RZ(theta0) on the target, then IsingZZ(kappa_j) on (target, neighbour j) for every j,

i.e. exp(-i H) with H = (theta0/2) Z_0 + sum_j (kappa_j/2) Z_0 Z_j, the Hamiltonian evolved by
``core.circuits.multi_qubit_zz_probe``. The target's twirl label s = -1 dresses the whole block
with X on the target, which flips the sign of every Z_0-bearing term. The ideal outcome is

    <Y_0> = s * sin(theta0 + sum_j kappa_j c_j).

Neighbour indices in records and estimates follow the ``core`` convention: target = qubit 1,
neighbours = qubits 2..Q+1 (wire j <-> qubit j + 1).
"""

from collections import Counter
from itertools import product

import numpy as np
import pennylane as qp


def _probe_ops(theta0, kappas, s, cs):
    qp.Hadamard(wires=0)
    for j, c in enumerate(cs, start=1):
        if c == -1:
            qp.PauliX(wires=j)
    if s == -1:
        qp.PauliX(wires=0)
    qp.RZ(theta0, wires=0)
    for j, kappa in enumerate(kappas, start=1):
        qp.IsingZZ(kappa, wires=[0, j])
    if s == -1:
        qp.PauliX(wires=0)
    # rotate Y into Z for the measurement
    qp.adjoint(qp.S(wires=0))
    qp.Hadamard(wires=0)


def zz_probe_qnode(dev, sample=False):
    """QNode f(theta0, kappas, s, cs) -> <Y> on the target (or per-shot +/-1 samples).

    ``kappas`` and ``cs`` are sequences of the same length Q (one entry per neighbour);
    ``s`` and every ``cs[j]`` are +/-1 labels supplied by the caller.
    """

    @qp.qnode(dev)
    def circuit(theta0, kappas, s, cs):
        if len(kappas) != len(cs):
            raise ValueError("kappas and cs must have one entry per neighbour")
        _probe_ops(theta0, kappas, s, cs)
        if sample:
            return qp.sample(qp.PauliZ(0))
        return qp.expval(qp.PauliZ(0))

    return circuit


def zz_visibility(dev, num_neighbors, noise_model=None, shots=None, repeats=1):
    """Visibility v: the probe at theta0 = pi/2, kappa = 0, all labels +1 (ideal <Y> = 1)."""
    qnode = zz_probe_qnode(dev)
    if noise_model is not None:
        qnode = qp.add_noise(qnode, noise_model)
    if shots is not None:
        qnode = qp.set_shots(qnode, shots)
    zeros, ones = [0.0] * num_neighbors, [1] * num_neighbors
    return float(np.mean([qnode(np.pi / 2, zeros, 1, ones) for _ in range(repeats)]))


def sample_zz_records(dev, theta0, kappas, shots, seed=None, noise_model=None):
    """Draw ``shots`` owned-label records (s, {j: c_j}, b) by sampling the PennyLane circuit.

    Every shot gets independent uniform labels. Shots that share a label configuration are
    executed together with ``qp.set_shots``; ``dev`` should be seeded for reproducibility.
    """
    rng = np.random.default_rng(seed)
    Q = len(kappas)
    labels = np.where(rng.random((shots, 1 + Q)) < 0.5, 1, -1)
    counts = Counter(map(tuple, labels))

    qnode = zz_probe_qnode(dev, sample=True)
    if noise_model is not None:
        qnode = qp.add_noise(qnode, noise_model)

    records = []
    for config in product((1, -1), repeat=1 + Q):
        n = counts.get(config, 0)
        if n == 0:
            continue
        s, cs = config[0], config[1:]
        outcomes = np.atleast_1d(qp.set_shots(qnode, n)(theta0, list(kappas), s, list(cs)))
        c_dict = {j + 2: int(c) for j, c in enumerate(cs)}
        records += [(int(s), c_dict, int(b)) for b in outcomes]
    order = rng.permutation(len(records))
    return [records[i] for i in order]


def _point_estimate(s, c, b, neighbors, v):
    X = float(np.mean(s * b)) / v
    Y = {j: float(np.mean(s * c[j] * b)) / v for j in neighbors}
    if len(neighbors) == 1:
        (j,) = neighbors
        a_plus = np.arcsin(np.clip(X + Y[j], -1.0, 1.0))
        a_minus = np.arcsin(np.clip(X - Y[j], -1.0, 1.0))
        return float(0.5 * (a_plus + a_minus)), {j: float(0.5 * (a_plus - a_minus))}
    theta0 = float(np.arcsin(np.clip(X, -1.0, 1.0)))
    cos0 = np.cos(theta0)
    return theta0, {j: float(np.arcsin(np.clip(Y[j], -1.0, 1.0)) / cos0) for j in neighbors}


def recover_zz(records, v=1.0, num_boot=0, seed=0):
    """Recover theta0 and every kappa_j from labeled records, debiased by the visibility v.

    Uses the correlations X = E[s b] / v and Y_j = E[s c_j b] / v. For one neighbour the
    inversion is exact (theta0 +/- kappa = arcsin(X +/- Y)); for several it is the leading-order
    inversion of ``core.multiqubit_zz``. With ``num_boot > 0`` a 95% bootstrap interval over
    shots is added.
    """
    records = list(records)
    neighbors = sorted(records[0][1])
    s = np.array([r[0] for r in records], float)
    b = np.array([r[2] for r in records], float)
    c = {j: np.array([r[1][j] for r in records], float) for j in neighbors}

    theta0, kappa = _point_estimate(s, c, b, neighbors, v)
    out = {"theta0_hat": theta0, "kappa_hat": kappa}
    if num_boot:
        rng = np.random.default_rng(seed)
        boots_t, boots_k = [], {j: [] for j in neighbors}
        for _ in range(num_boot):
            idx = rng.integers(len(records), size=len(records))
            t, k = _point_estimate(s[idx], {j: c[j][idx] for j in neighbors}, b[idx], neighbors, v)
            boots_t.append(t)
            for j in neighbors:
                boots_k[j].append(k[j])

        def ci(a):
            lo, hi = np.percentile(a, [2.5, 97.5])
            return float(lo), float(hi)

        out["ci95"] = {"theta0": ci(boots_t), "kappa": {j: ci(boots_k[j]) for j in neighbors}}
    return out
