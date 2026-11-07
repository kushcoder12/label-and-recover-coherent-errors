"""The two transforms: noiseless equivalence, label semantics and input checks."""

import numpy as np
import pennylane as qp
import pytest

from labeled_rc import pauli_twirl, sample_signs, sign_label_twirl, signal_tape_qnode


def _three_qubit_circuit(dev):
    @qp.qnode(dev)
    def circuit(a):
        qp.RX(a, wires=0)
        qp.RY(0.4, wires=1)
        qp.CNOT(wires=[0, 1])
        qp.CZ(wires=[1, 2])
        qp.RX(0.3, wires=2)
        qp.CNOT(wires=[2, 0])
        return qp.probs(wires=[0, 1, 2]), qp.expval(qp.PauliZ(0) @ qp.PauliX(2))

    return circuit


def test_pauli_twirl_noiseless_average_equals_original():
    dev = qp.device("default.qubit", wires=3)
    circuit = _three_qubit_circuit(dev)
    twirled = pauli_twirl(circuit, num_samples=16, seed=0)
    for out_tw, out in zip(twirled(0.7), circuit(0.7)):
        assert np.allclose(out_tw, out, atol=1e-12)


def test_pauli_twirl_every_instance_equals_original():
    dev = qp.device("default.qubit", wires=3)
    circuit = _three_qubit_circuit(dev)
    tape = qp.workflow.construct_tape(circuit)(0.7)
    tapes, _ = pauli_twirl(tape, num_samples=16, seed=1)
    reference = qp.execute([tape], dev)[0]
    for result in qp.execute(tapes, dev):
        for r, ref in zip(result, reference):
            assert np.allclose(r, ref, atol=1e-12)
    # the twirl actually inserted gates in at least one instance
    assert any(len(t.operations) > len(tape.operations) for t in tapes)


@pytest.mark.parametrize("L", [1, 3, 6])
def test_sign_label_twirl_gives_sin_theta_sum_s(L):
    dev = qp.device("default.qubit", wires=1)
    signs = sample_signs(12, L, seed=L)
    results = sign_label_twirl(signal_tape_qnode(dev, L), signs=signs)(0.23)
    expected = np.sin(0.23 * signs.sum(axis=1))
    assert results.shape == (12,)
    assert np.allclose(results, expected, atol=1e-12)


def test_sign_label_twirl_is_differentiable():
    dev = qp.device("default.qubit", wires=1)
    signs = np.array([[1, -1, -1], [1, 1, 1]])
    twirled = sign_label_twirl(signal_tape_qnode(dev, 3), signs=signs)
    from pennylane import numpy as pnp

    grad = qp.jacobian(twirled)(pnp.array(0.1, requires_grad=True))
    m = signs.sum(axis=1)
    assert np.allclose(grad, m * np.cos(0.1 * m), atol=1e-10)


def test_sign_label_twirl_rejects_bad_labels():
    dev = qp.device("default.qubit", wires=1)
    circuit = signal_tape_qnode(dev, 2)
    with pytest.raises(ValueError):
        sign_label_twirl(circuit, signs=np.array([[1, -1, 1]]))(0.1)  # 3 labels, 2 RZ
    with pytest.raises(ValueError):
        sign_label_twirl(circuit, signs=np.array([[1, 0]]))(0.1)  # 0 is not a label


def test_sample_signs_statistics():
    s = sample_signs(20000, 3, p_plus=0.8, seed=5)
    assert set(np.unique(s)) == {-1, 1}
    assert abs(np.mean(s) - 0.6) < 0.02  # mu = 2 p_plus - 1


def test_pauli_twirl_turns_coherent_error_growth_from_quadratic_to_linear():
    """Standard RC: a coherent ZZ over-rotation after every CNOT adds up quadratically
    with depth; after twirling it behaves like a Pauli channel and grows linearly."""
    eps = 0.08
    noise = qp.NoiseModel(
        {qp.noise.op_eq(qp.CNOT): lambda op, **kw: qp.IsingZZ(eps, wires=op.wires)}
    )
    dev = qp.device("default.mixed", wires=2)

    def make(k):
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            qp.Hadamard(1)
            for _ in range(k):
                qp.CNOT(wires=[0, 1])
                qp.CNOT(wires=[0, 1])
            qp.Hadamard(0)
            qp.Hadamard(1)
            return qp.probs(wires=[0, 1])

        return circuit

    def error(k, twirl):
        c = make(k)
        if twirl:
            c = pauli_twirl(c, num_samples=200, seed=1)
        return 1 - float(qp.add_noise(c, noise)()[0])

    coherent_ratio = error(8, False) / error(4, False)
    twirled_ratio = error(8, True) / error(4, True)
    assert 3.2 < coherent_ratio < 4.0  # ~4x per depth doubling: quadratic
    assert 1.8 < twirled_ratio < 2.3  # ~2x per depth doubling: linear
