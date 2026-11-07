# PennyLane implementation

Randomized compiling with kept labels, as PennyLane transforms. This folder implements the
method of [arXiv:2607.26756](https://arxiv.org/abs/2607.26756) in PennyLane and checks it against
the repository's numpy reference engine (`../core/`), the paper's numeric oracles, and the
`ibm_marrakesh` hardware records.

**Tutorial:** [`notebooks/labeled_rc_pennylane.ipynb`](notebooks/labeled_rc_pennylane.ipynb)
(executed, with outputs).

## What is here

| Module | What it provides |
|---|---|
| `labeled_rc/transforms.py` | `pauli_twirl`: standard Pauli twirling (randomized compiling) of CNOT/CZ as a `qp.transform`. `sign_label_twirl`: the paper's label-keeping twirl of coherent Z-error locations; labels are supplied by the caller and never averaged away |
| `labeled_rc/frames.py` | Twirl frames P' = G P G† computed by conjugation for any two-qubit Clifford |
| `labeled_rc/circuits.py` | The paper's hardware circuit, its visibility reference, and a label-in-ancilla circuit for exact Fisher information |
| `labeled_rc/estimator.py` | Label correlation, the least-squares estimator used in the hardware analysis, bootstrap intervals |
| `labeled_rc/fisher.py` | F_joint = F_marg + Δ with derivatives from `qp.jacobian` (no finite differences) |
| `labeled_rc/depth.py` | The depth theorem: labeled E[F_Q] ~ L/2 vs bounded coherent F_Q, computed with `qp.gradients.quantum_fisher` |
| `labeled_rc/crosstalk.py` | Static ZZ crosstalk: `RZ` + `IsingZZ` error block, owned neighbour labels, shot sampling and the joint-correlation estimator |
| `labeled_rc/hardware_data.py` | Re-analysis of the IBM hardware records (`hardware/online_data/raw_results.json`) |

As of PennyLane 0.45.1 (and `main`, November 2026) there is no built-in Pauli-twirling or
randomized-compiling transform.

## Quickstart

```python
import numpy as np
import pennylane as qp
import labeled_rc as lrc

dev = qp.device("default.qubit", wires=1)
L, theta = 4, 0.1
signs = lrc.sample_signs(num_samples=50, num_layers=L, seed=0)      # owned labels

y = lrc.sign_label_twirl(lrc.signal_tape_qnode(dev, L), signs=signs)(theta)  # one <Y> per label vector
print(lrc.marginal_mean(y))                          # standard RC: -0.047 here; -> 0 with more instances
print(lrc.fit_theta(signs.sum(axis=1), y, v=1.0))    # keep the labels: 0.1000000
```

## Tests

```bash
python -m pip install -r requirements.txt
python -m pytest tests          # run from this folder (or: pytest pennylane_impl/tests from the root)
```

The suite (105 tests) checks:

- **Frames:** all 16 frames of CNOT and CZ leave the gate unchanged; non-Cliffords are rejected.
- **Transforms:** twirled circuits equal the original without noise; labels give
  ⟨Y⟩ = sin(θ Σ s); results are differentiable; standard RC turns quadratic coherent-error growth
  into linear growth.
- **Conservation law:** F_joint = F_marg + Δ = F₀ to 1e-10 by autodiff, and again with
  `qp.gradients.classical_fisher` (joint distribution) against `qp.gradients.quantum_fisher`
  (untwirled error); the exact finite-θ marginal μ² cos²θ / (1 − μ² sin²θ); the paper's closed
  forms at θ = 0.
- **Depth theorem:** `qp.gradients.quantum_fisher` reproduces Oracles O2 and O3, the closed forms
  Σ sin²(lα) and (Σ sin(lα))², and `experiments/exp2_depth_scaling.py` at partial twirl.
- **ZZ crosstalk:** the PennyLane probe equals `core.circuits.multi_qubit_zz_probe` for one to
  three neighbours (1e-12); recovery from sampled shots agrees with
  `core.multiqubit_zz.joint_correlation_estimator`; under noise the visibility-debiased estimate
  covers the true values.
- **Noise:** exact recovery on `default.mixed` with phase damping, depolarizing and readout noise
  from `qp.NoiseModel`; finite-shot recovery inside a 95% bootstrap interval.
- **Hardware:** re-analysing the `ibm_marrakesh` records reproduces every point estimate,
  visibility and correlation in `hardware/online_data/analysis.json`.
- **Cross-checks with the reference engine:** per-instance ⟨Y⟩ against `core/` (1e-12), autodiff
  Fisher terms against `core/fisher.py`, and the estimator against `hardware/analyze.py`.

## Notes

- Readout error is attached through `qp.NoiseModel(meas_map=...)` because `default.mixed`'s
  `readout_prob` argument raises `TypeError: 'float' object is not iterable` when a circuit is
  executed (PennyLane 0.45.1; reported upstream as
  [PennyLaneAI/pennylane#9612](https://github.com/PennyLaneAI/pennylane/issues/9612)).
- In `crosstalk.py`, noise channels are attached after the error block. Phase damping and
  depolarizing noise commute with the X dressing there, so they only rescale the signal by the
  measured visibility; noise *between* the RZ and IsingZZ gates would not.
