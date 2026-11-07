# Coherent-Error Recovery from Randomized-Compiling Labels

Numerical and real-hardware backbone for the result:

> **Randomized compiling (RC) does not destroy coherent-error information — it relocates it
> into the twirl labels, where it is recoverable.** This is a Fisher-information conservation law:
> for a coherent error `exp(-iθG)` twirled with a per-shot sign label `s`, the total Fisher
> information about `θ` in the joint distribution of (outcome `b`, label `s`) is conserved and
> equals the untwirled quantum Fisher information `F_0`, independent of the twirl. Standard RC
> marginalizes over `s` (`F_marg → 0` at full twirl); the missing information reappears exactly in
> the label–outcome term `E_b[F_{s|b}] = F_0 − F_marg`. Keeping labels recovers `F_0`, and under
> depth the recoverable information grows linearly (`~L`) while bounded coherent accumulation
> saturates.

**Paper:** K. Vyas, *Recovering Coherent Errors from Randomized-Compiling Labels*,
[arXiv:2607.26756](https://arxiv.org/abs/2607.26756). The full derivation — conventions,
equations, numeric oracles, and the depth theorem — is also in **`label_and_recover_math_notes.pdf`**.

**PennyLane implementation:** [`pennylane_impl/`](pennylane_impl/) re-implements the method as
PennyLane transforms (`pauli_twirl`, `sign_label_twirl`), checks the conservation law and the
depth theorem with exact autodiff and `qp.gradients.classical_fisher` / `quantum_fisher`, extends
it to two-qubit ZZ crosstalk, re-analyses the `ibm_marrakesh` records, and is cross-validated
against the numpy engine below. Start with the
[tutorial notebook](pennylane_impl/notebooks/labeled_rc_pennylane.ipynb).

## Dependencies

The reference engine (`core/`, `experiments/`, `tests/`) uses **numpy + matplotlib only** — no
external quantum libraries — so it can serve as an independent check. The live hardware scripts
need qiskit (`hardware/requirements.txt`); the PennyLane layer has its own
`pennylane_impl/requirements.txt`.

```bash
# one-time setup
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib
```

All commands below assume `.venv/bin/python`. (Substitute your own interpreter if numpy +
matplotlib are already available system-wide.)

## Repository layout

```
core/             state-vector engine + physics modules
  noise.py        device-representative noise profile: T1, T2, depolarizing (1q/2q),
                  asymmetric readout; density-matrix + trajectory modes
  error_map.py    multi-parameter coherent-error recovery, visibility-debiased
  qasm_loader.py  native OpenQASM 2.0 -> gate list, at the I/O boundary (no qiskit)
experiments/      exp1..exp16 — each regenerates one figure and prints its oracle check
tests/            oracle-backed verification (run these before trusting anything)
figures/          output PDFs/PNGs (publication quality) + corpus_manifest.json
hardware/         real-IBM-device experiment with owned per-layer labels (see below)
hardware_zz/      multi-qubit ZZ and drive-crosstalk hardware experiments (own README)
data_analysis/    coherent-counterpart analysis of a public IBM CB dataset (see below)
pennylane_impl/   PennyLane transforms, tests and tutorial notebook (own README)
label_and_recover_math_notes.pdf   the specification: math, conventions, oracles
```

## External data & circuits (not redistributed in this repo)

Two inputs used by some experiments are **third-party** and are deliberately **not** committed
here. Download them from their upstream sources and place them where the code expects:

| Path (git-ignored) | What it is | Get it from |
|---|---|---|
| `data/` | Chen et al. IBM `ibmq_montreal` cycle-benchmarking dataset (used by `data_analysis/`) | https://github.com/csenrui/Pauli_Learnability |
| `qasm/` | QASMBench circuits used by the corpus experiment (`exp9`, `core/qasm_loader.py`) | https://github.com/Qiskit/benchpress |

These directories are listed in `.gitignore`; the core physics, all oracles (O1–O5), and the
hardware experiment do not depend on them.

## Run the verification suite

```bash
.venv/bin/python tests/run_all.py             # every test, in order (no pytest needed)
```

or individually:

```bash
.venv/bin/python tests/test_gates.py              # Oracle O5 (gates, Pauli propagation)
.venv/bin/python tests/test_fisher_identities.py  # Oracle O4 (QFI methods agree, chain rule)
.venv/bin/python tests/test_conservation.py       # Oracle O1 (single-qubit conservation)
.venv/bin/python tests/test_depth_table.py        # Oracles O2, O3 (depth tables)
.venv/bin/python tests/test_noisy_saturation.py   # gamma_z=0 recovers O2
.venv/bin/python tests/test_two_qubit_tfim.py     # 2-qubit conservation + kernel
.venv/bin/python tests/test_estimator.py          # CRB saturation + recover-and-correct payoff
.venv/bin/python tests/test_noise.py              # noise module N1-N6
.venv/bin/python tests/test_noise_estimation.py   # robustness / unbiasedness under noise
.venv/bin/python tests/test_error_map.py          # coherent-error map recovery
.venv/bin/python tests/test_corpus.py             # QASMBench corpus generality (skips without qasm/)
.venv/bin/python tests/test_shotcost.py           # shot-cost vs robust phase estimation
.venv/bin/python tests/test_multiqubit_zz.py      # static ZZ crosstalk (exp11-13)
.venv/bin/python tests/test_drive_crosstalk.py    # drive-induced crosstalk, Oracles DC1-DC3
```

The PennyLane layer has its own suite (needs `pennylane_impl/requirements.txt`):

```bash
python -m pytest pennylane_impl/tests
```

Each test cross-checks code output against the hand-derived numeric oracles (O1–O5) documented
in `label_and_recover_math_notes.pdf`. If code disagrees with an oracle, the **code** is wrong.

## Regenerate every figure (one command each)

| Figure | Command | What it shows |
|---|---|---|
| Fig. 1 | `.venv/bin/python experiments/exp1_conservation.py` | Conservation / complementarity of Fisher information vs tailoring `T` |
| Fig. 2 | `.venv/bin/python experiments/exp2_depth_scaling.py` | No washout with depth: labeled `E[F_Q]` grows `~L/2`; coherent bounded; RC-marginal ≈ 0 |
| Fig. 3 | `.venv/bin/python experiments/exp3_noisy_saturation.py` | Noisy saturation: linear growth bends to a plateau vs dephasing `gamma_z` |
| Fig. 4 | `.venv/bin/python experiments/exp4_two_qubit_tfim.py` | Two-qubit TFIM: conservation residual ~1e-12 + silent-kernel directions |
| Fig. 5 | `.venv/bin/python experiments/exp5_recover_and_correct.py` | Recover-and-correct payoff: labeled-RC+correction beats plain RC |
| Fig. 6 | `.venv/bin/python experiments/exp6_robustness.py` | Robustness under the full noise profile: noise-aware recovery unbiased, marginal biased by SPAM offset |
| Fig. 7 | `.venv/bin/python experiments/exp7_saturation_full.py` | Depth saturation `L*(T2)` under the full device profile (T1+T2+depol) |
| Fig. 8 | `.venv/bin/python experiments/exp8_error_map.py` | Recovered coherent-error MAP: full vector of 1q + 2q errors, unbiased, + chip heatmap |
| Fig. 9 | `.venv/bin/python experiments/exp9_corpus.py` | Corpus generality across QFT/adder/QAOA/VQE/… (needs `qasm/`; see above) |
| Fig. 10 | `.venv/bin/python experiments/exp10_shotcost.py` | Shot-cost positioning: labeled-RC (free, rides on RC) vs robust phase estimation |

`exp11`–`exp16` regenerate the multi-qubit extension figures (two- and three-qubit ZZ
crosstalk, joint-vs-pairwise cost, drive-induced crosstalk, variance cost; `fig12`–`fig18`).

Or regenerate all at once:

```bash
.venv/bin/python experiments/run_all.py
```

Figures are written to `figures/` as both `.pdf` (vector) and `.png` (`dpi=300`).

## Real-hardware validation (`hardware/`)

`hardware/` runs the label-recovery theorem on a real IBM device with **owned per-layer labels**:
one circuit per twirl instance, recording the exact per-cycle sign vector, so no label is ever
lost. It recovers the injected coherent phase `θ*` from the label–outcome correlation, debiased by
a measured visibility. The offline dress-rehearsal (`hardware/run_all.py`, no qiskit/network)
reproduces the theory; the live path (`submit → retrieve → analyze`) requires qiskit + IBM Cloud
credentials placed in `hardware/.env` (git-ignored — never committed). Run on `ibm_marrakesh`
(IBM Heron r2): 1,620 circuits over θ* ∈ {0, 0.05, 0.1, 0.2, 0.3} rad and L ∈ {1, …, 32},
with IBM's own twirling and dynamical decoupling off; see the paper for the results. The
per-circuit records and the analysis of that run are in `hardware/online_data/`.

## Coherent-counterpart analysis of a public dataset (`data_analysis/`)

`data_analysis/` applies the program to real data — the coherent (off-diagonal) counterpart to the
diagonal Pauli-fidelity analysis of Chen et al. (*Nat. Commun.* 14, 52, 2023). It (a) reproduces
the published Pauli fidelities as a credibility anchor (`max |Δλ| = 3.0e-4`), and (b) runs a
label-correlation coherent estimator with a permutation-null and a positive control. **Finding:**
the dataset retains only the *net* twirl frame, so the depth-accumulating coherent signature the
theorem predicts is (correctly) **null**; a real but shallow, non-accumulating correlation
survives at `L=2`. This is a structural argument for a purpose-built RC experiment that logs
per-layer labels — which is what `hardware/` does. The raw dataset itself is not redistributed
here (see "External data & circuits").

## Reproducibility notes

- Oracle-backed tests are seed-independent (exact enumeration / analytics, not sampling).
- For depth `L ≤ 12` sign patterns are enumerated exactly (`2^L`); larger `L` uses Monte-Carlo
  with reported error bars.
- Every `expN` script prints `max|code − oracle|` and raises if it exceeds tolerance.
- Environment note: on Python 3.9 + numpy 2.0.2 with Apple Accelerate BLAS, spurious
  "…encountered in matmul" warnings are scoped away by `core/_compat.quiet_accelerate`; outputs
  are verified finite and results are exact.

## Citation

If you use this code, please cite the accompanying paper,
[arXiv:2607.26756](https://arxiv.org/abs/2607.26756), and this repository (`CITATION.cff`). When using the external inputs, please also cite their sources: Chen et al.,
*The learnability of Pauli noise*, Nat. Commun. 14, 52 (2023)
(https://github.com/csenrui/Pauli_Learnability), and QASMBench / Qiskit benchpress
(https://github.com/Qiskit/benchpress).

## License

Released under the MIT License — see `LICENSE`.
