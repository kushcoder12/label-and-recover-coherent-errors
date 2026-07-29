# REAL_DATA_FINDINGS.md — Honest analysis of the IBM Pauli-learnability dataset

**Data source.** Chen, Wang, Wang, Wang, Zhou, Yao & Yuan (and co-authors),
*"The learnability of Pauli noise,"* **Nature Communications 14, 52 (2023)** —
cycle-benchmarking (CB) experiment on `ibmq_montreal`, 2 target qubits, taken
2022-03-23 (`token 8530634712`). Files analysed: the metadata file, the `_full`
file (per-circuit counts + shot-by-shot memory), and the `_result` file (the
paper's own fitted Pauli fidelities). This work is the **coherent (off-diagonal)**
counterpart to that paper's **diagonal (stochastic)** Pauli-fidelity analysis.

**Bottom line (which of the spec §4 outcomes occurred).**
> **Case 3 (structural) + Case 2 (null on the depth signature).** Our label-
> correlation estimator — validated two independent ways — finds a **genuine but
> shallow, non-accumulating** label–outcome correlation and **no depth-growing
> coherent signal**. The per-cycle coherent information our theorem shows is
> recoverable requires **per-layer twirl labels, which this dataset does not
> retain** (only the net frame survives). This motivates a purpose-built RC
> experiment that logs per-layer labels.

No coherent-error amplitude map is claimed, because the depth-growth signature
that would justify one is absent (see §3). Every number below carries an error
bar or a permutation-null *p*-value.

---

## 1. Data audit (`outputs/data_audit.json`)

| Fact | Value |
|---|---|
| Depths present (`Lrange`) | **[2, 4, 8, 16, 32, 64, 128]** — 7 depths (multi-depth ✔, SPAM-removable) |
| Randomizations per (target, depth) | **60** (`C=60`) |
| Shots per circuit | 1000 |
| CB target channels | 9 weight-2 Paulis `XX…ZZ` (also `int_cb`, `intc_cb`) |
| Shot-by-shot `memory` present | **Yes**, all circuits |
| What the per-circuit label is | **Net twirl frame** (a single 2-qubit Clifford tableau: 2 signed Z-stabilizers + 2 signed X-destabilizers). **NOT** the per-layer random Paulis. |
| Per-layer Paulis recoverable? | **No** — they live inside the qiskit `QuantumCircuit`, which deserializes to an opaque stub under our numpy-only `SafeUnpickler` (qiskit is not a dependency). |

**Label decoding (derived, not assumed, and validated at 100%).** For target Pauli
`AB`, the ideal per-qubit measured eigenvalue sign is read off the frame as:
X-letter → destabilizer sign; Z-letter → stabilizer sign; Y-letter → their product
(since `Y = iXZ`). At the shallowest depth the frame-predicted ideal outcome equals
the most-probable measured bitstring for **100.0 %** of circuits across all 9 targets
(`frame_prediction_check`). This certifies the extraction, the sign convention, and
the big-endian bit mapping.

---

## 2. Credibility anchor — reproducing the published CB fidelities (`pauli_fidelities.json`)

We fit the SPAM-free decay `f_O(L) = A · λ_O^L` for every non-identity 2-qubit Pauli,
pooling all randomizations (and, for weight-1 Paulis, all targets that measure them),
and compare `λ_O` to the paper's own stored values.

- **max |λ_fit − λ_paper| = 3.0 × 10⁻⁴**, **rms = 9.0 × 10⁻⁵** over 15 Paulis.
- The 9 weight-2 Paulis match to **< 1 × 10⁻⁵**; weight-1 Paulis to ≤ 3 × 10⁻⁴.

This is the proof that our data handling + fit are correct. See
`figures/fig_cb_validation.*` (a: our λ vs published, on the diagonal; b: the
exponential decays). The fitted `λ_O` are also the **visibilities `v`** used for
debiasing in §3.

---

## 3. Our method — the label-correlation coherent estimator (`coherent_recovery.json`)

**Estimator.** For candidate coherent generator `H` (a 2-qubit Pauli) and target `P`,
at each depth `L`,
```
C_H(P, L) = Cov_i( σ_H(frame_i),  O_i )
```
over the 60 twirl instances `i`, where `σ_H(frame) = +1` if the net frame commutes
with `H` else `−1`, and `O_i` is the sign-corrected survival of the target Pauli.
Our theorem's coherent-error signature is **growth of `|C|` with depth** (per-cycle
accumulation). 135 `(H, target)` combinations are scanned; significance is assessed
against a **frame-permutation null** (500 permutations, marginals preserved) that
absorbs the multiple-comparisons burden.

**Result 1 — no depth-growing coherent signal (Case 2 on our signature).**
Weighted slope of `|C(L)|` vs `L`, max over all 135 tests:
- observed max `growth_z` = **2.49**; permutation-null 95th percentile = **2.89**;
  **p = 0.18**. → *No coherent accumulation with depth above chance.*

**Result 2 — a real but shallow, non-accumulating correlation (Case 3, structural).**
`|C|` at the shallowest depth `L=2`, max over 135 tests:
- observed max `|z|` = **7.71**; null 95th percentile = **3.81**; **p = 0.002**.
- The strongest shallow correlations (`|C|≈0.02–0.025`) involve weight-1 generators
  and same-letter targets (e.g. `H=IZ`→`ZZ,YZ`; `H=IX`→`YX,XX`; `H=YI`→`YY`).
- These correlations **decay toward zero with depth** (see `fig_label_correlation.*`
  panel a), the *opposite* of accumulation.

**Why shallow-only, physically.** The retained label is the **net** frame — the
*product* of the `L` per-cycle random Paulis. At `L=2` it still tracks the individual
cycles, so a coherent-type effect is visible; as `L` grows the product decorrelates
from any single cycle's coherent contribution, so the correlation washes out with the
survival envelope instead of accumulating. Realizing the theorem's linear-in-depth
recoverable information requires **per-layer** labels, which are absent here.

**Honesty caveat on the shallow correlation.** At shallow depth this correlation
**cannot be cleanly attributed to gate-coherent errors** versus **readout asymmetry**:
an asymmetric readout also makes the sign-corrected survival frame-dependent (the
sign correction uses the frame), producing a `σ_H`-correlated `C`. Disentangling the
two would again require per-layer labels. We therefore do **not** map these into
coherent generator amplitudes.

**Positive control — the estimator is correct (`fig_label_correlation.*` panel b).**
Injecting a known label-correlated shift `θ_inj` into the real data recovers
`C ≈ θ_inj`:

| `θ_inj` | recovered `Ĉ` |
|---|---|
| 0.00 | −0.0025 ± 0.0018 |
| 0.02 | +0.0175 ± 0.0018 |
| 0.05 | +0.0473 ± 0.0019 |

So the absence of a depth-growing signal on real data is a property of the **data /
label structure**, not of the estimator.

---

## 4. What would be needed to claim a coherent map

- **Per-layer twirl labels** logged per circuit (the accumulating signature `Σ_ℓ`
  in our depth theorem is invisible to the net frame).
- A readout-asymmetry control (or a symmetric-readout / RC-of-readout design) to
  separate gate-coherent from SPAM-coherent contributions at shallow depth.
- More randomizations than `C=60` to push the per-depth covariance floor
  (currently ≈ `median se ≈ 0.004`, 1σ) below the target coherent amplitudes.

## 5. Guardrails honored (spec §5)

No single-depth "rates"; no "cross-target spread = coherence"; no `sqrt(|Δγ|/2)`
heuristic; every physics number has a bootstrap error bar or a permutation-null
*p*-value; §2 was validated against the published paper **before** trusting §3; and
because only the net frame (not per-layer labels) is available, we explicitly **stop**
short of manufacturing a coherent map from insufficient data.

---

*Reproduce:* `python data_analysis/code/audit.py` →
`python data_analysis/code/cb_validation.py` →
`python data_analysis/code/label_correlation.py` (or `run_all.py`). Cite Chen et al.,
Nat. Commun. 14, 52 (2023).
