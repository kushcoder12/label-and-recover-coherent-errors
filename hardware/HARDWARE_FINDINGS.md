# HARDWARE FINDINGS — real-device label recovery (owned labels)

Variant A (single qubit). Physics/estimator reuse the
project `core/` engine. This file has two parts: the **offline dress rehearsal** (done, in
this sandbox) and the **live hardware run** (to be filled in by the student after they run
`submit.py → retrieve.py → analyze.py` with their credentials).

---

## Part 1 — Offline dress rehearsal (§6)  ✅ PASS

Ran the *entire* pipeline (build → simulate → analyze → figures) on a local noisy
statevector/density-matrix simulator built from `core/` + `core.noise` (numpy only; no
qiskit, no network). This is the guarantee that the hardware code path is correct before
any QPU time is spent.

- **Circuits:** 1620 (1500 signal + 120 visibility-calibration), grid
  `θ* ∈ {0, 0.05, 0.1, 0.2, 0.3}` rad, `L ∈ {1,2,4,8,16,32}`, 50 twirl instances/point,
  3000 shots/circuit. Every circuit owns its exact per-cycle sign vector.
- **Representative offline noise:** T1=180 µs, T2=110 µs, single-qubit gate error 3e-4,
  asymmetric readout e01=0.012 / e10=0.022. This yields a visibility
  `v(L) ≈ 0.964 → 0.953` from L=1 to L=32 (gentle decay: RZ is ~virtual, so depth adds
  only a small residual idle — the T2 "ceiling").

**Result — recovery is unbiased and on the diagonal.** With `θ̂ = argmin_θ Σᵢ (⟨Y⟩ᵢ −
v·sin(θ·Σsᵢ))²` (reduces to `arcsin(C/v)` at L=1):

| θ* (rad) | θ̂ at L=4 | θ̂ at L=16 | θ̂ at L=32 |
|---|---|---|---|
| 0.00 | −0.0027 | −0.0010 | +0.0007 |
| 0.05 | +0.0516 | +0.0478 | +0.0491 |
| 0.10 | +0.0999 | +0.0993 | +0.0979 |
| 0.20 | +0.1986 | +0.1994 | +0.1981 |
| 0.30 | +0.2956 | +0.3000 | +0.2993 |

- **Diagonal recovery:** `max|θ̂ − θ*|` across all (θ*, L) with θ*>0 is ≈ 0.006 rad,
  at the shot-noise floor. Labeled-RC recovery sits on the ideal diagonal at every depth
  (Fig R1); the **marginal** (labels discarded) estimate is flat at ~0 with large CIs —
  the standard-RC blind spot.
- **Null control (θ*=0):** all six depths recover `|θ̂| < 0.01` rad (the physically-
  negligible shot+readout floor). 1/6 of the per-point 95% CIs excludes zero — exactly the
  ~26% chance expected for six independent CIs (multiple comparisons); the readout
  asymmetry produces a common-mode ⟨Y⟩ offset that is *uncorrelated with the labels*, so it
  does **not** bias `θ̂`.
- **Depth growth (Fig R2, θ*=0.2):** the label–outcome correlation `C(L)=⟨(Σs)⟨Y⟩⟩` rises
  monotonically with depth and tracks the exact model `v(L)·E[(Σs)sin(θ*Σs)]` within CIs,
  while the marginal stays flat at 0. This is the real-device analog of Fig 2: **the
  coherent information does not wash out with depth once the labels are owned.**

Positive-control gate in `run_all.py`: **DRESS REHEARSAL PASS — safe to run live.**

### Figures
- `figures/figR1_recovered_vs_injected.{png,pdf}` — recovered `θ̂` vs injected `θ*`
  (diagonal = success), marginal flat at ~0. The money figure (real-device Fig 8 analog).
- `figures/figR2_correlation_vs_depth.{png,pdf}` — `C(L)` grows with depth; marginal flat;
  exact-model reference overlaid. Real-device Fig 2 analog.

---

## Part 2 — Live hardware run  ✅ DONE (ibm_marrakesh, 2026-07-21)

Ran the full sweep on real IBM hardware. Data + figures live in `online_data/` and
`online_figures/` (kept separate from the offline rehearsal). Regenerate the analysis with
`python analyze.py --live`.

- **Backend:** `ibm_marrakesh` (Heron, 156 qubits).  **Physical qubit:** 2 (auto-selected:
  low readout error + high T2).  **Date (UTC):** 2026-07-21 21:39.
- **Calibration at run time** (`online_data/run_manifest.json`): T1 = 279.6 µs,
  T2 = 404.7 µs, readout error = 0.281 %, single-qubit gate error = 1.29e-4.
- **Sweep:** θ* ∈ {0, 0.05, 0.1, 0.2, 0.3} rad, L ∈ {1,2,4,8,16,32}, 50 twirl instances/point,
  3000 shots/circuit → 1620 circuits in 6 jobs.
- **Total QPU charge time:** ≈ **1286 s (~21.4 min)** (238 s per 300-circuit job; the 120-circuit
  viscal job 96 s). **Note:** this *exceeded* the nominal 10-min Open-Plan window — 3000 shots ×
  1620 circuits is a lot of shot-sequencing. The jobs completed (the instance had capacity), but
  a repeat should cut shots (e.g. 1000) and/or instances to stay under 10 min. See "budget" below.

- **Guardrail verification (all TRUE — run is valid, §2):**
  - [x] IBM gate/measurement twirling **OFF** (`options.twirling.enable_gates=False`,
        `enable_measure=False`) — recorded in `run_manifest.json`.
  - [x] Dynamical decoupling / resilience **OFF** (`dynamical_decoupling.enable=False`,
        `resilience_level=0`).
  - [x] Transpiler-integrity assert **passed for all 1620 circuits** (`optimization_level=0`,
        each transpiled circuit still carried its `L` injected phase layers, checked by barrier
        count == L+2 and injected-rz count == L).

### Result — coherent phase recovered from owned labels on real silicon (headline)

Depth-averaged recovered `θ̂` (mean over the six depths) vs injected `θ*`:

| θ* (rad) | mean θ̂ (6 depths) | θ̂ @ L=4 | θ̂ @ L=16 | θ̂ @ L=32 |
|---|---|---|---|---|
| 0.00 (null) | +0.0002 | −0.0030 | −0.0010 | −0.0000 |
| 0.05 | +0.0493 | +0.0502 | +0.0491 | +0.0493 |
| 0.10 | +0.0996 | +0.0963 | +0.1012 | +0.0990 |
| 0.20 | +0.1985 | +0.1988 | +0.1987 | +0.1976 |
| 0.30 | +0.2978 | +0.2937 | +0.2996 | +0.3005 |

- **Positive control (headline):** labeled-RC recovery lies on the ideal diagonal at every
  depth. `max |θ̂ − θ*| = 0.0063 rad` across all (θ*>0, L) — at the shot-noise floor. The
  **marginal** (labels discarded) estimate is flat at ~0 with large CIs (Fig R1) — the
  standard-RC blind spot, confirmed on hardware.
- **Null control:** θ*=0 recovers `|θ̂| < 0.005 rad` at all six depths (< NULL_TOL = 0.01).
- **Visibility:** v(L) ≈ **0.993**, essentially flat from L=1 to L=32 — as predicted, since
  RZ is virtual so depth adds negligible decoherence; the excellent qubit (0.28 % readout)
  keeps the signal high, and debiasing by v recovers θ* exactly.
- **Depth growth (Fig R2, θ*=0.2):** the label–outcome correlation C(L)=⟨(Σs)⟨Y⟩⟩ rises
  monotonically with depth and tracks the exact model v(L)·E[(Σs)sin(θ*Σs)] within CIs, while
  the marginal stays flat at 0. **The coherent information does not wash out with depth once
  the per-layer labels are owned** — the real-device confirmation of Fig 2 and the fix for the
  public-CB null (which kept only the net frame).

### Figures (from hardware data)
- `online_figures/figR1_recovered_vs_injected.{png,pdf}` — recovered vs injected (diagonal =
  success), marginal flat at ~0. The money figure.
- `online_figures/figR2_correlation_vs_depth.{png,pdf}` — C(L) grows with depth, tracks the
  model; marginal flat.

### Budget note for a repeat run
The nominal free Open-Plan window is 10 min QPU. This run used ~21 min because charge time
≈ shots × circuits × per-shot sequencing. To stay under 10 min next time: drop to 1000 shots
(the CIs here are already tight at 0.28 % readout) and/or 30 instances/point — that would be
~7 min while keeping the recovery well-resolved.

### Honesty checklist (§6) — all satisfied
- Positive control reported first (θ* recovered on the diagonal; θ*=0 → ~0).
- Every amplitude carries a bootstrap 95% CI (see `python analyze.py --live`).
- Guardrails (IBM-twirling-off, transpiler integrity) verified — run is valid.

---

*Backend: IBM `ibm_marrakesh` via IBM Quantum, 2026-07-21. Data: `online_data/raw_results.json`,
`online_data/run_manifest.json`; analysis `online_data/analysis.json`.*
