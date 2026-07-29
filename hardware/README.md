# hardware/ — Real-device label recovery of coherent error (owned labels)

Proves the label-recovery theorem on real IBM hardware. Unlike the public CB dataset
(which kept only the **net** twirl frame, so the coherent signal decorrelated with depth
— proven null), here **we generate one circuit per twirl instance and record the exact
per-cycle sign vector ourselves**, so every per-layer label is *owned* and never lost.

Physics + estimator reused from the project `core/` engine.

## The experiment (Variant A, single qubit)

For each depth `L`, twirl instance with sign vector `s = (s₁…s_L)`, and injected phase `θ*`:

1. `H` → |+⟩
2. `RZ(sℓ·θ*)` for `ℓ=1…L` (RZ is virtual/exact on IBM → clean, known coherent error)
3. `SDG` then `H` (rotates Y→Z, i.e. measures ⟨Y⟩)
4. `measure`

Ideal statistics: `⟨Y⟩ = sin(θ*·Σℓ sℓ)`. The label–outcome correlation
`C = ⟨(Σs)·⟨Y⟩⟩` recovers `θ*`; the device's T2/readout only *shrink* it by a
visibility `v`, removed by debiasing. Recovering `θ̂ = arcsin(C/v)` (proper 1-parameter
fit at `L>1`) reproduces the single-qubit theory and Fig 2 on real silicon.

**Sweep** (`config.py`): `θ* ∈ {0, 0.05, 0.1, 0.2, 0.3}` rad (0.0 = null control),
`L ∈ {1,2,4,8,16,32}`, 50 twirl instances/point, 3000 shots/circuit, plus per-depth
visibility-calibration circuits. ≈1620 circuits, batched into as few jobs as possible.

## Files

| file | role |
|---|---|
| `config.py` | sweep grid, runtime options, paths, `.env` loader (credentials from env only) |
| `build_circuits.py` | owned-label circuit **specs** (numpy) + lazy-qiskit builder + transpiler-integrity assert (§2.3) |
| `local_sim.py` | offline noisy statevector/DM simulator (reuses `core/` + `core.noise`) — the dress rehearsal engine |
| `submit.py` | **live**: build → transpile (opt_level 0, IBM-twirling OFF) → assert integrity → submit; writes `job_ids.json` immediately |
| `retrieve.py` | **live**: fetch job results → write raw JSON *before* any analysis (resumable) |
| `analyze.py` | label-correlation estimator + bootstrap CIs → figures R1, R2 |
| `run_all.py` | one entry point: offline dry-run (default) or `--live` / `--live-full` |
| `tests/test_offline_pipeline.py` | offline correctness (no qiskit/network) |

## Quick start

### 1. Offline dress rehearsal (no qiskit, no network — do this first)

```bash
python run_all.py
```

Builds all circuits, simulates them on the local noisy sim, recovers `θ*`, and writes
`figures/figR1_*` and `figures/figR2_*`. Prints the positive-control checks; the run only
"passes" if `θ̂` tracks `θ*` and the `θ*=0` null recovers ~0. **This is the guarantee that
the hardware code is correct before spending QPU time (§6).**

Run the tests:

```bash
python tests/test_offline_pipeline.py
```

### 2. Live hardware run (with credentials)

Needs a Python built against **OpenSSL 1.1.1+/3.x** (the macOS system Python's LibreSSL 2.8.3
is rejected by IBM Cloud with `TLSV1_ALERT_INTERNAL_ERROR`; check `python -c "import ssl;
print(ssl.OPENSSL_VERSION)"`, and see `requirements.txt`). Install the deps, put your
credentials in `hardware/.env` (copy `.env.example`), then:

```bash
python submit.py        # build + transpile + integrity asserts + submit
python submit.py --validate   # (optional) dry-run the ENTIRE live path against the backend,
                              # no QPU spent — verifies transpile/integrity/options first
# ... wait for the jobs to finish (queue wait is wall-clock, not billed) ...
python wait_jobs.py     # (optional) blocks, polling until all jobs are terminal
python retrieve.py      # fetch results (resumable; re-run if not ready)
python analyze.py       # estimator + figures R1, R2
```

Or, in one go: `python run_all.py --live-full` (submits, polls to completion, retrieves,
analyzes). **The live run writes to `online_data/` and `online_figures/`** (kept separate
from the offline `data/`+`figures/` so real-device results never overwrite the rehearsal).

## Credentials

`hardware/.env` (git-ignored, never committed) holds:

```
IBM_TOKEN=<your 44-char API key>
IBM_CRN=<your IBM Cloud CRN>
```

Get both from <https://quantum.cloud.ibm.com>. They are read from the environment only —
never hard-coded. The offline dress rehearsal does **not** need them.

## Regenerating each figure

```bash
python analyze.py                 # Fig R1 (recovered vs injected) + Fig R2 (correlation vs depth)
```

(both figures are written as `.png` @ 300 dpi and vector `.pdf` into `figures/`.)

## Guardrails (enforced in code)

- IBM's own gate/measurement twirling is turned **OFF** (it would scramble our owned labels).
- Dynamical decoupling / ZNE / PEC **OFF** — we *measure* coherent error, we don't suppress it.
- `optimization_level=0` + barriers around every injected layer; after transpiling we
  **assert** the injected `L` phase layers survive (the #1 failure mode) — else the run is
  invalid and we raise, we don't plot.
- A good physical qubit is pinned via `initial_layout`; T1/T2/readout/gate error are
  snapshotted into `data/run_manifest.json`.
- `submit → retrieve → analyze` are separate, resumable steps; raw data is written before
  any analysis, so nothing is lost if a job expires.
- Every recovered amplitude carries a bootstrap 95% CI; the `θ*=0` null must recover ~0.
