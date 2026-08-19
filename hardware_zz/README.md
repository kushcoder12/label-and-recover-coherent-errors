# hardware_zz/ — Real-device recovery of static ZZ crosstalk (owned labels)

Extends `hardware/`'s single-qubit label-recovery run to a second, independent
mechanism: static ZZ coupling between a target qubit and its neighbor(s). Physics +
estimator reused from `core/multiqubit_zz.py` and `core/circuits.py` (validated
offline in `experiments/exp11-13` and `tests/test_multiqubit_zz.py`, Oracles MQ1-MQ3);
see `MULTIQUBIT_CROSSTALK_SPEC.md` and `MULTIQUBIT_FINDINGS.md` for the full derivation.

## The experiment

Per shot, on the target qubit and its neighbor(s) (physical qubits picked live from
`backend.coupling_map`, never assumed):

1. `H` on the target → |+⟩
2. `X` on each neighbor iff its label `c_j = -1` (real state prep, not virtual)
3. `RZ(s1·θ0)` on the target (virtual/exact — the SAME kind of known, owned knob as
   the single-qubit run's `θ*`; `s1` is qubit 1's own RC sign)
4. `delay(1 µs)` on every qubit in use — RZ is virtual, so without an idle window the
   REAL static ZZ coupling has no time to act at all
5. `SDG`, `H` on the target (rotates `Y`→`Z`), then measure

Ideal statistics: `⟨Y⟩ = sin(s1·θ0 + Σⱼ κ_1j·cⱼ)` — the same closed form validated by
`core/circuits.multi_qubit_zz_probe` in simulation. `θ0` is **injected by us**, so
recovering it is a genuine positive control (diagonal = success, `θ0=0` → ~0). `κ_1j`
is the device's **real** coupling accumulated over the delay — there is no ground
truth to check it against; it is reported with a bootstrap 95% CI as a genuine
characterization result (Sec 6 of the spec).

Two topologies, in the same run: **pair** (target + 1 neighbor) and **star** (target +
2 neighbors, no neighbor-neighbor coupling required), so the joint multi-neighbor
recovery advantage (Fig 13/14) is also demonstrated on real hardware.

**Sweep** (`config.py`): `θ0 ∈ {0, 0.05, 0.10, 0.20}` rad, 30 instances/point (pair) /
40 instances/point (star), 2000 shots/circuit, plus 20 visibility-calibration repeats
per topology. 320 circuits total — deliberately smaller than the single-qubit run's
1620 (that run used ~21 min of QPU time; this keeps well inside the free-tier budget).

## A real hardware caveat the offline simulation cannot test

The neighbor's state-prep pulse (step 2) is itself a real drive event, which could
leak a small drive-crosstalk confound onto the target — the mechanism explicitly
out of scope for this work (`MULTIQUBIT_CROSSTALK_SPEC.md` §0). The 1 µs delay before
the probe is a judgment call meant to let that transient decay while the *static* ZZ
term (which does not decay) is still comfortably resolvable at typical reported IBM ZZ
rates. This has not been independently verified against the confound on hardware —
flagging it honestly rather than claiming it is ruled out.

## Files

| file | role |
|---|---|
| `config.py` | sweep grid, runtime options, paths; credentials read from `hardware/.env` (shared, not duplicated) |
| `build_circuits.py` | owned-label specs (numpy) + lazy-qiskit builder + transpiler-integrity assert (RZ **and** delay survive) |
| `local_sim.py` | offline noisy simulator — reuses `core.circuits.multi_qubit_zz_probe` directly, not a re-derivation |
| `submit.py` | **live**: pick coupled qubit groups from `backend.coupling_map` → transpile (opt_level 0, IBM twirling OFF, DD OFF) → assert integrity → submit |
| `retrieve.py` | **live**: fetch results → raw JSON before any analysis (resumable) |
| `analyze.py` | `core.multiqubit_zz.joint_correlation_estimator` + bootstrap CIs → Figure 15 |
| `run_all.py` | offline dry-run (default) or `--live` / `--live-full` |
| `tests/test_offline_pipeline.py` | offline correctness (no qiskit/network) |

## Quick start

```bash
python run_all.py                    # offline dress rehearsal (do this first)
python tests/test_offline_pipeline.py

python submit.py --validate          # dry-run the live path, no QPU spent
python submit.py                     # build + transpile + integrity asserts + submit
python wait_jobs.py                  # (optional) block until jobs are terminal
python retrieve.py                   # fetch results (resumable)
python analyze.py --live             # estimator + Figure 15
```

Or `python run_all.py --live-full` (submit, poll, retrieve, analyze in one go). The
live run writes to `online_data/` + `online_figures/`, kept separate from the offline
`data/` + `figures/`.

Needs the same OpenSSL 1.1.1+/3.x Python as `hardware/` (see `hardware/requirements.txt`
and `hardware/README.md`); this repo already has `.venv312` set up with
`qiskit>=1.2` + `qiskit-ibm-runtime>=0.30` for exactly this reason.

## Guardrails (enforced in code)

- IBM's own gate/measurement twirling **OFF**; dynamical decoupling **OFF** (it would
  scramble the idle window the static ZZ coupling needs).
- `optimization_level=0` + barriers; the transpiler-integrity assert checks the
  injected RZ **and** the idle delay on the target qubit both survive — else the run
  is invalid and we raise, we don't plot.
- Qubit groups are picked live from `backend.coupling_map` (never assumed connected),
  scored by readout error + T2; calibration is snapshotted into `run_manifest.json`.
- `submit → retrieve → analyze` are separate, resumable steps.
- `θ0` (injected) is reported as a positive control; `κ_1j` (real) is reported with a
  bootstrap CI and explicitly **not** claimed as validated against a known value.
