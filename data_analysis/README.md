# data_analysis/ — Real-hardware analysis of the IBM Pauli-learnability dataset

Honest, error-barred analysis of the Chen et al. (Nat. Commun. 14, 52, 2023)
cycle-benchmarking dataset (`ibmq_montreal`, 2 qubits), implementing the
Coherent-Error Recovery from Randomized-Compiling Labels program on real data.

**Read `REAL_DATA_FINDINGS.md` for the results.**

## Layout
```
data_analysis/
  code/
    cb_loader.py          # SafeUnpickler (numpy-only, no qiskit) + compact extraction
    audit.py              # §1 data audit; builds cache/extracted.pkl
    cb_validation.py      # §2 reproduce published Pauli fidelities (credibility anchor)
    label_correlation.py  # §3 label-correlation coherent estimator + permutation null
    run_all.py            # runs 1->2->3
  cache/extracted.pkl     # compact per-target records (built once from the 261 MB file)
  outputs/                # data_audit.json, pauli_fidelities.json, coherent_recovery.json
  figures/                # fig_cb_validation.*, fig_label_correlation.*
  REAL_DATA_FINDINGS.md   # the write-up (which §4 outcome occurred, with numbers)
```

## Reproduce
```bash
# from the repo root, using the project venv (numpy + matplotlib only)
.venv/bin/python data_analysis/code/run_all.py
```
The raw data lives in `../data/` (not modified). The first run parses the 261 MB
`_full` pickle once and caches a compact extract; later runs reuse `cache/extracted.pkl`.

## Key results (see FINDINGS for details + error bars)
- **Extraction validated:** frame-predicted ideal outcome matches the data at 100% (audit).
- **Credibility anchor:** our multi-depth CB fidelities match the paper to
  **max |Δλ| = 3.0e-4** (weight-2 Paulis < 1e-5).
- **Coherent estimator:** a real but **shallow, non-accumulating** label–outcome
  correlation (`p=0.002` vs permutation null); **no depth-growing** coherent signal
  (`p=0.18`). A positive control shows the estimator recovers an injected signal.
- **Outcome:** structural (Case 3) — the net-frame label this dataset retains cannot
  realize the per-cycle depth-accumulation; per-layer labels are required.

Data source: Chen et al., *The learnability of Pauli noise*, Nature Communications
14, 52 (2023).
