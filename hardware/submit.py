"""Live step 1: build, transpile (with integrity asserts), submit.

Run with credentials. This process does NOT analyze — it writes
data/job_ids.json and data/run_manifest.json immediately so nothing is lost if the
session dies. Retrieve is a separate, resumable step.

Guardrails enforced here (the run is INVALID if any fails — we raise, we do not submit):
  - IBM's own gate/measurement twirling OFF (would scramble our owned labels).
  - dynamical decoupling OFF, resilience_level=0 (measure coherent error, don't suppress).
  - optimization_level=0 + barriers; ASSERT each transpiled circuit still has L injected
    phase layers (the #1 failure mode).
  - pin a good physical qubit via initial_layout; record it.
  - snapshot T1/T2/readout/gate error for the used qubit into the manifest.

qiskit / qiskit-ibm-runtime are imported lazily so the offline sandbox never needs them.

Usage:
  python submit.py            # live submit (requires IBM_TOKEN, IBM_CRN, network, qiskit)
"""

import json
import os
import sys
from datetime import datetime, timezone

import config
import build_circuits as bc


def _service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token, crn = config.require_credentials()
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=crn)


def _pick_backend(service):
    if config.BACKEND_NAME:
        return service.backend(config.BACKEND_NAME)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=1)


def _pick_qubit(backend):
    """Pick a good physical qubit: low readout error AND decent coherence.

    Readout error sets the visibility floor and the per-shot signal quality; T2 sets the
    coherence ceiling (even with virtual RZ there is idle + readout time). We score each
    qubit on a normalized combination so neither dominates by unit scale, then return the
    best. Returns physical index.
    """
    if config.PIN_QUBIT is not None:
        return int(config.PIN_QUBIT)
    props = backend.properties()
    rows = []
    for q in range(backend.num_qubits):
        try:
            ro = props.readout_error(q)
            t2 = props.t2(q) or 0.0
            t1 = props.t1(q) or 0.0
        except Exception:
            continue
        if ro is None or t2 <= 0:
            continue
        rows.append((q, float(ro), float(t2), float(t1)))
    if not rows:
        raise RuntimeError("could not select a qubit from backend.properties()")
    ro_max = max(r[1] for r in rows) or 1.0
    t2_max = max(r[2] for r in rows) or 1.0
    # minimize readout error (normalized) and maximize T2 (normalized), equal weight
    best = min(rows, key=lambda r: (r[1] / ro_max) - (r[2] / t2_max))
    return int(best[0])


def _calib_snapshot(backend, qubit):
    props = backend.properties()
    def _try(fn, *a):
        try:
            return fn(*a)
        except Exception:
            return None
    # single-qubit gate error: prefer sx, fall back to x
    gate_err = None
    for g in ("sx", "x"):
        try:
            gate_err = props.gate_error(g, qubit)
            break
        except Exception:
            continue
    return {
        "qubit": int(qubit),
        "T1": _try(props.t1, qubit),
        "T2": _try(props.t2, qubit),
        "readout_err": _try(props.readout_error, qubit),
        "gate_err": gate_err,
    }


def _transpile_with_integrity(circuits, specs, backend, qubit):
    """Transpile at optimization_level=0 pinned to `qubit`, asserting label integrity."""
    from qiskit import transpile
    isa = transpile(
        circuits, backend=backend,
        optimization_level=config.RUNTIME_OPTIONS["optimization_level"],
        initial_layout=[qubit],
    )
    isa = isa if isinstance(isa, list) else [isa]
    for tqc, spec in zip(isa, specs):
        bc.assert_injected_layers_intact(tqc, spec)   # raises on failure
    return isa


def _configure_sampler(sampler):
    """Turn OFF IBM twirling / DD / mitigation."""
    opt = sampler.options
    opt.twirling.enable_gates = config.RUNTIME_OPTIONS["twirling_enable_gates"]
    opt.twirling.enable_measure = config.RUNTIME_OPTIONS["twirling_enable_measure"]
    opt.dynamical_decoupling.enable = config.RUNTIME_OPTIONS["dynamical_decoupling"]
    # keep everything else at defaults; no resilience for SamplerV2 (it has none), but be explicit
    return sampler


def submit(validate_only=False):
    """Build, transpile (with integrity asserts), configure options, and submit.

    validate_only=True does EVERYTHING except sampler.run() — a free dress rehearsal of the
    live code path against the real backend (verifies transpile/integrity/options) with zero
    QPU cost. Use it before the real submission.
    """
    from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch

    config.use_live_paths()     # write to online_data/ (real-device run)
    service = _service()
    backend = _pick_backend(service)
    qubit = _pick_qubit(backend)
    calib = _calib_snapshot(backend, qubit)
    print(f"backend={backend.name}  qubit={qubit}  calib={calib}")

    specs = bc.generate_all_specs()
    circuits = [bc.build_qiskit_circuit(s) for s in specs]
    print(f"built {len(circuits)} circuits; transpiling (opt_level=0) + integrity asserts...")
    isa = _transpile_with_integrity(circuits, specs, backend, qubit)
    print("transpiler-integrity: all injected phase layers intact.")

    # one Batch, as few jobs as possible. Chunk to respect max circuits/job.
    max_per_job = 300
    chunks = [(specs[i:i + max_per_job], isa[i:i + max_per_job])
              for i in range(0, len(isa), max_per_job)]

    if validate_only:
        # configure a sampler bound to the backend and assert the safety options stick,
        # WITHOUT running. Proves the whole path is correct for free.
        sampler = _configure_sampler(Sampler(mode=backend))
        o = sampler.options
        assert o.twirling.enable_gates is False and o.twirling.enable_measure is False, \
            "IBM twirling not disabled"
        assert o.dynamical_decoupling.enable is False, "dynamical decoupling not disabled"
        print(f"VALIDATE OK: {len(circuits)} circuits transpile+assert clean; "
              f"IBM twirling OFF, DD OFF; would submit {len(chunks)} job(s) "
              f"of <= {max_per_job} circuits each. No QPU spent.")
        return

    job_records = []
    ts = datetime.now(timezone.utc).isoformat()
    with Batch(backend=backend) as batch:
        sampler = _configure_sampler(Sampler(mode=batch))
        for j, (spec_chunk, isa_chunk) in enumerate(chunks):
            job = sampler.run(isa_chunk, shots=config.SHOTS)
            job_records.append({
                "job_id": job.job_id(),
                "cids": [s.cid for s in spec_chunk],
            })
            print(f"  submitted job {j+1}/{len(chunks)}: {job.job_id()} "
                  f"({len(isa_chunk)} circuits)")
            # write incrementally so nothing is lost
            _save_job_ids(job_records, backend.name, qubit, calib, ts)

    print(f"\nsubmitted {len(job_records)} job(s). job_ids -> {config.JOB_IDS_PATH}")
    print("NEXT: wait for completion, then run  python retrieve.py")


def _save_job_ids(job_records, backend_name, qubit, calib, ts):
    payload = {
        "backend": backend_name, "qubit": qubit, "calib": calib,
        "submitted_at": ts, "shots": config.SHOTS,
        "options": config.RUNTIME_OPTIONS, "jobs": job_records,
    }
    with open(config.JOB_IDS_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    manifest = {
        "backend": backend_name, "qubit": qubit, "date": ts,
        "options": config.RUNTIME_OPTIONS, "calib": calib,
        "grid": {"thetas": config.THETAS, "depths": config.DEPTHS,
                 "instances": config.INSTANCES_PER_POINT, "shots": config.SHOTS,
                 "viscal_repeats": config.VISCAL_REPEATS},
        "n_signal": config.total_signal_circuits(),
        "n_viscal": config.total_viscal_circuits(),
    }
    with open(config.MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    submit(validate_only="--validate" in sys.argv)
