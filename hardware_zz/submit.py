"""Live step 1: pick coupled qubit groups, build, transpile (with integrity asserts), submit.

Does NOT analyze -- writes data/job_ids.json and data/run_manifest.json immediately so
nothing is lost if the session dies. Retrieve is a separate, resumable step.

Guardrails (the run is INVALID if any fails -- we raise, we do not submit):
  - IBM's own gate/measurement twirling OFF; dynamical decoupling OFF (it would also
    scramble the idle window the static ZZ coupling needs to accumulate in);
    resilience_level=0.
  - optimization_level=0 + barriers; ASSERT each transpiled circuit still has its
    injected RZ AND its idle delay on the target qubit intact.
  - "pair" topology needs 2 physically coupled qubits; "star" needs a qubit with >=2
    coupled neighbors (checked against backend.coupling_map, not assumed).
  - snapshot T1/T2/readout/gate error for every qubit used into the manifest.

qiskit / qiskit-ibm-runtime imported lazily so the offline sandbox never needs them.

Usage:
  python submit.py              # live submit
  python submit.py --validate   # dry-run the whole path against the backend, no QPU spent
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
    return service.least_busy(operational=True, simulator=False, min_num_qubits=3)


def _adjacency(backend):
    adj = {}
    for a, b in backend.coupling_map.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def _score_table(backend):
    """qubit -> (readout_error, T2), skipping qubits with missing calibration."""
    props = backend.properties()
    scores = {}
    for q in range(backend.num_qubits):
        try:
            ro = props.readout_error(q)
            t2 = props.t2(q)
        except Exception:
            continue
        if ro is None or not t2 or t2 <= 0:
            continue
        scores[q] = (float(ro), float(t2))
    return scores


def _pick_pair(backend):
    """Best physically-coupled 2-qubit group: low combined readout error, high T2."""
    scores = _score_table(backend)
    ro_max = max((s[0] for s in scores.values()), default=1.0) or 1.0
    t2_max = max((s[1] for s in scores.values()), default=1.0) or 1.0

    def cost(q):
        ro, t2 = scores[q]
        return ro / ro_max - t2 / t2_max

    best, best_cost = None, None
    for a, nbrs in _adjacency(backend).items():
        if a not in scores:
            continue
        for b in nbrs:
            if b not in scores:
                continue
            c = cost(a) + cost(b)
            if best_cost is None or c < best_cost:
                best_cost, best = c, (a, b)
    if best is None:
        raise RuntimeError("could not find a connected, calibrated qubit pair on this backend")
    return list(best)


def _pick_star(backend, exclude=()):
    """Best calibrated (center, neighbor, neighbor) star: center coupled to both
    neighbors; neighbor-neighbor coupling not required (Sec 4 topology)."""
    scores = _score_table(backend)
    ro_max = max((s[0] for s in scores.values()), default=1.0) or 1.0
    t2_max = max((s[1] for s in scores.values()), default=1.0) or 1.0

    def cost(q):
        ro, t2 = scores[q]
        return ro / ro_max - t2 / t2_max

    best, best_cost = None, None
    for center, nbrs in _adjacency(backend).items():
        if center in exclude or center not in scores:
            continue
        cands = sorted((n for n in nbrs if n in scores and n not in exclude), key=cost)
        if len(cands) < 2:
            continue
        n1, n2 = cands[0], cands[1]
        c = cost(center) + cost(n1) + cost(n2)
        if best_cost is None or c < best_cost:
            best_cost, best = c, (center, n1, n2)
    if best is None:
        raise RuntimeError("could not find a calibrated 3-qubit star on this backend")
    return list(best)


def _calib_snapshot(backend, qubits):
    props = backend.properties()

    def _try(fn, *a):
        try:
            return fn(*a)
        except Exception:
            return None

    out = []
    for q in qubits:
        gate_err = None
        for g in ("sx", "x"):
            try:
                gate_err = props.gate_error(g, q)
                break
            except Exception:
                continue
        out.append({"qubit": int(q), "T1": _try(props.t1, q), "T2": _try(props.t2, q),
                    "readout_err": _try(props.readout_error, q), "gate_err": gate_err})
    return out


def _transpile_group(circuits, specs, backend, physical_qubits):
    from qiskit import transpile
    isa = transpile(circuits, backend=backend,
                    optimization_level=config.RUNTIME_OPTIONS["optimization_level"],
                    initial_layout=physical_qubits)
    isa = isa if isinstance(isa, list) else [isa]
    target_physical_qubit = physical_qubits[0]
    for tqc, spec in zip(isa, specs):
        bc.assert_injected_layers_intact(tqc, spec, target_physical_qubit)
    return isa


def _configure_sampler(sampler):
    opt = sampler.options
    opt.twirling.enable_gates = config.RUNTIME_OPTIONS["twirling_enable_gates"]
    opt.twirling.enable_measure = config.RUNTIME_OPTIONS["twirling_enable_measure"]
    opt.dynamical_decoupling.enable = config.RUNTIME_OPTIONS["dynamical_decoupling"]
    return sampler


def submit(validate_only=False):
    from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch

    config.use_live_paths()
    service = _service()
    backend = _pick_backend(service)
    pair_qubits = _pick_pair(backend)
    star_qubits = _pick_star(backend)
    print(f"backend={backend.name}  pair={pair_qubits}  star={star_qubits}")
    calib = {"pair": _calib_snapshot(backend, pair_qubits),
            "star": _calib_snapshot(backend, star_qubits)}

    specs = bc.generate_all_specs()
    specs_by_topo = {"pair": [s for s in specs if s.topology == "pair"],
                     "star": [s for s in specs if s.topology == "star"]}
    layout = {"pair": pair_qubits, "star": star_qubits}

    isa_all, specs_all = [], []
    for topology in ("pair", "star"):
        group_specs = specs_by_topo[topology]
        circuits = [bc.build_qiskit_circuit(s) for s in group_specs]
        print(f"transpiling {len(circuits)} {topology}-topology circuits "
              f"(opt_level=0) onto {layout[topology]} + integrity asserts...")
        isa = _transpile_group(circuits, group_specs, backend, layout[topology])
        isa_all.extend(isa)
        specs_all.extend(group_specs)
    print("transpiler-integrity: all injected RZ + delay layers intact.")

    max_per_job = 300
    chunks = [(specs_all[i:i + max_per_job], isa_all[i:i + max_per_job])
              for i in range(0, len(isa_all), max_per_job)]

    if validate_only:
        sampler = _configure_sampler(Sampler(mode=backend))
        o = sampler.options
        assert o.twirling.enable_gates is False and o.twirling.enable_measure is False
        assert o.dynamical_decoupling.enable is False
        print(f"VALIDATE OK: {len(isa_all)} circuits transpile+assert clean; "
              f"IBM twirling OFF, DD OFF; would submit {len(chunks)} job(s). No QPU spent.")
        return

    job_records = []
    ts = datetime.now(timezone.utc).isoformat()
    with Batch(backend=backend) as batch:
        sampler = _configure_sampler(Sampler(mode=batch))
        for j, (spec_chunk, isa_chunk) in enumerate(chunks):
            job = sampler.run(isa_chunk, shots=config.SHOTS)
            job_records.append({"job_id": job.job_id(), "cids": [s.cid for s in spec_chunk]})
            print(f"  submitted job {j+1}/{len(chunks)}: {job.job_id()} ({len(isa_chunk)} circuits)")
            _save_job_ids(job_records, backend.name, pair_qubits, star_qubits, calib, ts)

    print(f"\nsubmitted {len(job_records)} job(s). job_ids -> {config.JOB_IDS_PATH}")
    print("NEXT: wait for completion, then run  python retrieve.py")


def _save_job_ids(job_records, backend_name, pair_qubits, star_qubits, calib, ts):
    payload = {"backend": backend_name, "pair_qubits": pair_qubits, "star_qubits": star_qubits,
              "calib": calib, "submitted_at": ts, "shots": config.SHOTS,
              "options": config.RUNTIME_OPTIONS, "jobs": job_records}
    with open(config.JOB_IDS_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    manifest = {"backend": backend_name, "pair_qubits": pair_qubits, "star_qubits": star_qubits,
               "date": ts, "options": config.RUNTIME_OPTIONS, "calib": calib,
               "grid": {"thetas": config.THETA0_SWEEP, "delay_s": config.DELAY_S,
                        "instances": {"pair": config.INSTANCES_PAIR, "star": config.INSTANCES_STAR},
                        "shots": config.SHOTS, "viscal_repeats": config.VISCAL_REPEATS},
               "n_signal": config.total_signal_circuits(), "n_viscal": config.total_viscal_circuits()}
    with open(config.MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    submit(validate_only="--validate" in sys.argv)
