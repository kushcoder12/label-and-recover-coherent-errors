"""Diagnostic: does the delay ALONE explain the null-offset / visibility drift?

Part 2/3 of HARDWARE_ZZ_FINDINGS.md swept delay 1us -> 8us together with the full
theta0 sweep and randomized neighbor labels, and saw both the null-control offset and
the visibility get worse. That confounds three things at once. This script isolates
ONE variable -- idle delay -- on the SAME qubit pair ([53,54], target=53) used there:

  - theta0 fixed at 0 (no injected phase; "drift" circuit is just H -> delay -> SDG,H)
  - neighbor (qubit 54) held fixed at |0>, never driven, never randomized
  - a companion "viscal" circuit (RZ(pi/2) instead of the delay-only drift) at the
    SAME delay gives v(delay) independently, so drift and decoherence don't get
    conflated into one number.

Both circuit kinds get MULTIPLE independent repeats per delay (for an empirical SEM,
not just the binomial-counting formula) rather than many random "instances" -- there
is nothing left to randomize once theta0=0 and the neighbor is fixed.

No qiskit at import time (offline-checkable); qiskit only inside build/submit/retrieve.
Results are written to files distinct from the main run -- nothing here overwrites
online_data/raw_results.json, analysis.json, or run_manifest.json.

Usage:
  python diagnostic_delay_sweep.py --check      # structural self-check, no network
  python diagnostic_delay_sweep.py --validate   # dry-run against the live backend, no QPU
  python diagnostic_delay_sweep.py --submit      # submit for real
  python diagnostic_delay_sweep.py --retrieve    # fetch + analyze + figure
"""

import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import config

sys.path.insert(0, os.path.join(config.REPO_ROOT, "experiments"))   # for _style (figures)

DELAYS_US = [1.0, 2.0, 4.0, 8.0]
REPEATS = 5
SHOTS = 4000
TARGET_QUBIT = 53
NEIGHBOR_QUBIT = 54          # held fixed at |0>, never driven

RAW_PATH = os.path.join(config.HARDWARE_ZZ_DIR, "online_data", "diagnostic_delay_sweep_raw.json")
ANALYSIS_PATH = os.path.join(config.HARDWARE_ZZ_DIR, "online_data", "diagnostic_delay_sweep_analysis.json")
JOB_IDS_PATH = os.path.join(config.HARDWARE_ZZ_DIR, "online_data", "diagnostic_delay_sweep_job_ids.json")
FIG_PATH = os.path.join(config.HARDWARE_ZZ_DIR, "online_figures", "fig16_delay_diagnostic")


@dataclass
class DiagSpec:
    cid: str
    kind: str          # "drift" (theta0=0) or "viscal" (theta0=pi/2)
    delay_us: float
    rep: int


def generate_specs():
    specs = []
    for delay_us in DELAYS_US:
        for kind in ("drift", "viscal"):
            for rep in range(REPEATS):
                cid = f"diag_{kind}_d{delay_us:.1f}us_r{rep:02d}"
                specs.append(DiagSpec(cid=cid, kind=kind, delay_us=delay_us, rep=rep))
    return specs


def build_qiskit_circuit(spec):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 1, name=spec.cid)
    qc.h(0)
    qc.barrier()
    if spec.kind == "viscal":
        qc.rz(math.pi / 2, 0)
    qc.barrier()
    delay_s = spec.delay_us * 1.0e-6
    qc.delay(delay_s, 0, unit="s")
    qc.delay(delay_s, 1, unit="s")     # neighbor idle, same duration, never driven
    qc.barrier()
    qc.sdg(0)
    qc.h(0)
    qc.barrier()
    qc.measure(0, 0)
    return qc


def assert_integrity(transpiled_qc, spec, target_physical_qubit, atol=1e-9):
    n_barriers = sum(1 for ci in transpiled_qc.data if ci.operation.name == "barrier")
    if n_barriers != 4:
        raise AssertionError(f"[{spec.cid}] {n_barriers} barriers, expected 4. Run INVALID.")
    if spec.kind == "viscal":
        found = any(ci.operation.name == "rz"
                   and round(abs((float(ci.operation.params[0]) + math.pi) % (2 * math.pi) - math.pi), 9)
                       == round(math.pi / 2, 9)
                   for ci in transpiled_qc.data)
        if not found:
            raise AssertionError(f"[{spec.cid}] injected RZ(pi/2) not found intact. Run INVALID.")
    total_delay = sum(float(ci.operation.duration) for ci in transpiled_qc.data
                      if ci.operation.name == "delay"
                      and transpiled_qc.find_bit(ci.qubits[0]).index == target_physical_qubit)
    if total_delay <= 0:
        raise AssertionError(f"[{spec.cid}] idle delay on the target qubit was dropped. Run INVALID.")
    return True


def structural_check():
    """No qiskit / no network: confirm spec generation is sane before anything else."""
    specs = generate_specs()
    assert len(specs) == len(DELAYS_US) * 2 * REPEATS
    for s in specs:
        assert s.kind in ("drift", "viscal")
        assert s.delay_us in DELAYS_US
    print(f"structural check OK: {len(specs)} circuits "
          f"({len(DELAYS_US)} delays x 2 kinds x {REPEATS} repeats), {SHOTS} shots each.")
    print(f"target qubit={TARGET_QUBIT}, neighbor qubit={NEIGHBOR_QUBIT} (fixed at |0>, never driven).")


def _service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token, crn = config.require_credentials()
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=crn)


def _transpile(circuits, specs, backend):
    from qiskit import transpile
    isa = transpile(circuits, backend=backend,
                    optimization_level=config.RUNTIME_OPTIONS["optimization_level"],
                    initial_layout=[TARGET_QUBIT, NEIGHBOR_QUBIT])
    isa = isa if isinstance(isa, list) else [isa]
    for tqc, spec in zip(isa, specs):
        assert_integrity(tqc, spec, TARGET_QUBIT)
    return isa


def _configure_sampler(sampler):
    opt = sampler.options
    opt.twirling.enable_gates = config.RUNTIME_OPTIONS["twirling_enable_gates"]
    opt.twirling.enable_measure = config.RUNTIME_OPTIONS["twirling_enable_measure"]
    opt.dynamical_decoupling.enable = config.RUNTIME_OPTIONS["dynamical_decoupling"]
    return sampler


def run_validate_or_submit(validate_only):
    from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch

    config.use_live_paths()
    service = _service()
    backend = service.backend(config.BACKEND_NAME) if config.BACKEND_NAME else \
        service.least_busy(operational=True, simulator=False, min_num_qubits=2)

    specs = generate_specs()
    circuits = [build_qiskit_circuit(s) for s in specs]
    print(f"transpiling {len(circuits)} circuits onto [{TARGET_QUBIT},{NEIGHBOR_QUBIT}] "
          f"(opt_level=0) + integrity asserts...")
    isa = _transpile(circuits, specs, backend)
    print("transpiler-integrity: all delays (+ viscal RZ) intact.")

    if validate_only:
        sampler = _configure_sampler(Sampler(mode=backend))
        o = sampler.options
        assert o.twirling.enable_gates is False and o.twirling.enable_measure is False
        assert o.dynamical_decoupling.enable is False
        print(f"VALIDATE OK: {len(circuits)} circuits transpile+assert clean; "
              f"IBM twirling OFF, DD OFF. No QPU spent.")
        return

    ts = datetime.now(timezone.utc).isoformat()
    with Batch(backend=backend) as batch:
        sampler = _configure_sampler(Sampler(mode=batch))
        job = sampler.run(isa, shots=SHOTS)
        payload = {"backend": backend.name, "target_qubit": TARGET_QUBIT,
                  "neighbor_qubit": NEIGHBOR_QUBIT, "submitted_at": ts, "shots": SHOTS,
                  "job_id": job.job_id(), "cids": [s.cid for s in specs]}
        with open(JOB_IDS_PATH, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"submitted job {job.job_id()} ({len(circuits)} circuits) -> {JOB_IDS_PATH}")


def _counts_from_pub(pub_result):
    data = pub_result.data
    creg = None
    for f in dir(data):
        if f.startswith("_"):
            continue
        obj = getattr(data, f)
        if hasattr(obj, "get_counts"):
            creg = obj
            break
    counts = creg.get_counts()
    n0 = sum(v for k, v in counts.items() if k.endswith("0"))
    n1 = sum(v for k, v in counts.items() if k.endswith("1"))
    return {"0": int(n0), "1": int(n1)}


def retrieve():
    config.use_live_paths()
    if not os.path.exists(JOB_IDS_PATH):
        raise SystemExit(f"no {JOB_IDS_PATH}; run --submit first.")
    with open(JOB_IDS_PATH) as fh:
        jm = json.load(fh)

    service = _service()
    job = service.job(jm["job_id"])
    status = job.status()
    if str(status) not in ("DONE", "JobStatus.DONE", "Done"):
        raise SystemExit(f"job {jm['job_id']} status={status}, not ready yet.")
    result = job.result()

    specs = {s.cid: s for s in generate_specs()}
    records = []
    for pub_result, cid in zip(result, jm["cids"]):
        spec = specs[cid]
        counts = _counts_from_pub(pub_result)
        records.append({"cid": cid, "kind": spec.kind, "delay_us": spec.delay_us,
                        "rep": spec.rep, "counts": counts, "shots": jm["shots"],
                        "backend": jm["backend"], "job_id": jm["job_id"]})
    with open(RAW_PATH, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"wrote {len(records)} records -> {RAW_PATH}")
    return records


def expval_Y(counts):
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    tot = n0 + n1
    return (n0 - n1) / tot if tot else 0.0


def analyze(records=None):
    if records is None:
        with open(RAW_PATH) as fh:
            records = json.load(fh)

    by_delay = {}
    for r in records:
        by_delay.setdefault(r["delay_us"], {"drift": [], "viscal": []})[r["kind"]].append(
            expval_Y(r["counts"]))

    import numpy as np
    rows = []
    for delay_us in sorted(by_delay):
        drift_ys = np.asarray(by_delay[delay_us]["drift"], float)
        viscal_ys = np.asarray(by_delay[delay_us]["viscal"], float)
        v = float(np.clip(viscal_ys.mean(), 1e-6, 1.0))
        v_sem = float(viscal_ys.std(ddof=1) / np.sqrt(len(viscal_ys))) if len(viscal_ys) > 1 else 0.0
        y_drift = float(drift_ys.mean())
        y_drift_sem = float(drift_ys.std(ddof=1) / np.sqrt(len(drift_ys))) if len(drift_ys) > 1 else 0.0
        drift_angle = float(np.arcsin(np.clip(y_drift / v, -1.0, 1.0)))
        rows.append({"delay_us": delay_us, "v": v, "v_sem": v_sem,
                    "y_drift_raw": y_drift, "y_drift_sem": y_drift_sem,
                    "drift_angle_hat": drift_angle,
                    "n_repeats": len(drift_ys)})
    out = {"target_qubit": TARGET_QUBIT, "neighbor_qubit": NEIGHBOR_QUBIT,
          "shots_per_circuit": SHOTS, "repeats": REPEATS, "rows": rows}
    with open(ANALYSIS_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def make_figure(analysis_out):
    from _style import setup
    C = setup()
    import matplotlib.pyplot as plt

    rows = analysis_out["rows"]
    delays = [r["delay_us"] for r in rows]
    v = [r["v"] for r in rows]
    v_sem = [r["v_sem"] for r in rows]
    drift = [r["drift_angle_hat"] for r in rows]
    drift_sem_angle = [abs(r["y_drift_sem"] / max(r["v"], 1e-6)) for r in rows]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.6))
    axA.errorbar(delays, drift, yerr=drift_sem_angle, fmt="o-", color=C["vermil"],
                ms=7, capsize=3, lw=1.6)
    axA.axhline(0.0, color="grey", lw=0.8, alpha=0.6)
    axA.set_xlabel(r"idle delay ($\mu$s)")
    axA.set_ylabel(r"apparent drift angle (rad), $\theta_1^{(0)}=0$, neighbor fixed")
    axA.set_title("(a) Null-offset vs delay alone")

    axB.errorbar(delays, v, yerr=v_sem, fmt="s-", color=C["blue"], ms=7, capsize=3, lw=1.6)
    axB.set_xlabel(r"idle delay ($\mu$s)")
    axB.set_ylabel(r"visibility $v$")
    axB.set_title("(b) Visibility vs delay alone")
    axB.set_ylim(0, 1.05)

    fig.suptitle(f"Fig 16 — Delay-only diagnostic (qubits {TARGET_QUBIT}/{NEIGHBOR_QUBIT}, "
                f"neighbor fixed, not randomized)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH + ".pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIG_PATH + ".png", bbox_inches="tight", pad_inches=0.05, dpi=300)
    print(f"  wrote {FIG_PATH}.pdf")
    print(f"  wrote {FIG_PATH}.png")
    plt.close(fig)


def main():
    args = set(sys.argv[1:])
    if "--check" in args or not args:
        structural_check()
        return
    if "--validate" in args:
        run_validate_or_submit(validate_only=True)
        return
    if "--submit" in args:
        run_validate_or_submit(validate_only=False)
        return
    if "--retrieve" in args:
        records = retrieve()
        out = analyze(records)
        for r in out["rows"]:
            print(f"  delay={r['delay_us']:.1f}us  v={r['v']:.3f}+-{r['v_sem']:.3f}  "
                  f"raw<Y>={r['y_drift_raw']:+.4f}+-{r['y_drift_sem']:.4f}  "
                  f"drift_angle_hat={r['drift_angle_hat']:+.4f}")
        make_figure(out)
        print(f"\nanalysis -> {ANALYSIS_PATH}")
        print(f"figure -> {FIG_PATH}.png/.pdf")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
