"""One entry point for the ZZ-crosstalk hardware pipeline.

Default (offline dress rehearsal): build specs -> simulate on the local noisy sim
(reuses core/) -> raw data -> analyze -> figure. Proves the pipeline recovers the
injected theta0 BEFORE any QPU time is spent. No qiskit, no network.

--live: submit -> [wait] -> retrieve -> analyze on real hardware. Requires qiskit,
qiskit-ibm-runtime, network, and IBM_TOKEN/IBM_CRN (see hardware/.env, shared with the
single-qubit run).

Usage:
  python run_all.py               # offline dry-run (recommended first)
  python run_all.py --live        # submit to hardware (then retrieve.py, analyze.py)
  python run_all.py --live-full   # submit, poll to completion, retrieve, analyze
"""

import json
import sys
import time

import config
import build_circuits as bc
import analyze as an


def run_offline():
    import local_sim as ls
    print("=== OFFLINE DRESS REHEARSAL (local noisy sim; reuses core/) ===")
    specs = bc.generate_all_specs()
    bad = [s.cid for s in specs if not bc.spec_integrity_ok(s)]
    if bad:
        raise SystemExit(f"spec-integrity failures: {bad[:5]} ...")
    print(f"built {len(specs)} circuits "
          f"({config.total_signal_circuits()} signal + {config.total_viscal_circuits()} viscal); "
          f"integrity OK.")

    records = ls.run_offline(specs)
    with open(config.RAW_RESULTS_PATH, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"simulated {len(records)} circuits -> {config.RAW_RESULTS_PATH}")

    analysis = an.main(config.RAW_RESULTS_PATH)
    _positive_control(analysis)
    return analysis


def _positive_control(analysis):
    print("\n=== POSITIVE-CONTROL CHECKS ===")
    ok = True
    for topology in ("pair", "star"):
        for key, r in analysis["results"][topology].items():
            if abs(r["theta0_star"]) < 1e-9:
                passed = abs(r["theta0_hat"]) <= config.NULL_TOL
                ok = ok and passed
                print(f"  [{topology}] null theta0*=0: theta0_hat={r['theta0_hat']:+.4f}  "
                      f"{'PASS' if passed else 'FAIL'}  (|.|<{config.NULL_TOL})")
        diag_ok = True
        for key, r in analysis["results"][topology].items():
            if r["theta0_star"] > 0:
                lo, hi = r["bootstrap_ci"]["theta0"]
                half = 0.5 * (hi - lo)
                near = abs(r["theta0_hat"] - r["theta0_star"]) <= max(3 * half, 0.03)
                diag_ok = diag_ok and near
        print(f"  [{topology}] diagonal recovery: {'PASS' if diag_ok else 'CHECK'}")
        ok = ok and diag_ok

        kappa_true = config.OFFLINE_TRUE_KAPPA
        first = next(iter(analysis["results"][topology].values()))
        for j, kh in first["kappa_hat"].items():
            err = abs(kh - kappa_true[j])
            lo, hi = first["bootstrap_ci"]["kappa"][j]
            half = 0.5 * (hi - lo)
            kappa_ok = err <= max(3 * half, 0.02)
            ok = ok and kappa_ok
            print(f"  [{topology}] kappa_1{j}: hat={kh:+.4f} assumed-true={kappa_true[j]:.4f} "
                  f"|err|={err:.4f}  {'PASS' if kappa_ok else 'CHECK'}")

    print(f"\nDRESS REHEARSAL: {'PASS — pipeline recovers theta0 and kappa; safe to run live.' if ok else 'FAIL — fix before spending QPU time.'}")
    if not ok:
        raise SystemExit(1)


def run_live(full=False):
    import submit as sub
    print("=== LIVE HARDWARE RUN (output -> online_data/, online_figures/) ===")
    config.use_live_paths()
    config.require_credentials()
    sub.submit()
    if not full:
        print("\nJobs submitted. When they finish:\n  python retrieve.py\n  python analyze.py --live")
        return
    import retrieve as ret
    from qiskit_ibm_runtime import QiskitRuntimeService
    token, crn = config.require_credentials()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=crn)
    with open(config.JOB_IDS_PATH) as fh:
        jobmeta = json.load(fh)
    print("polling jobs to completion...")
    for jrec in jobmeta["jobs"]:
        job = service.job(jrec["job_id"])
        while str(job.status()) not in ("DONE", "JobStatus.DONE", "Done",
                                        "ERROR", "JobStatus.ERROR", "CANCELLED"):
            time.sleep(15)
        print(f"  job {jrec['job_id']}: {job.status()}")
    ret.retrieve()
    an.main(config.RAW_RESULTS_PATH)


def main():
    args = set(sys.argv[1:])
    if "--help" in args or "-h" in args:
        print(__doc__)
        return
    if "--live-full" in args:
        run_live(full=True)
    elif "--live" in args:
        run_live(full=False)
    else:
        run_offline()


if __name__ == "__main__":
    main()
