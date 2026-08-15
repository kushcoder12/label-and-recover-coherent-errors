"""One entry point for the whole pipeline.

Default (offline dress rehearsal): build specs -> simulate on the local noisy
statevector sim (reuse core + noise) -> write raw data -> analyze -> figures. Proves the
pipeline recovers theta* BEFORE any QPU time is spent. No qiskit, no network.

--live: run the real IBM flow as SEPARATE resumable steps (submit -> [wait] -> retrieve ->
analyze). Requires qiskit, qiskit-ibm-runtime, network, and IBM_TOKEN/IBM_CRN. Because jobs
queue, --live submits and then instructs you to run retrieve.py + analyze.py once done; it
does not block the process waiting.

Usage:
  python run_all.py               # offline dry-run (recommended first)
  python run_all.py --live        # submit to hardware (then retrieve.py, analyze.py)
  python run_all.py --live-full   # submit, poll to completion, retrieve, analyze in one go
"""

import json
import sys
import time

import config
import build_circuits as bc
import analyze as an


def run_offline():
    import local_sim as ls
    print("=== OFFLINE DRESS REHEARSAL (local noisy statevector sim; reuse core + noise) ===")
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
    """Guardrails: theta*=0 null must recover ~0; theta_hat must track theta* (diagonal)."""
    print("\n=== POSITIVE-CONTROL CHECKS ===")
    ok = True
    # null control at every depth: theta_hat must be within the physically-negligible
    # NULL_TOL band. We ALSO report per-point 95%-CI containment for information, noting
    # that with several nulls ~1-in-20 excludes zero by chance (multiple comparisons).
    n_ci_exclude = 0
    n_null = 0
    for key, r in analysis["results"].items():
        if abs(r["theta_star"]) < 1e-9:
            n_null += 1
            lo, hi = r["theta_ci"]
            ci_has_zero = lo <= 0.0 <= hi
            n_ci_exclude += 0 if ci_has_zero else 1
            passed = abs(r["theta_hat"]) <= config.NULL_TOL
            ok = ok and passed
            note = "" if ci_has_zero else "  [95% CI excludes 0 — expected for ~1/20]"
            print(f"  null theta*=0 L={r['L']:>2}: theta_hat={r['theta_hat']:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]  {'PASS' if passed else 'FAIL'}"
                  f"  (|.|<{config.NULL_TOL}){note}")
    print(f"  null summary: {n_ci_exclude}/{n_null} of the 95% CIs exclude 0 "
          f"(chance ~{1 - 0.95**max(n_null,1):.0%}); all |theta_hat| within {config.NULL_TOL} rad.")
    # diagonal: at a mid depth, |theta_hat - theta*| within a few CI widths
    diag_ok = True
    for key, r in analysis["results"].items():
        if r["theta_star"] > 0 and r["L"] in (4, 8):
            lo, hi = r["theta_ci"]
            half = 0.5 * (hi - lo)
            near = abs(r["theta_hat"] - r["theta_star"]) <= max(3 * half, 0.02)
            diag_ok = diag_ok and near
    print(f"  diagonal recovery (L in 4,8): {'PASS' if diag_ok else 'CHECK'}")
    ok = ok and diag_ok
    print(f"\nDRESS REHEARSAL: {'PASS — pipeline recovers theta*; safe to run live.' if ok else 'FAIL — fix before spending QPU time.'}")
    if not ok:
        raise SystemExit(1)


def run_live(full=False):
    import submit as sub
    print("=== LIVE HARDWARE RUN (output -> online_data/, online_figures/) ===")
    config.use_live_paths()
    config.require_credentials()   # fail loudly before touching qiskit if creds missing
    sub.submit()
    if not full:
        print("\nJobs submitted. When they finish:\n  python retrieve.py\n  python analyze.py")
        return
    # --live-full: poll then retrieve+analyze
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
