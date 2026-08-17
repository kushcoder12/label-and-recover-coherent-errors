"""Live step 2: fetch job results, write raw data to disk.

Separate, resumable process: reads data/job_ids.json, fetches each job, writes
data/raw_results.json BEFORE any analysis. Safe to re-run: already-fetched jobs are
skipped.

Usage:
  python retrieve.py
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


def _spec_index():
    return {s.cid: s for s in bc.generate_all_specs()}


def _counts_from_pub(pub_result):
    data = pub_result.data
    fields = [f for f in dir(data) if not f.startswith("_")]
    creg = None
    for f in fields:
        obj = getattr(data, f)
        if hasattr(obj, "get_counts"):
            creg = obj
            break
    if creg is None:
        raise RuntimeError("no classical register with get_counts on this PubResult")
    counts = creg.get_counts()
    n0 = sum(v for k, v in counts.items() if k.endswith("0"))
    n1 = sum(v for k, v in counts.items() if k.endswith("1"))
    bitstrings = creg.get_bitstrings() if config.SAVE_BITSTRINGS else None
    return {"0": int(n0), "1": int(n1)}, bitstrings


def retrieve():
    config.use_live_paths()
    if not os.path.exists(config.JOB_IDS_PATH):
        raise SystemExit(f"no {config.JOB_IDS_PATH}; run submit.py first.")
    with open(config.JOB_IDS_PATH) as fh:
        jobmeta = json.load(fh)

    specs = _spec_index()
    service = _service()

    existing = {}
    if os.path.exists(config.RAW_RESULTS_PATH):
        with open(config.RAW_RESULTS_PATH) as fh:
            for r in json.load(fh):
                existing[r["cid"]] = r

    backend = jobmeta["backend"]
    pair_qubits = jobmeta["pair_qubits"]
    star_qubits = jobmeta["star_qubits"]
    calib = jobmeta.get("calib", {})
    ts = datetime.now(timezone.utc).isoformat()
    qubit_map = {"pair": pair_qubits, "star": star_qubits}

    records = dict(existing)
    for jrec in jobmeta["jobs"]:
        cids = jrec["cids"]
        if all(c in records for c in cids):
            print(f"  job {jrec['job_id']}: already fetched, skipping.")
            continue
        job = service.job(jrec["job_id"])
        status = job.status()
        if str(status) not in ("DONE", "JobStatus.DONE", "Done"):
            print(f"  job {jrec['job_id']}: status={status}, not ready -- skipping.")
            continue
        result = job.result()
        for pub_result, cid in zip(result, cids):
            spec = specs[cid]
            counts, bitstrings = _counts_from_pub(pub_result)
            rec = {
                "cid": cid, "kind": spec.kind, "topology": spec.topology,
                "theta0": spec.theta0, "s1": spec.s1, "c": spec.c, "delay_s": spec.delay_s,
                "qubits": qubit_map[spec.topology], "backend": backend, "job_id": jrec["job_id"],
                "timestamp": ts, "shots": config.SHOTS, "counts": counts,
                "calib": calib.get(spec.topology, {}),
            }
            if bitstrings is not None:
                rec["bitstrings"] = bitstrings
            records[cid] = rec
        print(f"  job {jrec['job_id']}: fetched {len(cids)} circuits.")
        _write(records)

    _write(records)
    print(f"\nwrote {len(records)} records -> {config.RAW_RESULTS_PATH}")
    print("NEXT: python analyze.py --live")


def _write(records):
    with open(config.RAW_RESULTS_PATH, "w") as fh:
        json.dump(list(records.values()), fh, indent=2)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    retrieve()
