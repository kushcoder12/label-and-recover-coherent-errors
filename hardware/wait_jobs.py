"""hardware/wait_jobs.py — poll live jobs until all reach a terminal state, then exit.

Prints a timestamped status line each cycle. Exit 0 when every job is DONE/ERROR/CANCELLED.
Used to block on the (wall-clock, unbilled) queue wait before retrieve.py.
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import config
from qiskit_ibm_runtime import QiskitRuntimeService

TERMINAL = {"DONE", "ERROR", "CANCELLED"}


def _norm(status):
    s = str(status).split(".")[-1].upper()
    return s


def main():
    config.use_live_paths()
    token, crn = config.require_credentials()
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=crn)
    jm = json.load(open(config.JOB_IDS_PATH))
    job_ids = [jr["job_id"] for jr in jm["jobs"]]

    poll = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    while True:
        states = {}
        for jid in job_ids:
            try:
                states[jid] = _norm(svc.job(jid).status())
            except Exception as e:
                states[jid] = f"ERR({type(e).__name__})"
        ts = time.strftime("%H:%M:%S")
        summary = ", ".join(f"{jid[-4:]}={st}" for jid, st in states.items())
        print(f"[{ts}] {summary}", flush=True)
        if all(st in TERMINAL for st in states.values()):
            done = sum(st == "DONE" for st in states.values())
            print(f"ALL TERMINAL: {done}/{len(job_ids)} DONE", flush=True)
            return 0
        time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main())
