"""Single source of truth for the ZZ-crosstalk sweep, paths, and hardware run options.

No qiskit / no network here — pure configuration, mirroring hardware/config.py's split
between the offline dress rehearsal and the live pipeline. Credentials are read from
hardware/.env (the SAME file used by the single-qubit run, not a duplicate) so a token
never needs to be copied into a second location.
"""

import os

HARDWARE_ZZ_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HARDWARE_ZZ_DIR)
ENV_PATH = os.path.join(REPO_ROOT, "hardware", ".env")

_OFFLINE_DIRS = ("data", "figures")
_ONLINE_DIRS = ("online_data", "online_figures")


def _bind_paths(data_name, fig_name):
    g = globals()
    g["DATA_DIR"] = os.path.join(HARDWARE_ZZ_DIR, data_name)
    g["FIG_DIR"] = os.path.join(HARDWARE_ZZ_DIR, fig_name)
    g["JOB_IDS_PATH"] = os.path.join(g["DATA_DIR"], "job_ids.json")
    g["MANIFEST_PATH"] = os.path.join(g["DATA_DIR"], "run_manifest.json")
    g["RAW_RESULTS_PATH"] = os.path.join(g["DATA_DIR"], "raw_results.json")
    g["ANALYSIS_PATH"] = os.path.join(g["DATA_DIR"], "analysis.json")
    for _d in (g["DATA_DIR"], g["FIG_DIR"]):
        os.makedirs(_d, exist_ok=True)


def use_live_paths():
    _bind_paths(*_ONLINE_DIRS)


def use_offline_paths():
    _bind_paths(*_OFFLINE_DIRS)


_bind_paths(*_OFFLINE_DIRS)

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
# theta0 is qubit 1's OWN injected virtual-Z knob (same role as the single-qubit run's
# theta*: a KNOWN, controlled quantity -> a genuine positive control). kappa_1j is the
# device's real static ZZ coupling -- never injected, so it has no ground truth to check
# against (see hardware_zz/README.md and MULTIQUBIT_CROSSTALK_SPEC.md Sec 6).
THETA0_SWEEP = [0.0, 0.05, 0.10, 0.20]        # rad; 0.0 is the null control

# Idle duration inserted between neighbor state-prep and the probe, during which the
# REAL static ZZ coupling accumulates (RZ is virtual, so without this delay there is no
# time for J_1j to act at all). kappa_1j grows linearly with this delay while shot noise
# does not, so it is the cheapest lever for resolving a small coupling -- raised from the
# first run's 1 us (which did not resolve kappa_1j above zero) to 8 us: still a small
# fraction of the measured T2 on these qubits (159-280 us), so decoherence loss stays
# modest, while giving 8x more accumulated phase for the same shot budget.
DELAY_S = 8.0e-6

INSTANCES_PAIR = 30     # twirl instances/point, 2-qubit (1 neighbor) topology
INSTANCES_STAR = 40     # twirl instances/point, 3-qubit (2 neighbor, star) topology
SHOTS = 2000
VISCAL_REPEATS = 20
VISCAL_NET_PHASE = 3.141592653589793 / 2.0    # pi/2; sin is stationary there, so a
                                              # small unknown kappa contribution biases
                                              # the visibility calibration only at O(kappa^2)

SEED = 20260816

# theta0=0 null-control tolerance (rad), same reasoning as hardware/config.py.
NULL_TOL = 0.01

# ---------------------------------------------------------------------------
# Hardware / runtime options -- identical guardrails to hardware/config.py, plus:
# dynamical_decoupling MUST stay off here for a second reason beyond "don't suppress
# what we're measuring": DD inserts pulses into idle windows, which would scramble the
# very delay period the static ZZ coupling needs to accumulate in.
# ---------------------------------------------------------------------------
RUNTIME_OPTIONS = {
    "optimization_level": 0,
    "twirling_enable_gates": False,
    "twirling_enable_measure": False,
    "dynamical_decoupling": False,
    "measurement_level": 2,
    "resilience_level": 0,
}

BACKEND_NAME = "ibm_marrakesh"
SAVE_BITSTRINGS = False

# ---------------------------------------------------------------------------
# Representative offline noise model -- dress rehearsal only; also stands in for an
# assumed ground-truth kappa_1j (the live run has none, see above).
# ---------------------------------------------------------------------------
OFFLINE_NOISE = {
    "T1": 180.0,
    "T2": 110.0,
    "dt_fixed": 1.2,       # us; prep + probe overhead, independent of the delay
    "r_1q": 3.0e-4,
    "e01": 0.012,
    "e10": 0.022,
}
OFFLINE_TRUE_KAPPA = {2: 0.05, 3: 0.03}   # assumed ground truth for the dress rehearsal


def load_env(path=ENV_PATH):
    if not os.path.exists(path):
        return
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def require_credentials():
    load_env()
    token = os.environ.get("IBM_TOKEN", "").strip()
    crn = os.environ.get("IBM_CRN", "").strip()
    missing = [n for n, v in (("IBM_TOKEN", token), ("IBM_CRN", crn)) if not v]
    if missing:
        raise RuntimeError(
            "Missing IBM credential(s): " + ", ".join(missing) + ".\n"
            f"Set them in {ENV_PATH} (shared with hardware/) or export them."
        )
    return token, crn


def total_signal_circuits():
    return len(THETA0_SWEEP) * (INSTANCES_PAIR + INSTANCES_STAR)


def total_viscal_circuits():
    return VISCAL_REPEATS * 2   # one set for pair topology, one for star
