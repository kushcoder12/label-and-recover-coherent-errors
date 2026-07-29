"""hardware/config.py — single source of truth for the sweep, paths, and options.

Implements IBM_HARDWARE_SPEC.md §1 (sweep), §2 (pipeline-correctness options),
§3 (data schema paths). No qiskit / no network here — pure configuration so both
the offline dress-rehearsal (local_sim) and the live pipeline (submit/retrieve)
read identical parameters.

Credentials are read from environment variables ONLY (§0). Never hard-code a token.
A tiny .env loader is provided so the student can keep IBM_TOKEN / IBM_CRN in
hardware/.env (git-ignored) without installing python-dotenv.
"""

import os

# ---------------------------------------------------------------------------
# Paths — mode-aware. The OFFLINE dress rehearsal writes to data/ + figures/;
# the LIVE hardware run writes to online_data/ + online_figures/ so real-device
# results never overwrite the simulated dress rehearsal. Modules that emit output
# read config.DATA_DIR / config.FIG_DIR at call time, and the live entry points
# call use_live_paths() first.
# ---------------------------------------------------------------------------
HARDWARE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HARDWARE_DIR)
ENV_PATH = os.path.join(HARDWARE_DIR, ".env")

_OFFLINE_DIRS = ("data", "figures")
_ONLINE_DIRS = ("online_data", "online_figures")


def _bind_paths(data_name, fig_name):
    """(Re)bind the module-level output paths to the given subfolders and mkdir them."""
    g = globals()
    g["DATA_DIR"] = os.path.join(HARDWARE_DIR, data_name)
    g["FIG_DIR"] = os.path.join(HARDWARE_DIR, fig_name)
    # Resumable-I/O artifacts (§2.7, §3)
    g["JOB_IDS_PATH"] = os.path.join(g["DATA_DIR"], "job_ids.json")
    g["MANIFEST_PATH"] = os.path.join(g["DATA_DIR"], "run_manifest.json")
    g["RAW_RESULTS_PATH"] = os.path.join(g["DATA_DIR"], "raw_results.json")  # one record/circuit
    g["ANALYSIS_PATH"] = os.path.join(g["DATA_DIR"], "analysis.json")
    for _d in (g["DATA_DIR"], g["FIG_DIR"]):
        os.makedirs(_d, exist_ok=True)


def use_live_paths():
    """Switch output to online_data/ + online_figures/ (the real-device run)."""
    _bind_paths(*_ONLINE_DIRS)


def use_offline_paths():
    """Switch output to data/ + figures/ (the offline dress rehearsal)."""
    _bind_paths(*_OFFLINE_DIRS)


# default: offline paths (the dress rehearsal). Live entry points call use_live_paths().
_bind_paths(*_OFFLINE_DIRS)


# ---------------------------------------------------------------------------
# Experiment grid — IBM_HARDWARE_SPEC.md §1 (Variant A, single qubit)
# ---------------------------------------------------------------------------
THETAS = [0.0, 0.05, 0.1, 0.2, 0.3]          # rad; 0.0 is the null control
DEPTHS = [1, 2, 4, 8, 16, 32]                # Trotter cycles
INSTANCES_PER_POINT = 50                     # distinct random sign vectors (spec: 40-60)
SHOTS = 3000                                 # per circuit (spec: 2000-4000)

# Per-depth visibility-calibration circuits (§4: v estimated from the data).
# Same L layers -> same decoherence as the signal circuits at that depth, but a
# KNOWN net phase pi/2 so ideal <Y> = 1 and the measured <Y> = v(L) directly.
VISCAL_REPEATS = 20
VISCAL_NET_PHASE = 3.141592653589793 / 2.0   # pi/2

# Reproducibility for the offline dress rehearsal and for choosing sign vectors.
SEED = 20260722

# Null-control tolerance (rad). At theta*=0 the recovered theta_hat must be consistent
# with zero. We use an ABSOLUTE band rather than a per-point 95% CI test: with several
# null depths, ~1-in-20 of the 95% CIs excludes zero by pure chance (multiple comparisons),
# so a per-point CI gate would flake. 0.01 rad is the physically-negligible shot+readout
# floor for this shot budget; any |theta_hat| below it is "recovered ~0" (§6 null check).
NULL_TOL = 0.01

# ---------------------------------------------------------------------------
# Hardware / runtime options — IBM_HARDWARE_SPEC.md §2
# ---------------------------------------------------------------------------
# These are asserted onto the SamplerV2 in submit.py. Recorded into the manifest.
RUNTIME_OPTIONS = {
    "optimization_level": 0,          # §2.3 — do not let the transpiler eat the circuit
    "twirling_enable_gates": False,   # §2.1 — IBM's own twirl would scramble our labels
    "twirling_enable_measure": False, # §2.1
    "dynamical_decoupling": False,    # §2.2 — we MEASURE coherent error, not suppress it
    "measurement_level": 2,           # §2.4 — per-shot bits
    "resilience_level": 0,            # §2.2 — no ZNE / PEC
}

# Physical-qubit selection (§2.5). If None, submit.py picks the best qubit from
# backend.properties() (low readout error, decent T2). Set an int to pin explicitly.
PIN_QUBIT = None

# Backend selection for the live run. None -> least-busy operational backend.
BACKEND_NAME = "ibm_marrakesh"

# Save per-shot bitstrings into the raw data (§3, optional). True is heavier on disk
# but keeps the full record; the analysis only needs counts.
SAVE_BITSTRINGS = False


# ---------------------------------------------------------------------------
# Representative offline noise model — for the dress rehearsal ONLY (§6).
# Not used on hardware; the device supplies its own noise. Values are typical of
# an IBM Eagle/Heron qubit (times in microseconds). Chosen to give a realistic
# visibility v ~ 0.9-0.97 so the debiasing step is genuinely exercised.
# ---------------------------------------------------------------------------
OFFLINE_NOISE = {
    "T1": 180.0,          # us
    "T2": 110.0,          # us  (<= 2 T1)
    "dt_fixed": 1.2,      # us  effective decoherence time of prep+probe+idle (L-independent)
    "idle_per_cycle": 0.05,  # us  small residual per-cycle idle (RZ is ~virtual, not exactly 0)
    "r_1q": 3.0e-4,       # avg single-qubit gate error
    "e01": 0.012,         # readout P(measure 1 | true 0)
    "e10": 0.022,         # readout P(measure 0 | true 1)
}


# ---------------------------------------------------------------------------
# .env loading (no external dependency) — IBM_HARDWARE_SPEC.md §0
# ---------------------------------------------------------------------------
def load_env(path=ENV_PATH):
    """Populate os.environ from hardware/.env if present (does not overwrite existing).

    Lines are KEY=VALUE; blank lines and #-comments ignored. Values may be quoted.
    Silently no-ops when the file is absent so the offline pipeline never needs it.
    """
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
    """Return (token, crn), loading hardware/.env first. Fails loudly if unset (§0)."""
    load_env()
    token = os.environ.get("IBM_TOKEN", "").strip()
    crn = os.environ.get("IBM_CRN", "").strip()
    missing = [n for n, v in (("IBM_TOKEN", token), ("IBM_CRN", crn)) if not v]
    if missing:
        raise RuntimeError(
            "Missing IBM credential(s): " + ", ".join(missing) + ".\n"
            "Set them in hardware/.env (see hardware/.env.example) or export them:\n"
            "  export IBM_TOKEN=...   # 44-char API key\n"
            "  export IBM_CRN=...     # IBM Cloud CRN"
        )
    return token, crn


def total_signal_circuits():
    return len(THETAS) * len(DEPTHS) * INSTANCES_PER_POINT


def total_viscal_circuits():
    return len(DEPTHS) * VISCAL_REPEATS
