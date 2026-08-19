"""Live characterization of drive-induced leakage phi_1j on real hardware (DRIVE_CROSSTALK_
SPEC.md Sec 8), extending the static-ZZ result (build_circuits.py / Parts 1-2 above) with the
third label p_j.

Self-contained, mirroring diagnostic_delay_sweep.py's pattern (own paths, own circuit builder,
own CLI) rather than editing build_circuits.py/submit.py/retrieve.py/analyze.py in place: those
already carry the committed, working Parts 1-3 pipeline and its data, and this is a genuinely
separate circuit (adds padding pulses, no accumulation delay) recovering a different parameter.
Nothing here reads or writes online_data/{raw_results,analysis,job_ids,run_manifest}.json or
diagnostic_delay_sweep_*.json -- all paths below are new.

Design decisions, and why (build on what Parts 2-4 already found on this exact device/qubits):

  - Qubits: reuses [53, 54] (target=53, neighbor=54) from Parts 2-4, unchanged. Not re-picked
    live -- this is the SAME pair already calibrated and characterized above, so results are
    directly comparable, and one less unknown going into a first phi_1j attempt.
  - Delay: 0. Part 4 isolated a real, delay-dependent idle drift on qubit 53 (not from drive
    crosstalk) that grows from ~-0.05 rad at 1us to ~-0.15 rad at 8us. phi_1j comes from REAL
    PULSE COUNT, not idle time, so removing the delay window removes that confound entirely
    instead of fighting it -- kappa_1j (which DOES need idle time to accumulate) is not the
    target of this run and is expected to come back near zero, which is itself a consistency
    check on the delay=0 design, not a failure.
  - Topology: pair only (single neighbor), matching the spec's construction (Sec 1-2 is written
    for one neighbor j); star/multi-neighbor padding is a natural extension left for later.
  - theta0 stays the genuine positive control (Parts 1-2's convention); kappa_1j and phi_1j have
    no ground truth on real hardware and are reported with bootstrap CIs and K_disagreement,
    never against a truth line (Sec 6/8 honesty requirement, carried over unchanged).

Per-shot circuit (physical qubit 0 = target/53, 1 = neighbor/54, via initial_layout):
    1. H on target                              -> |+>
    2. X on neighbor iff c = -1                  -> real state prep (sp_pulses)
    3. X on neighbor TWICE MORE iff p = -1       -> padding (pad_pulses), immediately after (2)
    4. RZ(s1 * theta0) on target                 -> qubit 1's own known, injected knob
    5. SDG, H on target                          -> rotates Y into Z
    6. measure target

No delay stage (see above) -- step 4 immediately follows step 3, matching the spec's "immediately
after the c_j-required preparation, fire ... two more times" ordering for the padding itself,
and putting the target's own injected RZ right after so nothing sits idle waiting on the neighbor.

No qiskit at import time (offline-checkable); qiskit only inside validate/submit/retrieve.

Usage:
  python drive_crosstalk_hardware.py --check      # structural self-check, no qiskit/network
  python drive_crosstalk_hardware.py --offline     # dress rehearsal on the local noisy sim
  python drive_crosstalk_hardware.py --validate    # dry-run against the live backend, no QPU
  python drive_crosstalk_hardware.py --submit      # submit for real
  python drive_crosstalk_hardware.py --retrieve    # fetch + analyze + figure
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict

import numpy as np

import config

sys.path.insert(0, config.REPO_ROOT)
sys.path.insert(0, os.path.join(config.REPO_ROOT, "experiments"))   # for _style (figures)

TARGET_QUBIT = 53
NEIGHBOR_QUBIT = 54

THETA0_SWEEP_DC = [0.0, 0.10, 0.20]
INSTANCES = 32           # ~16 per p_j stratum per theta0 point
VISCAL_REPEATS = 20
SHOTS = 2000
SEED = 20260817

# Assumed ground truth for the OFFLINE dress rehearsal only -- the live run has none.
OFFLINE_TRUE_KAPPA = 0.0     # delay=0, so the real coupling has no time to accumulate here
OFFLINE_TRUE_PHI = 0.05

DATA_DIR = os.path.join(config.HARDWARE_ZZ_DIR, "online_data")
FIG_DIR = os.path.join(config.HARDWARE_ZZ_DIR, "online_figures")
OFFLINE_DIR = os.path.join(config.HARDWARE_ZZ_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OFFLINE_DIR, exist_ok=True)

JOB_IDS_PATH = os.path.join(DATA_DIR, "drive_crosstalk_job_ids.json")
RAW_PATH = os.path.join(DATA_DIR, "drive_crosstalk_raw.json")
ANALYSIS_PATH = os.path.join(DATA_DIR, "drive_crosstalk_analysis.json")
OFFLINE_RAW_PATH = os.path.join(OFFLINE_DIR, "drive_crosstalk_offline_raw.json")
OFFLINE_ANALYSIS_PATH = os.path.join(OFFLINE_DIR, "drive_crosstalk_offline_analysis.json")
FIG_PATH = os.path.join(FIG_DIR, "fig19_drive_crosstalk_hardware")
OFFLINE_FIG_PATH = os.path.join(config.REPO_ROOT, "figures", "fig19_drive_crosstalk_hardware_offline")


@dataclass
class DriveCrosstalkSpec:
    cid: str
    kind: str          # "signal" or "viscal"
    theta0: float
    s1: int
    c: int              # neighbor's state-prep label, +-1
    p: int               # padding label, +-1
    meta: dict = field(default_factory=dict)

    def to_record(self):
        return asdict(self)


def _rng(seed):
    return np.random.default_rng(seed)


def generate_signal_specs(thetas=None, instances=None, seed=None):
    thetas = THETA0_SWEEP_DC if thetas is None else thetas
    instances = INSTANCES if instances is None else instances
    seed = SEED if seed is None else seed
    rng = _rng(seed)
    specs = []
    for theta0 in thetas:
        for i in range(instances):
            s1 = int(rng.choice([-1, 1]))
            c = int(rng.choice([-1, 1]))
            p = int(rng.choice([-1, 1]))
            cid = f"dcsig_t{theta0:.2f}_i{i:03d}"
            specs.append(DriveCrosstalkSpec(cid=cid, kind="signal", theta0=float(theta0),
                                            s1=s1, c=c, p=p))
    return specs


def generate_viscal_specs(repeats=None, seed=None):
    """Visibility reference at theta0=pi/2, s1=1, c=1, p=1 -- n_j=0 at this point (no real
    pulse fires at all), the cleanest possible calibration: nothing to debias the leakage
    term against, only decoherence + readout."""
    repeats = VISCAL_REPEATS if repeats is None else repeats
    seed = (SEED + 777) if seed is None else seed
    specs = []
    for r in range(repeats):
        cid = f"dcvis_r{r:03d}"
        specs.append(DriveCrosstalkSpec(cid=cid, kind="viscal", theta0=math.pi / 2,
                                        s1=1, c=1, p=1, meta={"net_phase": math.pi / 2}))
    return specs


def generate_all_specs():
    return generate_signal_specs() + generate_viscal_specs()


def spec_integrity_ok(spec):
    if spec.s1 not in (1, -1):
        return False
    if spec.c not in (1, -1):
        return False
    if spec.p not in (1, -1):
        return False
    return True


def n_real_pulses(c, p):
    sp_pulses = 0 if c == 1 else 1
    pad_pulses = 0 if p == 1 else 2
    return sp_pulses + pad_pulses


# ---------------------------------------------------------------------------
# qiskit builder (LAZY import) -- only needed for validate/submit
# ---------------------------------------------------------------------------
def build_qiskit_circuit(spec):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 1, name=spec.cid)
    qc.h(0)
    qc.barrier()
    if spec.c == -1:
        qc.x(1)
    if spec.p == -1:
        qc.x(1)
        qc.x(1)
    qc.barrier()
    qc.rz(spec.s1 * spec.theta0, 0)
    qc.barrier()
    qc.sdg(0)
    qc.h(0)
    qc.barrier()
    qc.measure(0, 0)
    return qc


def assert_integrity(transpiled_qc, spec, target_physical_qubit, atol=1e-9):
    """Confirm the injected RZ(s1*theta0) survived transpilation intact, and the barrier
    count matches the fixed 4-stage structure (no delay stage in this circuit)."""
    n_barriers = sum(1 for ci in transpiled_qc.data if ci.operation.name == "barrier")
    if n_barriers != 4:
        raise AssertionError(f"[{spec.cid}] {n_barriers} barriers, expected 4. Run INVALID.")

    if abs(spec.theta0) > atol:
        target_angle = round(abs((spec.s1 * spec.theta0 + math.pi) % (2 * math.pi) - math.pi), 12)
        found = False
        for ci in transpiled_qc.data:
            if ci.operation.name != "rz":
                continue
            ang = float(ci.operation.params[0])
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            if round(abs(ang), 12) == target_angle:
                found = True
                break
        if not found:
            raise AssertionError(
                f"[{spec.cid}] injected RZ({spec.s1*spec.theta0}) not found intact. Run INVALID.")
    return True


def structural_check():
    """No qiskit / no network: confirm spec generation and n_real_pulses table are sane."""
    specs = generate_all_specs()
    n_sig = sum(1 for s in specs if s.kind == "signal")
    n_vis = sum(1 for s in specs if s.kind == "viscal")
    bad = [s.cid for s in specs if not spec_integrity_ok(s)]
    assert not bad, bad
    assert n_real_pulses(1, 1) == 0
    assert n_real_pulses(1, -1) == 2
    assert n_real_pulses(-1, 1) == 1
    assert n_real_pulses(-1, -1) == 3
    for theta0 in THETA0_SWEEP_DC:
        both = {1: 0, -1: 0}
        for s in specs:
            if s.kind == "signal" and abs(s.theta0 - theta0) < 1e-9:
                both[s.p] += 1
        assert both[1] > 0 and both[-1] > 0, (theta0, both)
    print(f"generated {len(specs)} circuits: {n_sig} signal + {n_vis} viscal")
    print(f"target={TARGET_QUBIT}, neighbor={NEIGHBOR_QUBIT}, delay=0, "
          f"thetas={THETA0_SWEEP_DC}, instances={INSTANCES}, shots={SHOTS}")
    print("spec-integrity, n_real_pulses table, and both-p-strata-present: all OK.")


# ---------------------------------------------------------------------------
# Offline dress rehearsal -- reuses core.circuits.three_label_probe + core.noise, the
# SAME builder validated by tests/test_drive_crosstalk.py, exp14-16 (Oracles DC1-DC3),
# not a re-derivation of it. Proves this pipeline recovers assumed truth BEFORE any
# QPU time is spent, same discipline as run_all.py's run_offline() above.
# ---------------------------------------------------------------------------
def make_offline_profile():
    from core.noise import NoiseProfile
    return NoiseProfile(dt=1.2, T1=180.0, T2=110.0, r_1q=3.0e-4, e01=0.012, e10=0.022)


def simulate_counts(spec, shots, kappa_true, phi_true, profile, rng):
    from core.circuits import three_label_probe
    from core.noise import apply_readout

    y = three_label_probe(spec.theta0, kappa_true, phi_true, spec.s1, spec.c, spec.p,
                          noise=profile)
    p_true = np.clip(np.array([(1.0 + y) / 2.0, (1.0 - y) / 2.0]), 0.0, None)
    p_true = p_true / p_true.sum()
    p_meas = np.clip(apply_readout(p_true, profile, mode="probs"), 0.0, 1.0)
    p_meas = p_meas / p_meas.sum()
    n0 = int(rng.binomial(shots, p_meas[0]))
    return {"0": n0, "1": shots - n0}


def run_offline(specs, shots=SHOTS, seed=SEED, kappa_true=OFFLINE_TRUE_KAPPA,
                phi_true=OFFLINE_TRUE_PHI):
    rng = np.random.default_rng(seed + 101)
    profile = make_offline_profile()
    records = []
    for spec in specs:
        counts = simulate_counts(spec, shots, kappa_true, phi_true, profile, rng)
        records.append({
            "cid": spec.cid, "kind": spec.kind, "theta0": spec.theta0, "s1": spec.s1,
            "c": spec.c, "p": spec.p, "qubits": [TARGET_QUBIT, NEIGHBOR_QUBIT],
            "backend": "offline_local_sim", "job_id": "offline", "shots": shots,
            "counts": counts,
        })
    return records


# ---------------------------------------------------------------------------
# Analysis -- reuses core.multiqubit_zz.stratified_three_label_estimator (the SAME
# stratify/combine machinery validated by Oracles DC1-DC3), never reimplemented here.
# ---------------------------------------------------------------------------
def expval_Y(counts):
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    tot = n0 + n1
    return (n0 - n1) / tot if tot else 0.0


def _group(records):
    signal, viscal = {}, []
    for r in records:
        y = expval_Y(r["counts"])
        if r["kind"] == "viscal":
            viscal.append(y)
        else:
            key = round(float(r["theta0"]), 6)
            signal.setdefault(key, []).append((r["s1"], r["c"], r["p"], y))
    return signal, viscal


def visibility(viscal_ys):
    if not viscal_ys:
        return 1.0, 0.0
    ys = np.asarray(viscal_ys, float)
    v = float(np.clip(ys.mean(), 1e-6, 1.0))
    sem = float(ys.std(ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0
    return v, sem


def analyze(records, B=1000, seed=None):
    from core.multiqubit_zz import stratified_three_label_estimator

    seed = SEED if seed is None else seed
    signal, viscal = _group(records)
    v, v_sem = visibility(viscal)

    by_theta = {}
    for theta0, rows in sorted(signal.items()):
        recs = [(s1, c, p, y_i / v) for (s1, c, p, y_i) in rows]
        fit = stratified_three_label_estimator(recs, B=B,
                                               seed=seed + int(round(theta0 * 1000)))
        by_theta[f"theta{theta0:.3f}"] = {"theta0_star": theta0, "v": v, "v_sem": v_sem,
                                          "n_instances": len(rows), **fit}
    return {
        "results": by_theta,
        "grid": {"thetas": THETA0_SWEEP_DC, "instances": INSTANCES, "shots": SHOTS,
                "viscal_repeats": VISCAL_REPEATS, "delay_s": 0.0,
                "target_qubit": TARGET_QUBIT, "neighbor_qubit": NEIGHBOR_QUBIT},
        "backend": records[0].get("backend", "unknown") if records else "unknown",
    }


NULL_TOL = 0.03    # looser than the main run's 0.01: half the instance count, delay=0
                   # circuit is structurally different, no prior baseline to calibrate against


def _positive_control(analysis):
    print("\n=== POSITIVE-CONTROL CHECKS ===")
    ok = True
    for key, r in analysis["results"].items():
        if abs(r["theta0_star"]) < 1e-9:
            passed = abs(r["theta_1_0_hat"]) <= NULL_TOL
            ok = ok and passed
            print(f"  null theta0*=0: theta_1_0_hat={r['theta_1_0_hat']:+.4f}  "
                  f"{'PASS' if passed else 'FAIL'}  (|.|<{NULL_TOL})")
    diag_ok = True
    for key, r in analysis["results"].items():
        if r["theta0_star"] > 0:
            lo, hi = r["bootstrap_ci"]["theta_1_0"]
            half = 0.5 * (hi - lo)
            near = abs(r["theta_1_0_hat"] - r["theta0_star"]) <= max(3 * half, 0.05)
            diag_ok = diag_ok and near
    print(f"  diagonal recovery: {'PASS' if diag_ok else 'CHECK'}")
    ok = ok and diag_ok

    first = next(iter(analysis["results"].values()))
    kappa_err = abs(first["kappa_1j_hat"] - OFFLINE_TRUE_KAPPA)
    lo, hi = first["bootstrap_ci"]["kappa_1j"]
    kappa_ok = kappa_err <= max(3 * 0.5 * (hi - lo), 0.03)
    ok = ok and kappa_ok
    print(f"  kappa_1j: hat={first['kappa_1j_hat']:+.4f} assumed-true={OFFLINE_TRUE_KAPPA:.4f} "
          f"|err|={kappa_err:.4f}  {'PASS' if kappa_ok else 'CHECK'}")

    phi_err = abs(first["phi_1j_hat"] - OFFLINE_TRUE_PHI)
    lo, hi = first["bootstrap_ci"]["phi_1j"]
    phi_ok = phi_err <= max(3 * 0.5 * (hi - lo), 0.03)
    ok = ok and phi_ok
    print(f"  phi_1j: hat={first['phi_1j_hat']:+.4f} assumed-true={OFFLINE_TRUE_PHI:.4f} "
          f"|err|={phi_err:.4f}  {'PASS' if phi_ok else 'CHECK'}")

    print(f"\nDRESS REHEARSAL: "
          f"{'PASS — pipeline recovers theta0, kappa, and phi; safe to run live.' if ok else 'FAIL — fix before spending QPU time.'}")
    if not ok:
        raise SystemExit(1)
    return ok


# ---------------------------------------------------------------------------
# Figure 19
# ---------------------------------------------------------------------------
def make_figure(analysis, path):
    from _style import setup
    C = setup()
    import matplotlib.pyplot as plt

    res = analysis["results"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 5.2))

    lo, hi = min(THETA0_SWEEP_DC), max(THETA0_SWEEP_DC)
    axA.plot([lo, hi], [lo, hi], "--", color=C["black"], lw=1.2, label="ideal recovery")
    xs, ys, elo, ehi = [], [], [], []
    for key, r in sorted(res.items(), key=lambda kv: kv[1]["theta0_star"]):
        xs.append(r["theta0_star"]); ys.append(r["theta_1_0_hat"])
        clo, chi = r["bootstrap_ci"]["theta_1_0"]
        elo.append(r["theta_1_0_hat"] - clo); ehi.append(chi - r["theta_1_0_hat"])
    axA.errorbar(xs, ys, yerr=[elo, ehi], fmt="o", color=C["blue"], ms=7, mec="k", mew=0.5,
                capsize=3, lw=1.3, label=r"recovered $\hat\theta_1^{(0)}$")
    axA.set_xlabel(r"injected  $\theta_1^{(0)}$  (rad)")
    axA.set_ylabel(r"recovered  $\hat\theta_1^{(0)}$  (rad)")
    axA.set_title("(a) Positive control: known injected phase", fontsize=11)
    axA.legend(fontsize=9, loc="upper left")

    labels, vals, elo2, ehi2 = [], [], [], []
    for name, key in [(r"$\kappa_{1j}$" "\n(qubit 54)", "kappa_1j_hat"),
                      (r"$\phi_{1j}$" "\n(qubit 54)", "phi_1j_hat")]:
        ks, ws = [], []
        for r in res.values():
            kh = r[key]
            clo, chi = r["bootstrap_ci"][key.replace("_hat", "")]
            w = 1.0 / max((chi - clo), 1e-6) ** 2
            ks.append(kh); ws.append(w)
        ks, ws = np.array(ks), np.array(ws)
        kbar = float(np.sum(ks * ws) / np.sum(ws))
        spread = float(np.sqrt(1.0 / np.sum(ws)))
        labels.append(name); vals.append(kbar); elo2.append(spread); ehi2.append(spread)
    x = np.arange(len(labels))
    axB.errorbar(x, vals, yerr=[elo2, ehi2], fmt="D", color=C["vermil"], ms=9, mec="k",
                mew=0.5, capsize=4, lw=1.4)
    axB.axhline(0.0, color="grey", lw=0.8, alpha=0.6)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=10)
    axB.set_ylabel("recovered value (rad)")
    axB.set_title("(b) Real static-ZZ + drive-leakage\n(characterization, no ground truth)",
                 fontsize=11)

    fig.suptitle("Fig 19 — Drive-crosstalk " r"($\phi_{1j}$)" " recovered on real hardware "
                f"(qubits {TARGET_QUBIT}/{NEIGHBOR_QUBIT}, delay=0)", fontsize=12.5, y=1.04)
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path + ".png", bbox_inches="tight", pad_inches=0.05, dpi=300)
    print(f"  wrote {path}.pdf")
    print(f"  wrote {path}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Offline dress rehearsal entry point
# ---------------------------------------------------------------------------
def run_offline_rehearsal():
    print("=== OFFLINE DRESS REHEARSAL (local noisy sim; reuses core/) ===")
    structural_check()
    specs = generate_all_specs()
    records = run_offline(specs)
    with open(OFFLINE_RAW_PATH, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"simulated {len(records)} circuits -> {OFFLINE_RAW_PATH}")

    analysis = analyze(records)
    with open(OFFLINE_ANALYSIS_PATH, "w") as fh:
        json.dump(analysis, fh, indent=2)
    for key, r in sorted(analysis["results"].items(), key=lambda kv: kv[1]["theta0_star"]):
        print(f"  theta0*={r['theta0_star']:.3f}: theta_1_0_hat={r['theta_1_0_hat']:+.4f} "
              f"kappa_1j_hat={r['kappa_1j_hat']:+.4f} phi_1j_hat={r['phi_1j_hat']:+.4f} "
              f"K_disagreement={r['K_disagreement']:.2e}")
    _positive_control(analysis)
    make_figure(analysis, OFFLINE_FIG_PATH)


# ---------------------------------------------------------------------------
# Live path
# ---------------------------------------------------------------------------
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

    service = _service()
    backend = service.backend(config.BACKEND_NAME) if config.BACKEND_NAME else \
        service.least_busy(operational=True, simulator=False, min_num_qubits=2)

    adj = {}
    for a, b in backend.coupling_map.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    if NEIGHBOR_QUBIT not in adj.get(TARGET_QUBIT, set()):
        raise RuntimeError(f"qubits {TARGET_QUBIT}/{NEIGHBOR_QUBIT} are not coupled on "
                           f"{backend.name}'s current coupling map -- device topology may "
                           f"have changed since Parts 2-4.")

    specs = generate_all_specs()
    circuits = [build_qiskit_circuit(s) for s in specs]
    print(f"transpiling {len(circuits)} circuits onto [{TARGET_QUBIT},{NEIGHBOR_QUBIT}] "
          f"(opt_level=0) + integrity asserts...")
    isa = _transpile(circuits, specs, backend)
    print("transpiler-integrity: all injected RZ layers intact.")

    if validate_only:
        sampler = _configure_sampler(Sampler(mode=backend))
        o = sampler.options
        assert o.twirling.enable_gates is False and o.twirling.enable_measure is False
        assert o.dynamical_decoupling.enable is False
        print(f"VALIDATE OK: {len(circuits)} circuits transpile+assert clean; "
              f"IBM twirling OFF, DD OFF, on {backend.name}. No QPU spent.")
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

    specs = {s.cid: s for s in generate_all_specs()}
    records = []
    for pub_result, cid in zip(result, jm["cids"]):
        spec = specs[cid]
        counts = _counts_from_pub(pub_result)
        records.append({"cid": cid, "kind": spec.kind, "theta0": spec.theta0, "s1": spec.s1,
                        "c": spec.c, "p": spec.p, "qubits": [TARGET_QUBIT, NEIGHBOR_QUBIT],
                        "backend": jm["backend"], "job_id": jm["job_id"], "shots": jm["shots"],
                        "counts": counts})
    with open(RAW_PATH, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"wrote {len(records)} records -> {RAW_PATH}")
    return records


def main():
    args = set(sys.argv[1:])
    if "--help" in args or "-h" in args:
        print(__doc__)
        return
    if "--check" in args:
        structural_check()
        return
    if "--offline" in args or not args:
        run_offline_rehearsal()
        return
    if "--validate" in args:
        run_validate_or_submit(validate_only=True)
        return
    if "--submit" in args:
        run_validate_or_submit(validate_only=False)
        return
    if "--retrieve" in args:
        records = retrieve()
        analysis = analyze(records)
        with open(ANALYSIS_PATH, "w") as fh:
            json.dump(analysis, fh, indent=2)
        print(f"backend: {analysis['backend']}")
        for key, r in sorted(analysis["results"].items(), key=lambda kv: kv[1]["theta0_star"]):
            lo, hi = r["bootstrap_ci"]["theta_1_0"]
            flag = ""
            if abs(r["theta0_star"]) < 1e-9 and abs(r["theta_1_0_hat"]) > NULL_TOL:
                flag = "  <-- NULL CONTROL FAIL"
            print(f"  theta0*={r['theta0_star']:.3f}: theta_1_0_hat={r['theta_1_0_hat']:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]  v={r['v']:.3f}{flag}")
            klo, khi = r["bootstrap_ci"]["kappa_1j"]
            plo, phi = r["bootstrap_ci"]["phi_1j"]
            print(f"      kappa_1j_hat={r['kappa_1j_hat']:+.4f} [{klo:+.4f},{khi:+.4f}]  "
                  f"phi_1j_hat={r['phi_1j_hat']:+.4f} [{plo:+.4f},{phi:+.4f}]  "
                  f"K_disagreement={r['K_disagreement']:.2e}")
        make_figure(analysis, FIG_PATH)
        print(f"\nanalysis -> {ANALYSIS_PATH}")
        print(f"figure -> {FIG_PATH}.png/.pdf")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
