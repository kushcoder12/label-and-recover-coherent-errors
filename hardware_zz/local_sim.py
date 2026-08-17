"""Offline noisy simulator standing in for the real device.

Reuses core.circuits.multi_qubit_zz_probe -- the SAME builder already validated by
tests/test_multiqubit_zz.py, exp11, exp12 -- plus core.noise's readout model, so this
dress rehearsal exercises identical physics to the live pipeline, not a re-derivation
of it. Given a CircuitSpec it returns counts in the raw-data schema submit -> retrieve
would produce, so submit/retrieve/analyze can be exercised end to end offline.

The live run has no ground truth for kappa_1j (it's the device's real coupling); this
offline path assumes one (config.OFFLINE_TRUE_KAPPA) purely to prove the pipeline
recovers it before spending QPU time.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import multi_qubit_zz_probe
from core.noise import NoiseProfile, apply_readout

import config


def make_offline_profile(params=None):
    p = dict(config.OFFLINE_NOISE if params is None else params)
    dt = p["dt_fixed"] + config.DELAY_S * 1.0e6      # delay_s -> us, on top of fixed overhead
    return NoiseProfile(dt=dt, T1=p["T1"], T2=p["T2"], r_1q=p["r_1q"], e01=p["e01"], e10=p["e10"])


def simulate_counts(spec, shots, kappa_true, profile=None, rng=None):
    """(counts_dict, meas_probs) for one CircuitSpec, exactly like the single-qubit
    local_sim.simulate_counts. counts_dict={"0":n0,"1":n1}, target-qubit measurement."""
    rng = np.random.default_rng() if rng is None else rng
    profile = make_offline_profile() if profile is None else profile
    kappa = {j: kappa_true[j] for j in spec.c}
    y = multi_qubit_zz_probe(spec.theta0, kappa, spec.s1, spec.c, noise=profile)

    p_true = np.clip(np.array([(1.0 + y) / 2.0, (1.0 - y) / 2.0]), 0.0, None)
    p_true = p_true / p_true.sum()
    p_meas = np.clip(apply_readout(p_true, profile, mode="probs"), 0.0, 1.0)
    p_meas = p_meas / p_meas.sum()

    n0 = int(rng.binomial(shots, p_meas[0]))
    return {"0": n0, "1": shots - n0}, p_meas


def calib_snapshot(profile, qubits):
    return {"T1": profile.T1, "T2": profile.T2, "readout_err": 0.5 * (profile.e01 + profile.e10),
           "e01": profile.e01, "e10": profile.e10, "gate_err": profile.r_1q,
           "dt": profile.dt, "qubits": qubits}


def run_offline(specs, shots=None, seed=None, kappa_true=None):
    shots = config.SHOTS if shots is None else shots
    seed = config.SEED if seed is None else seed
    kappa_true = config.OFFLINE_TRUE_KAPPA if kappa_true is None else kappa_true
    rng = np.random.default_rng(seed + 101)
    profile = make_offline_profile()

    records = []
    for spec in specs:
        counts, _ = simulate_counts(spec, shots, kappa_true, profile=profile, rng=rng)
        qubits = list(range(1 + len(spec.c)))
        rec = {
            "cid": spec.cid, "kind": spec.kind, "topology": spec.topology,
            "theta0": spec.theta0, "s1": spec.s1, "c": spec.c, "delay_s": spec.delay_s,
            "qubits": qubits, "backend": "offline_local_sim", "job_id": "offline",
            "timestamp": None, "shots": shots, "counts": counts,
            "calib": calib_snapshot(profile, qubits),
        }
        records.append(rec)
    return records
