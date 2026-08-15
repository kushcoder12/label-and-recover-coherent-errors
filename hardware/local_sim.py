"""Offline noisy statevector/DM simulator standing in for the real device.

Before spending QPU time, this runs the whole pipeline on a local noisy
statevector sim (reuse core + noise) and confirms it recovers theta*. Given a
CircuitSpec it returns counts in the same raw-data schema as the real device,
so submit -> retrieve -> analyze can be exercised end to end with no qiskit
and no network.

Reuses the project engine (numpy only): core.single_qubit_gates, core.operator_on_state,
core.noise (NoiseProfile, apply_incoherent_layer, apply_readout).

Physics of the model (why it mirrors the device):
  - The signal sin(theta* * sum s) lives in the x-y plane after the RZ layers (an
    off-diagonal coherence). We apply ONE incoherent layer THERE (dephasing/T1/gate error),
    which shrinks the coherence -> the visibility v < 1. Then the SDG,H basis change maps Y
    onto Z, and readout error (e01,e10) further biases the two populations.
  - RZ is virtual on IBM (~zero duration), so depth L adds little decoherence; we model a
    small residual per-cycle idle so v(L) drifts gently with depth (the T2 "ceiling").
The device's T2/readout only shrink the label-outcome correlation (visibility v); debiasing
by the measured v(L) recovers theta*. This is the dress rehearsal.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.single_qubit_gates import Hadamard, Rz, P_gate
from core.operator_on_state import single_qubit_op
from core.noise import NoiseProfile, apply_incoherent_layer, apply_readout

import config

_H = Hadamard()
_SDG = P_gate(-np.pi / 2)          # S-dagger = diag(1, -i)
_BASIS_CHANGE = _H @ _SDG          # U with U^dag Z U = Y  (measure <Y> in the Z basis)


def make_offline_profile(L, params=None):
    """NoiseProfile for a depth-L circuit under the representative offline model (config).

    Effective decoherence time dt(L) = dt_fixed + L*idle_per_cycle (RZ ~virtual, so depth
    adds only a small residual idle). Times in microseconds (consistent T1/T2/dt units).
    """
    p = dict(config.OFFLINE_NOISE if params is None else params)
    dt = p["dt_fixed"] + L * p["idle_per_cycle"]
    return NoiseProfile(
        dt=dt, T1=p["T1"], T2=p["T2"], r_1q=p["r_1q"], e01=p["e01"], e10=p["e10"],
    )


def _ideal_pre_measure_state(spec):
    """State after H and all RZ layers (signal still in the x-y plane), n=1."""
    state = single_qubit_op(np.array([1, 0], dtype=complex), _H, 1, 1)  # |+>
    total_angle = float(np.sum(spec.layer_angles))
    state = single_qubit_op(state, Rz(total_angle), 1, 1)               # RZ layers commute
    return state


def simulate_counts(spec, shots, profile=None, rng=None):
    """Return (counts_dict, meas_probs) for one CircuitSpec under the offline noise model.

    counts_dict = {"0": n0, "1": n1}; n0+n1 = shots. Mirrors the device's per-circuit output.
    """
    if rng is None:
        rng = np.random.default_rng()
    if profile is None:
        profile = make_offline_profile(spec.L)

    # 1-2. prep + RZ layers -> signal in x-y plane
    state = _ideal_pre_measure_state(spec)
    rho = np.outer(state, state.conj())

    # decohere WHERE the signal is transverse -> visibility loss (T2, gate error, T1)
    rho = apply_incoherent_layer(rho, profile, n=1, mode="dm")

    # 3. basis change SDG,H as a unitary on the density matrix: rho -> U rho U^dag
    U = _BASIS_CHANGE
    rho = U @ rho @ U.conj().T

    # 4. measurement populations (true), then asymmetric readout error
    p_true = np.clip(np.array([rho[0, 0].real, rho[1, 1].real]), 0.0, None)
    p_true = p_true / p_true.sum()
    p_meas = apply_readout(p_true, profile, mode="probs")
    p_meas = np.clip(p_meas, 0.0, 1.0)
    p_meas = p_meas / p_meas.sum()

    n0 = int(rng.binomial(shots, p_meas[0]))
    n1 = shots - n0
    return {"0": n0, "1": n1}, p_meas


def calib_snapshot(profile, qubit=0):
    """Per-qubit calibration snapshot in the raw-data schema (offline surrogate for
    backend.properties())."""
    return {
        "T1": profile.T1, "T2": profile.T2,
        "readout_err": 0.5 * (profile.e01 + profile.e10),
        "e01": profile.e01, "e10": profile.e10,
        "gate_err": profile.r_1q, "dt": profile.dt, "qubit": qubit,
    }


def run_offline(specs, shots=None, seed=None, save_bitstrings=None):
    """Execute all specs on the local simulator -> list of raw records.

    This is exactly what retrieve.py writes after a live run, so analyze.py is agnostic to
    whether the data came from the device or from here.
    """
    shots = config.SHOTS if shots is None else shots
    seed = config.SEED if seed is None else seed
    save_bitstrings = config.SAVE_BITSTRINGS if save_bitstrings is None else save_bitstrings
    rng = np.random.default_rng(seed + 101)

    records = []
    for spec in specs:
        profile = make_offline_profile(spec.L)
        counts, _ = simulate_counts(spec, shots, profile=profile, rng=rng)
        rec = {
            "cid": spec.cid, "kind": spec.kind,
            "theta_star": spec.theta_star, "L": spec.L,
            "sign_vector": spec.sign_vector, "sum_signs": spec.sum_signs,
            "qubit": 0, "backend": "offline_local_sim", "job_id": "offline",
            "timestamp": None, "shots": shots, "counts": counts,
            "calib": calib_snapshot(profile),
        }
        if save_bitstrings:
            bits = ["0"] * counts["0"] + ["1"] * counts["1"]
            rng.shuffle(bits)
            rec["bitstrings"] = bits
        records.append(rec)
    return records
