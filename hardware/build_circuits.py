"""hardware/build_circuits.py — owned-label circuit specifications (IBM_HARDWARE_SPEC.md §1).

THE POINT (spec preamble): we generate ONE circuit per twirl instance and record the
exact per-cycle sign vector s = (s_1..s_L) ourselves, so every per-layer label is OWNED
and never lost. The public CB dataset kept only the NET frame; this fixes that.

Variant A per-shot circuit for (depth L, sign vector s, injected theta*):
    1. H                       -> |+>  (Bloch +x)
    2. RZ(s_l * theta*), l=1..L (RZ is virtual/exact on IBM -> clean known error)
    3. SDG, then H             -> rotates Y into Z (measures <Y>)   [== RX(pi/2) up to phase]
    4. measure

Ideal (noiseless) statistics: <Y> = sin(theta* * sum_l s_l).

This module is PURE PYTHON / numpy (no qiskit) so it runs in the offline sandbox. The
qiskit QuantumCircuit builder and the transpiler-integrity assert (§2.3) live here too but
import qiskit LAZILY, so they only require qiskit when actually building for hardware.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np

import config


@dataclass
class CircuitSpec:
    """One circuit. `layer_angles[l]` is the RZ angle physically applied at cycle l.

    kind="signal": layer_angles = s_l * theta_star (the owned twirl labels are in sign_vector).
    kind="viscal": a visibility reference at the same depth L with a KNOWN net phase pi/2,
                   so ideal <Y> = 1 and measured <Y> = v(L). sign_vector is all +1 there.
    """
    cid: str
    kind: str
    theta_star: float
    L: int
    sign_vector: List[int]
    layer_angles: List[float]
    sum_signs: int
    meta: dict = field(default_factory=dict)

    def to_record(self):
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Spec generation (owned labels)
# ---------------------------------------------------------------------------
def _rng(seed):
    return np.random.default_rng(seed)


def generate_signal_specs(thetas=None, depths=None, instances=None, seed=None):
    """All signal circuits over the (theta*, L) grid, `instances` twirl vectors each.

    Sign vectors are drawn i.i.d. uniform +/-1 (the fully-twirled RC limit, p=0.5). Every
    vector is stored verbatim in the spec -> the labels are owned, never averaged away.
    """
    thetas = config.THETAS if thetas is None else thetas
    depths = config.DEPTHS if depths is None else depths
    instances = config.INSTANCES_PER_POINT if instances is None else instances
    seed = config.SEED if seed is None else seed
    rng = _rng(seed)

    specs = []
    for theta in thetas:
        for L in depths:
            for i in range(instances):
                s = rng.choice([-1, 1], size=L).astype(int)
                angles = [float(theta * sk) for sk in s]
                cid = f"sig_t{theta:.2f}_L{L}_i{i:03d}"
                specs.append(CircuitSpec(
                    cid=cid, kind="signal", theta_star=float(theta), L=int(L),
                    sign_vector=[int(x) for x in s], layer_angles=angles,
                    sum_signs=int(s.sum()),
                ))
    return specs


def generate_viscal_specs(depths=None, repeats=None, seed=None):
    """Per-depth visibility references (net phase pi/2 spread over L layers).

    Same L layers -> same accumulated decoherence as the signal circuits at that depth,
    so the measured <Y> directly estimates the visibility v(L) used to debias (§4).
    """
    depths = config.DEPTHS if depths is None else depths
    repeats = config.VISCAL_REPEATS if repeats is None else repeats
    seed = (config.SEED + 777) if seed is None else seed

    specs = []
    for L in depths:
        per_layer = config.VISCAL_NET_PHASE / L      # net = pi/2 exactly
        for r in range(repeats):
            cid = f"vis_L{L}_r{r:03d}"
            specs.append(CircuitSpec(
                cid=cid, kind="viscal", theta_star=float("nan"), L=int(L),
                sign_vector=[1] * L, layer_angles=[float(per_layer)] * L,
                sum_signs=L, meta={"net_phase": config.VISCAL_NET_PHASE},
            ))
    return specs


def generate_all_specs():
    """Full circuit list: signal grid + visibility references."""
    return generate_signal_specs() + generate_viscal_specs()


def spec_integrity_ok(spec):
    """Spec-level analog of the transpiler-integrity assert (§2.3): confirm the spec still
    carries exactly L injected phase layers with the intended per-layer angles.

    This is the numpy-only guard used by the offline dress rehearsal. The qiskit-level
    assert on the *transpiled* circuit is assert_injected_layers_intact() below.
    """
    if len(spec.layer_angles) != spec.L:
        return False
    if len(spec.sign_vector) != spec.L:
        return False
    if spec.kind == "signal":
        for sk, ang in zip(spec.sign_vector, spec.layer_angles):
            if not math.isclose(ang, sk * spec.theta_star, abs_tol=1e-12):
                return False
    return True


# ---------------------------------------------------------------------------
# qiskit builder (LAZY import) — only needed for the live hardware run
# ---------------------------------------------------------------------------
def build_qiskit_circuit(spec):
    """Return a 1-qubit, 1-clbit qiskit QuantumCircuit implementing `spec`.

    barrier() is placed around EVERY injected-RZ and the prep/probe (§2.3) so that
    optimization_level=0 cannot merge or drop the injected phase layers.
    """
    from qiskit import QuantumCircuit   # lazy: not needed offline

    qc = QuantumCircuit(1, 1, name=spec.cid)
    qc.h(0)
    qc.barrier()
    for ang in spec.layer_angles:
        qc.rz(ang, 0)
        qc.barrier()           # one barrier per injected layer -> countable, unmergeable
    qc.sdg(0)
    qc.h(0)
    qc.barrier()
    qc.measure(0, 0)
    return qc


def assert_injected_layers_intact(transpiled_qc, spec, atol=1e-9):
    """§2.3 (#1 failure mode): confirm the transpiled circuit STILL has L injected phase
    layers. Raises AssertionError if the transpiler merged/removed them.

    Strategy: at optimization_level=0 with per-layer barriers, each injected rz(s_l*theta*)
    survives as an `rz` with its exact angle. We count rz instructions whose angle matches
    one of the injected angles. H and SDG also decompose to rz on IBM basis, so we exclude
    those by matching the *specific* injected angle magnitude(s), and additionally require
    the barrier count to be consistent. For the theta*=0 null the injected angle is 0 and
    indistinguishable from basis rz(0); there we fall back to counting barriers.
    """
    # Count barriers first (each injected layer contributes exactly one; prep + probe add 2).
    n_barriers = sum(1 for ci in transpiled_qc.data if ci.operation.name == "barrier")
    expected_barriers = spec.L + 2
    if n_barriers != expected_barriers:
        raise AssertionError(
            f"[{spec.cid}] transpiler-integrity FAIL: {n_barriers} barriers, "
            f"expected {expected_barriers} (L={spec.L}+2). The transpiler restructured "
            f"the circuit; run is INVALID (§2.3)."
        )

    if spec.kind == "signal" and abs(spec.theta_star) > atol:
        injected = {round(abs(a), 12) for a in spec.layer_angles if abs(a) > atol}
        n_injected_rz = 0
        for ci in transpiled_qc.data:
            if ci.operation.name != "rz":
                continue
            ang = float(ci.operation.params[0])
            # fold into (-pi, pi] then take magnitude for matching
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            if round(abs(ang), 12) in injected:
                n_injected_rz += 1
        if n_injected_rz != spec.L:
            raise AssertionError(
                f"[{spec.cid}] transpiler-integrity FAIL: found {n_injected_rz} injected "
                f"rz layers, expected {spec.L}. Injected phases were merged/removed; run "
                f"is INVALID (§2.3)."
            )
    return True


if __name__ == "__main__":
    specs = generate_all_specs()
    n_sig = sum(1 for s in specs if s.kind == "signal")
    n_vis = sum(1 for s in specs if s.kind == "viscal")
    bad = [s.cid for s in specs if not spec_integrity_ok(s)]
    print(f"generated {len(specs)} circuits: {n_sig} signal + {n_vis} viscal")
    print(f"grid: thetas={config.THETAS} depths={config.DEPTHS} "
          f"instances={config.INSTANCES_PER_POINT} shots={config.SHOTS}")
    print(f"spec-integrity failures: {len(bad)}")
    assert not bad, bad
