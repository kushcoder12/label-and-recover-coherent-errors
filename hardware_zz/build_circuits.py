"""Owned-label circuit specifications for the static-ZZ hardware run.

One circuit per twirl instance; every label (qubit 1's own sign s1, and every
neighbor's state-prep label c_j) is generated and stored here, never lost. Two
topologies: "pair" (qubit 1 + one neighbor) and "star" (qubit 1 + two neighbors, no
neighbor-neighbor coupling required).

Per-shot circuit (physical qubit 0 = target, 1..Q = neighbors, via initial_layout):
    1. H on target                          -> |+>
    2. X on neighbor j iff c_j = -1          -> real state prep (Sec 6)
    3. RZ(s1 * theta0) on target             -> qubit 1's own known, injected knob
    4. delay(DELAY_S) on every qubit used    -> where the REAL static ZZ accumulates
    5. SDG, H on target                      -> rotates Y into Z
    6. measure target

Ideal (noiseless) statistics: <Y_target> = sin(s1*theta0 + sum_j kappa_1j*c_j), the
SAME closed form validated in core/circuits.py + tests/test_multiqubit_zz.py -- kappa_1j
here is the device's real coupling accumulated over DELAY_S, not injected.

Pure Python / numpy at spec-generation time; the qiskit builder and transpiler-integrity
assert import qiskit lazily.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np

import config


@dataclass
class CircuitSpec:
    cid: str
    kind: str          # "signal" or "viscal"
    topology: str       # "pair" (1 neighbor) or "star" (2 neighbors)
    theta0: float
    s1: int
    c: Dict[int, int]   # {neighbor logical index (2, 3): +-1}
    delay_s: float
    meta: dict = field(default_factory=dict)

    def to_record(self):
        return asdict(self)


def _rng(seed):
    return np.random.default_rng(seed)


def generate_signal_specs(thetas=None, seed=None):
    thetas = config.THETA0_SWEEP if thetas is None else thetas
    seed = config.SEED if seed is None else seed
    rng = _rng(seed)
    specs = []
    for topology, neighbors, instances in (
        ("pair", [2], config.INSTANCES_PAIR),
        ("star", [2, 3], config.INSTANCES_STAR),
    ):
        for theta0 in thetas:
            for i in range(instances):
                s1 = int(rng.choice([-1, 1]))
                c = {j: int(rng.choice([-1, 1])) for j in neighbors}
                cid = f"sig_{topology}_t{theta0:.2f}_i{i:03d}"
                specs.append(CircuitSpec(cid=cid, kind="signal", topology=topology,
                                         theta0=float(theta0), s1=s1, c=c,
                                         delay_s=config.DELAY_S))
    return specs


def generate_viscal_specs(repeats=None, seed=None):
    """Visibility references at net phase pi/2 (Sec on VISCAL_NET_PHASE in config.py:
    stationary point of sin, so an unknown kappa contribution biases v only at O(kappa^2)).
    Same delay as the signal circuits -> same decoherence to debias against."""
    repeats = config.VISCAL_REPEATS if repeats is None else repeats
    seed = (config.SEED + 777) if seed is None else seed
    rng = _rng(seed)
    specs = []
    for topology, neighbors in (("pair", [2]), ("star", [2, 3])):
        for r in range(repeats):
            c = {j: int(rng.choice([-1, 1])) for j in neighbors}
            cid = f"vis_{topology}_r{r:03d}"
            specs.append(CircuitSpec(cid=cid, kind="viscal", topology=topology,
                                     theta0=config.VISCAL_NET_PHASE, s1=1, c=c,
                                     delay_s=config.DELAY_S,
                                     meta={"net_phase": config.VISCAL_NET_PHASE}))
    return specs


def generate_all_specs():
    return generate_signal_specs() + generate_viscal_specs()


def spec_integrity_ok(spec):
    if spec.topology == "pair" and set(spec.c) != {2}:
        return False
    if spec.topology == "star" and set(spec.c) != {2, 3}:
        return False
    if spec.s1 not in (1, -1):
        return False
    if any(v not in (1, -1) for v in spec.c.values()):
        return False
    return True


# ---------------------------------------------------------------------------
# qiskit builder (LAZY import) -- only needed for the live hardware run
# ---------------------------------------------------------------------------
def build_qiskit_circuit(spec):
    """n-qubit circuit (n = 1 + len(spec.c)); circuit qubit 0 = target, 1..n-1 =
    neighbors in sorted(spec.c) order. initial_layout maps these to physical qubits
    at transpile time. Barriers isolate every stage so opt_level=0 cannot merge them."""
    from qiskit import QuantumCircuit

    neighbors = sorted(spec.c)
    n = 1 + len(neighbors)
    qc = QuantumCircuit(n, 1, name=spec.cid)
    qc.h(0)
    qc.barrier()
    for i, j in enumerate(neighbors):
        if spec.c[j] == -1:
            qc.x(i + 1)
    qc.barrier()
    qc.rz(spec.s1 * spec.theta0, 0)
    qc.barrier()
    if spec.delay_s > 0:
        for q in range(n):
            qc.delay(spec.delay_s, q, unit="s")
    qc.barrier()
    qc.sdg(0)
    qc.h(0)
    qc.barrier()
    qc.measure(0, 0)
    return qc


def assert_injected_layers_intact(transpiled_qc, spec, target_physical_qubit, atol=1e-9):
    """Confirm the transpiled circuit still carries: the injected RZ(s1*theta0) on the
    target, and a delay of >= spec.delay_s on the target (the #1 and #2 failure modes
    -- a merged/dropped RZ, or a delay padded away).

    target_physical_qubit: the backend qubit index initial_layout mapped circuit
    qubit 0 (the target) onto -- needed because the transpiled circuit's own register
    spans the whole backend, so "qubit 0 of transpiled_qc" is NOT the target."""
    n_barriers = sum(1 for ci in transpiled_qc.data if ci.operation.name == "barrier")
    expected_barriers = 5 if spec.delay_s > 0 else 4
    if n_barriers != expected_barriers:
        raise AssertionError(
            f"[{spec.cid}] transpiler-integrity FAIL: {n_barriers} barriers, "
            f"expected {expected_barriers}. Run is INVALID.")

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
                f"[{spec.cid}] transpiler-integrity FAIL: injected RZ({spec.s1*spec.theta0}) "
                f"not found intact. Run is INVALID.")

    if spec.delay_s > 0:
        total_delay = sum(float(ci.operation.duration) for ci in transpiled_qc.data
                          if ci.operation.name == "delay"
                          and transpiled_qc.find_bit(ci.qubits[0]).index == target_physical_qubit)
        # duration units depend on the backend's scheduling; only assert it is nonzero --
        # exact unit conversion is checked once at submission time against backend.dt.
        if total_delay <= 0:
            raise AssertionError(
                f"[{spec.cid}] transpiler-integrity FAIL: idle delay on the target qubit "
                f"was dropped. Run is INVALID.")
    return True


if __name__ == "__main__":
    specs = generate_all_specs()
    n_sig = sum(1 for s in specs if s.kind == "signal")
    n_vis = sum(1 for s in specs if s.kind == "viscal")
    bad = [s.cid for s in specs if not spec_integrity_ok(s)]
    print(f"generated {len(specs)} circuits: {n_sig} signal + {n_vis} viscal")
    print(f"grid: thetas={config.THETA0_SWEEP} delay_s={config.DELAY_S} "
          f"instances(pair,star)=({config.INSTANCES_PAIR},{config.INSTANCES_STAR}) "
          f"shots={config.SHOTS}")
    print(f"spec-integrity failures: {len(bad)}")
    assert not bad, bad
