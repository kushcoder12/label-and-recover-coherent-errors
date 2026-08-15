"""Native OpenQASM 2.0 -> gate-list loader (I/O boundary only).

Circuits enter the pipeline at this I/O boundary and everything downstream
runs in the numpy core. qiskit is not required: this is a small, self-
contained parser for the qelib1 gate set used by the bundled QASMBench circuits, so the
corpus is fully reproducible with no heavy dependency. (A qiskit path could be added
here if ever desired; it is deliberately not needed.)

Qubit mapping: QASM q[i] -> engine qubit (i+1), 1-indexed big-endian (consistent
throughout; the physics is invariant to the absolute labeling).

Only unitary gates are loaded; barrier/measure/reset/creg are ignored. Files that
define custom `gate` blocks or use `if` are not handled (curate around them).
"""

import os
import re

import numpy as np

from .single_qubit_gates import Hadamard, X_gate, Y_gate, Z_gate, Rx, Ry, Rz, U3, P_gate
from .two_qubit_gates import CX_gate, CZ, CPhase, SWAP_gate, Rzz
from .noise import _embed_two_qubit_op

_SQRT = 1 / np.sqrt(2)


def _sx():
    return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)


def _eval_expr(expr):
    """Evaluate a QASM angle expression (pi, arithmetic) safely."""
    expr = expr.strip()
    if expr == "":
        return 0.0
    allowed = {"pi": np.pi, "PI": np.pi}
    if not re.fullmatch(r"[0-9pPiI\.\+\-\*\/\(\)\seE]*", expr):
        raise ValueError(f"unsafe QASM expression: {expr!r}")
    return float(eval(expr, {"__builtins__": {}}, allowed))


# 1-qubit gate matrices
def _u2(phi, lam):
    return U3(np.pi / 2, phi, lam)


_ONEQ = {
    "h": lambda p: Hadamard(),
    "x": lambda p: X_gate(),
    "y": lambda p: Y_gate(),
    "z": lambda p: Z_gate(),
    "s": lambda p: P_gate(np.pi / 2),
    "sdg": lambda p: P_gate(-np.pi / 2),
    "t": lambda p: P_gate(np.pi / 4),
    "tdg": lambda p: P_gate(-np.pi / 4),
    "sx": lambda p: _sx(),
    "sxdg": lambda p: _sx().conj().T,
    "id": lambda p: np.eye(2, dtype=complex),
    "rx": lambda p: Rx(p[0]),
    "ry": lambda p: Ry(p[0]),
    "rz": lambda p: Rz(p[0]),
    "u1": lambda p: P_gate(p[0]),
    "p": lambda p: P_gate(p[0]),
    "u2": lambda p: _u2(p[0], p[1]),
    "u3": lambda p: U3(p[0], p[1], p[2]),
    "u": lambda p: U3(p[0], p[1], p[2]),
}


def _crz(lam):
    return np.diag([1.0, 1.0, np.exp(-1j * lam / 2), np.exp(1j * lam / 2)]).astype(complex)


def _cy():
    m = np.eye(4, dtype=complex)
    m[2, 2] = 0; m[3, 3] = 0; m[2, 3] = -1j; m[3, 2] = 1j
    return m


def _ch():
    m = np.eye(4, dtype=complex)
    H = Hadamard()
    m[2:4, 2:4] = H
    return m


_TWOQ = {
    "cx": lambda p: CX_gate(),
    "cnot": lambda p: CX_gate(),
    "cz": lambda p: CZ(),
    "cy": lambda p: _cy(),
    "ch": lambda p: _ch(),
    "swap": lambda p: SWAP_gate(),
    "cu1": lambda p: CPhase(p[0]),
    "cp": lambda p: CPhase(p[0]),
    "crz": lambda p: _crz(p[0]),
    "rzz": lambda p: Rzz(p[0]),
}


def _embed_3q(mat8, qubits, n):
    """Embed an 8x8 gate on ordered `qubits` (1-indexed) into 2^n, big-endian."""
    dim = 2**n
    bitpos = [n - q for q in qubits]           # bit index of each of the 3 qubits
    out = np.zeros((dim, dim), dtype=complex)
    for r in range(dim):
        sub_r = sum(((r >> bitpos[k]) & 1) << (2 - k) for k in range(3))
        for sub_c in range(8):
            val = mat8[sub_r, sub_c]
            if val == 0:
                continue
            c = r
            for k in range(3):
                bit = (sub_c >> (2 - k)) & 1
                c = (c & ~(1 << bitpos[k])) | (bit << bitpos[k])
            out[r, c] += val
    return out


def _ccx8():
    m = np.eye(8, dtype=complex)
    m[6, 6] = 0; m[7, 7] = 0; m[6, 7] = 1; m[7, 6] = 1     # flip target when both controls =1
    return m


def _cswap8():
    m = np.eye(8, dtype=complex)
    # control = qubit0 (msb of the 3); swap the other two when control=1 -> |101>|110> swap
    m[5, 5] = 0; m[6, 6] = 0; m[5, 6] = 1; m[6, 5] = 1
    return m


def parse_qasm(path):
    """Parse a QASM 2.0 file -> (n_qubits, gatelist).

    gatelist entries: (name, params_list, qubit_indices_1based).
    """
    text = open(path).read()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)      # block comments
    n = None
    qreg_name = None
    gates = []
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        for stmt in line.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            head = stmt.split()[0]
            if head in ("OPENQASM", "include", "creg", "barrier", "measure", "reset"):
                continue
            if head == "qreg":
                m = re.match(r"qreg\s+(\w+)\s*\[(\d+)\]", stmt)
                qreg_name, n = m.group(1), int(m.group(2))
                continue
            if head in ("gate", "if", "opaque"):
                raise ValueError(f"unsupported construct in {os.path.basename(path)}: {head}")
            # a gate application:  name(params) q[a], q[b], ...
            m = re.match(r"([a-zA-Z][\w]*)\s*(\(([^)]*)\))?\s*(.+)", stmt)
            if not m:
                continue
            name = m.group(1)
            params = [_eval_expr(x) for x in m.group(3).split(",")] if m.group(3) else []
            qargs = re.findall(r"\w+\s*\[\s*(\d+)\s*\]", m.group(4))
            qubits = tuple(int(q) + 1 for q in qargs)         # q[i] -> engine qubit i+1
            gates.append((name, params, qubits))
    if n is None:
        raise ValueError(f"no qreg found in {path}")
    return n, gates


def circuit_to_ops(n, gatelist):
    """Convert a gatelist to a list of dense 2^n unitaries (small n only)."""
    ops = []
    for name, params, qubits in gatelist:
        if name in _ONEQ:
            ops.append(_embed_1q(_ONEQ[name](params), qubits[0], n))
        elif name in _TWOQ:
            ops.append(_embed_two_qubit_op(_TWOQ[name](params), qubits[0], qubits[1], n))
        elif name in ("ccx", "toffoli"):
            ops.append(_embed_3q(_ccx8(), qubits, n))
        elif name in ("cswap", "fredkin"):
            ops.append(_embed_3q(_cswap8(), qubits, n))
        else:
            raise ValueError(f"unsupported gate: {name}")
    return ops


def _embed_1q(op2, q, n):
    ops = [np.eye(2, dtype=complex)] * n
    ops[q - 1] = op2
    M = ops[0]
    for o in ops[1:]:
        M = np.kron(M, o)
    return M


def load_circuit(path):
    """Convenience: parse a QASM file and return (n, ideal_ops)."""
    n, gl = parse_qasm(path)
    return n, circuit_to_ops(n, gl), gl
