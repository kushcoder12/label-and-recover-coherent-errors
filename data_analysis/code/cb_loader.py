"""Safe deserialization + compact extraction of the IBM cycle-benchmarking dataset.

Data source: Chen et al., "The learnability of Pauli noise", Nature Communications
2023 (ibmq_montreal, 2 target qubits).  Files (in ../data):
  - ibmq_experiment_all_20220323_8530634712        (circuits + metadata, NO results)
  - ibmq_experiment_all_20220323_8530634712_full    (circuits + metadata + results)
  - ibmq_experiment_all_20220323_8530634712_result   (paper's OWN fitted fidelities)

These are qiskit-pickled Python objects.  qiskit is intentionally NOT a dependency
of this repo (numpy-only, reproducible).  We therefore deserialize with a
``SafeUnpickler`` that maps every non-builtin class to an inert, permissive stub.
The stub tolerates any attribute/method access that qiskit's ``__reduce__`` logic
performs during unpickling, and records object state via ``__setstate__``.

IMPORTANT provenance note (verified empirically in the data audit, do not assume):
  * Top level dict keys: 'token', 'cb', 'int_cb', 'intc_cb'.
  * cb / int_cb hold 9 two-qubit target channels 'cb_XX'..'cb_ZZ'.
  * Per target: 'parameters' (n, shots, Lrange, C, repeat, pauli),
    'batch_0' (list of C*len(Lrange) per-circuit metadata dicts), and (in _full)
    'result' -> [Result] whose .results is a list aligned 1:1 with batch_0.
  * Per circuit metadata: L (depth), c (randomization index), pauli (target,
    possibly signed e.g. '-XY'), clifford = {stabilizer: [signed Z-Paulis],
    destabilizer: [signed X-Paulis]}.  This is the NET twirl frame (a single
    2-qubit Clifford tableau), NOT the per-layer random Paulis -- those live
    inside the qiskit QuantumCircuit, which does not survive safe-unpickling.
  * Per result: data.counts (hex outcome -> count) and data.memory (per-shot hex).
  * The QuantumCircuit object itself deserializes to an opaque stub and is dropped.
"""

import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data")
CACHE_DIR = os.path.join(HERE, "..", "cache")

STEM = "ibmq_experiment_all_20220323_8530634712"
PATH_META = os.path.join(DATA_DIR, STEM)            # circuits + metadata
PATH_FULL = os.path.join(DATA_DIR, STEM + "_full")  # + results (counts, memory)
PATH_RESULT = os.path.join(DATA_DIR, STEM + "_result")  # paper's fitted fidelities

CB_TARGETS = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]

# Builtin / numpy modules we let pickle reconstruct natively.
_NATIVE_MODULES = {
    "builtins", "collections", "collections.abc",
    "numpy", "numpy.core.multiarray", "numpy.core.numeric", "numpy._core",
    "numpy._core.multiarray", "numpy._core.numeric", "_codecs", "copyreg",
}


def _noop(*a, **k):
    return _Stub()


class _Stub:
    """Inert, permissive stand-in for any qiskit (non-builtin) class.

    Tolerates arbitrary attribute access (returns a callable that yields another
    stub) so qiskit's reduce/build logic never raises during unpickling.  Object
    state supplied via ``__setstate__`` is stored on ``_state`` for inspection.
    """

    def __init__(self, *a, **k):
        object.__setattr__(self, "_args", a)
        object.__setattr__(self, "_kw", k)

    def __setstate__(self, state):
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name):
        return _noop

    def __repr__(self):
        return "<Stub %s>" % object.__getattribute__(self, "__class__").__name__


def _make_stub(module, name):
    return type("Stub_%s" % name, (_Stub,), {"_qname": "%s.%s" % (module, name)})


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that reconstructs builtins/numpy natively and stubs everything else."""

    def find_class(self, module, name):
        if module in _NATIVE_MODULES:
            try:
                return super().find_class(module, name)
            except Exception:
                pass
        return _make_stub(module, name)


def safe_load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


def _state(obj):
    """Return the dict state of a stubbed qiskit object."""
    s = getattr(obj, "_state", None)
    if isinstance(s, dict):
        return s
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# Pauli-frame parsing
# --------------------------------------------------------------------------- #
def parse_signed_pauli(s):
    """'-XY' -> (-1, 'XY');  'IZ' -> (+1, 'IZ')."""
    sign = 1
    if s[0] in "+-":
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    return sign, s


def _generator_signs(stab, destab):
    """Return per-qubit (dx, sz): the signs of the X-type destabilizer and Z-type
    stabilizer generators.  Each generator is non-identity on exactly one qubit;
    qubit 0 is position 0 in the 2-char body (big-endian)."""
    dx = [1, 1]
    sz = [1, 1]
    for gen in destab:  # X-type
        sg, body = parse_signed_pauli(gen)
        for q, ch in enumerate(body):
            if ch != "I":
                dx[q] = sg
    for gen in stab:    # Z-type
        sg, body = parse_signed_pauli(gen)
        for q, ch in enumerate(body):
            if ch != "I":
                sz[q] = sg
    return dx, sz


def readout_signs(stab, destab, target_body):
    """Ideal per-qubit eigenvalue signs (s0, s1) of the *measured* target Paulis.

    For target Pauli 'AB', qubit q measures its letter Lq in {X, Y, Z} (the readout
    basis rotation is baked into the circuit, so outcomes are always computational Z).
    The ideal sign of that measurement is read off the net frame:
        X-letter -> destabilizer sign dx_q
        Z-letter -> stabilizer  sign sz_q
        Y-letter -> product dx_q * sz_q   (since Y = i X Z, its stabilizer sign is
                                            the product of the X and Z frame signs)
    Derived and validated at 100% against the shallowest-depth argmax outcomes for
    all 9 targets (see audit.py frame_prediction_check).
    """
    dx, sz = _generator_signs(stab, destab)
    s = []
    for q, letter in enumerate(target_body):
        if letter == "X":
            s.append(dx[q])
        elif letter == "Z":
            s.append(sz[q])
        elif letter == "Y":
            s.append(dx[q] * sz[q])
        else:  # 'I' -> not measured for a weight-2 target, but handle gracefully
            s.append(1)
    return s[0], s[1]


def net_frame_pauli(stab, destab):
    """Decode the 2-qubit net Pauli byproduct from stabilizer/destabilizer signs.

    Pauli-frame tracking: for each qubit, a Z byproduct flips the X-observable sign
    (destabilizer -), an X byproduct flips the Z-observable sign (stabilizer -):
        (X-sign, Z-sign) = (+,+)->I  (-,+)->Z  (+,-)->X  (-,-)->Y
    Returns a length-2 string over {'I','X','Y','Z'} (qubit 0 first).
    """
    dx, sz = _generator_signs(stab, destab)
    out = []
    for q in (0, 1):
        xf = dx[q] < 0   # X-observable flipped -> Z byproduct present
        zf = sz[q] < 0   # Z-observable flipped -> X byproduct present
        out.append({(False, False): "I", (True, False): "Z",
                    (False, True): "X", (True, True): "Y"}[(xf, zf)])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _counts_from_experiment(exp):
    """Return {outcome_int: count} for one ExperimentResult stub."""
    data = _state(_state(exp)["data"])
    counts = data.get("counts", {})
    return {int(k, 16): int(v) for k, v in counts.items()}


def _has_memory(exp):
    data = _state(_state(exp)["data"])
    mem = data.get("memory", None)
    return mem is not None and hasattr(mem, "__len__") and len(mem) > 0


def extract_group(root, group, target):
    """Extract one target channel (e.g. group='cb', target='XX').

    Returns a dict with per-circuit records aligned by index:
      L (int), c (int), pauli_sign (int), pauli_body (str),
      s0,s1 (readout signs), frame (net Pauli str), counts (dict int->int),
      n0,n1,n01 (sufficient statistics: sum of m0, m1, m0*m1 over shots), shots.
    plus 'Lrange', 'C', 'shots' from parameters and 'has_memory'.
    """
    key = "%s_%s" % (group, target)
    tgt = root[group][key]
    params = tgt["parameters"]
    batch = tgt["batch_0"]
    exps = _state(tgt["result"][0])["results"] if "result" in tgt else None

    recs = []
    has_mem = False
    for i, circ in enumerate(batch):
        cl = circ["clifford"]
        psign, pbody = parse_signed_pauli(circ["pauli"])
        s0, s1 = readout_signs(cl["stabilizer"], cl["destabilizer"], pbody)
        frame = net_frame_pauli(cl["stabilizer"], cl["destabilizer"])
        rec = dict(L=int(circ["L"]), c=int(circ["c"]),
                   pauli_sign=psign, pauli_body=pbody,
                   s0=int(s0), s1=int(s1), frame=frame)
        if exps is not None:
            counts = _counts_from_experiment(exps[i])
            shots = sum(counts.values())
            # m0 = (-1)^{qubit0 bit}, qubit0 is the high bit (value>>1); qubit1 = value&1
            sm0 = sm1 = sm01 = 0
            for v, ct in counts.items():
                m0 = 1 - 2 * ((v >> 1) & 1)
                m1 = 1 - 2 * (v & 1)
                sm0 += m0 * ct
                sm1 += m1 * ct
                sm01 += m0 * m1 * ct
            rec.update(counts=counts, shots=shots, sm0=sm0, sm1=sm1, sm01=sm01)
            if _has_memory(exps[i]):
                has_mem = True
        recs.append(rec)
    return dict(Lrange=list(params["Lrange"]), C=int(params["C"]),
                shots=int(params["shots"]), pauli=params["pauli"],
                has_memory=has_mem, records=recs)


def load_stored_fidelities():
    """Return the paper's own fitted Pauli fidelities dict {Pauli: lambda}."""
    r = safe_load(PATH_RESULT)
    out = {}
    for section in ("cb", "int_cb"):
        if section in r and "fidelity" in r[section]:
            out[section] = {k: float(v) for k, v in r[section]["fidelity"].items()}
    return out


if __name__ == "__main__":
    # Smoke test: load metadata-only file and print one target summary.
    root = safe_load(PATH_META)
    print("top keys:", list(root.keys()))
    g = extract_group(safe_load(PATH_FULL), "cb", "XX")
    print("cb_XX: Lrange=%s C=%d shots=%d has_memory=%s n_circ=%d"
          % (g["Lrange"], g["C"], g["shots"], g["has_memory"], len(g["records"])))
    r0 = g["records"][0]
    print("record[0]:", {k: r0[k] for k in ("L", "c", "pauli_body", "s0", "s1",
                                            "frame", "sm0", "sm1", "sm01", "shots")})
