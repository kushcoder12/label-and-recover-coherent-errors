"""§1 Data audit  +  builds the compact cache used by all downstream analysis.

Run:  python data_analysis/code/audit.py
Writes: data_analysis/cache/extracted.pkl   (compact, safe, our own data)
        data_analysis/outputs/data_audit.json

The audit answers, from the data itself (nothing assumed):
  * the full Lrange present per target (SPAM removal needs multi-depth);
  * number of twirl instances (randomizations) per (target, depth);
  * exactly what the per-circuit "label" is (net twirl frame vs per-layer Paulis);
  * whether shot-by-shot memory is present;
  * an empirical check that the recovered ideal-outcome frame predicts the data.
"""

import json
import os
import pickle
from collections import Counter

import numpy as np

import cb_loader as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
CACHE = os.path.join(HERE, "..", "cache")


def build_cache():
    """Parse the 261 MB _full file once; extract cb + int_cb; pickle a compact dict."""
    print("Loading _full (261 MB, qiskit-pickled) with SafeUnpickler ...")
    full = L.safe_load(L.PATH_FULL)
    cache = {"cb": {}, "int_cb": {}}
    for group in ("cb", "int_cb"):
        for tgt in L.CB_TARGETS:
            cache[group][tgt] = L.extract_group(full, group, tgt)
        print("  extracted %s (%d targets)" % (group, len(cache[group])))
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, "extracted.pkl"), "wb") as f:
        pickle.dump(cache, f)
    print("cache written -> cache/extracted.pkl")
    return cache


def load_cache():
    p = os.path.join(CACHE, "extracted.pkl")
    if not os.path.exists(p):
        return build_cache()
    with open(p, "rb") as f:
        return pickle.load(f)


def _ideal_bits_from_frame(rec):
    """Predicted ideal outcome integer from the destabilizer readout signs.

    qubit q bit = 1 iff readout sign is -1.  qubit0 is the high bit.
    """
    b0 = 1 if rec["s0"] < 0 else 0
    b1 = 1 if rec["s1"] < 0 else 0
    return (b0 << 1) | b1


def audit():
    cache = load_cache()
    report = {}

    # --- global facts ---
    report["dataset"] = {
        "source": "Chen et al., Nat. Commun. 2023 (ibmq_montreal, 2 target qubits)",
        "file_full": os.path.basename(L.PATH_FULL),
        "top_level_keys": ["token", "cb", "int_cb", "intc_cb"],
        "cb_targets": L.CB_TARGETS,
    }

    # --- per-group audit ---
    for group in ("cb", "int_cb"):
        ginfo = {}
        for tgt in L.CB_TARGETS:
            g = cache[group][tgt]
            recs = g["records"]
            per_L = Counter(r["L"] for r in recs)
            ginfo[tgt] = {
                "Lrange": g["Lrange"],
                "C_randomizations": g["C"],
                "shots_per_circuit": g["shots"],
                "n_circuits": len(recs),
                "circuits_per_depth": {str(k): per_L[k] for k in sorted(per_L)},
                "has_shot_memory": bool(g["has_memory"]),
                "target_pauli": g["pauli"],
            }
        report[group] = ginfo

    # --- what is the label? ---
    report["label_definition"] = {
        "what_clifford_encodes": "NET twirl frame (single 2-qubit Clifford tableau: "
        "2 stabilizer generators [Z-type, signed] + 2 destabilizer generators "
        "[X-type, signed]). NOT the per-layer random Paulis.",
        "per_layer_paulis_recoverable": False,
        "reason": "The per-layer random Paulis live inside the qiskit QuantumCircuit "
        "object, which deserializes to an opaque stub under numpy-only SafeUnpickler "
        "(qiskit not installed). Only the net frame survives.",
        "label_used": "readout signs (s0, s1) from the destabilizer generators, and "
        "the net Pauli byproduct frame decoded from stabilizer+destabilizer signs.",
    }

    # --- empirical validation: does the recovered ideal frame predict the data? ---
    # At the shallowest depth (highest fidelity) the argmax outcome should equal the
    # frame-predicted ideal bitstring for (almost) every circuit.
    val = {}
    for tgt in L.CB_TARGETS:
        g = cache["cb"][tgt]
        Lmin = min(g["Lrange"])
        match = tot = 0
        for r in g["records"]:
            if r["L"] != Lmin:
                continue
            pred = _ideal_bits_from_frame(r)
            argmax = max(r["counts"], key=r["counts"].get)
            match += int(pred == argmax)
            tot += 1
        val[tgt] = {"depth": Lmin, "n": tot,
                    "frac_argmax_matches_frame": round(match / tot, 4)}
    report["frame_prediction_check"] = {
        "description": "Fraction of shallowest-depth circuits whose most-probable "
        "outcome equals the destabilizer-frame-predicted ideal bitstring. ~1 confirms "
        "the label extraction and big-endian bit mapping are correct.",
        "per_target": val,
        "mean_frac": round(float(np.mean([v["frac_argmax_matches_frame"]
                                          for v in val.values()])), 4),
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "data_audit.json"), "w") as f:
        json.dump(report, f, indent=2)

    # --- console summary ---
    print("\n===== DATA AUDIT =====")
    print("Lrange (cb_XX):", cache["cb"]["XX"]["Lrange"])
    print("randomizations C per depth:", cache["cb"]["XX"]["C"],
          "| shots/circuit:", cache["cb"]["XX"]["shots"])
    print("cb targets:", L.CB_TARGETS)
    print("shot-by-shot memory present:",
          all(cache["cb"][t]["has_memory"] for t in L.CB_TARGETS))
    print("label = NET twirl frame (per-layer Paulis NOT recoverable)")
    print("frame-prediction check (mean frac argmax==ideal):",
          report["frame_prediction_check"]["mean_frac"])
    for t in L.CB_TARGETS:
        print("   cb_%s: %.3f" % (t, val[t]["frac_argmax_matches_frame"]))
    print("wrote outputs/data_audit.json")
    return report


if __name__ == "__main__":
    audit()
