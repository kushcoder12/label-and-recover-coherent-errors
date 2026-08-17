"""Recover theta0 and every kappa_1j from owned per-circuit labels.

Reuses core.multiqubit_zz.joint_correlation_estimator (the same exact/leading-order
inversion validated by Oracles MQ1-MQ3) rather than re-deriving the formulas here.
Per circuit instance i, <Y>_i = (n0-n1)/shots stands in for the per-shot outcome mean;
since the estimator's point estimate is itself just a mean over records, feeding it
one (s1, c, <Y>_i) row per INSTANCE (rather than per raw shot) gives the identical
point estimate, and bootstrapping over instances matches the single-qubit hardware
run's convention (hardware/analyze.py).

theta0 is a genuine positive control (we chose it); kappa_1j is the device's real
coupling and has no ground truth here -- report it with its CI, not against a
"truth" line (Sec 6 honesty requirement).
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.multiqubit_zz import joint_correlation_estimator

import config

C_TARGET = "#0072B2"
C_MARGINAL = "#D55E00"
C_TRUTH = "#000000"
C_STAR = "#CC79A7"


def expval_Y(counts):
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    tot = n0 + n1
    return (n0 - n1) / tot if tot else 0.0


def _group(records):
    out = {"pair": {"signal": {}, "viscal": []}, "star": {"signal": {}, "viscal": []}}
    for r in records:
        y = expval_Y(r["counts"])
        topo = r["topology"]
        c = {int(k): v for k, v in r["c"].items()}
        if r["kind"] == "viscal":
            out[topo]["viscal"].append(y)
        else:
            key = round(float(r["theta0"]), 6)
            out[topo]["signal"].setdefault(key, []).append((r["s1"], c, y))
    return out


def visibility(viscal_ys):
    if not viscal_ys:
        return 1.0, 0.0
    ys = np.asarray(viscal_ys, float)
    v = float(np.clip(ys.mean(), 1e-6, 1.0))
    sem = float(ys.std(ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0
    return v, sem


NEIGHBOR_QUBITS = {"pair": [2], "star": [2, 3]}


def analyze(records, B=1000, seed=None):
    seed = config.SEED if seed is None else seed
    grouped = _group(records)
    results = {}
    for topology in ("pair", "star"):
        v, v_sem = visibility(grouped[topology]["viscal"])
        neighbor_qubits = NEIGHBOR_QUBITS[topology]
        by_theta = {}
        for theta0, rows in sorted(grouped[topology]["signal"].items()):
            recs = [(s1, c, y_i / v) for (s1, c, y_i) in rows]
            fit = joint_correlation_estimator(recs, 1, neighbor_qubits, B=B,
                                              seed=seed + int(round(theta0 * 1000)))
            by_theta[f"theta{theta0:.3f}"] = {
                "theta0_star": theta0, "v": v, "v_sem": v_sem,
                "n_instances": len(rows), **fit,
            }
        results[topology] = by_theta
    return {
        "results": results,
        "grid": {"thetas": config.THETA0_SWEEP, "delay_s": config.DELAY_S,
                 "instances": {"pair": config.INSTANCES_PAIR, "star": config.INSTANCES_STAR},
                 "shots": config.SHOTS},
        "backend": records[0].get("backend", "unknown") if records else "unknown",
    }


# ---------------------------------------------------------------------------
# Figure 15
# ---------------------------------------------------------------------------
def _backend_caption(analysis, records):
    b = analysis.get("backend", "unknown")
    qubits = records[0].get("qubits", "?") if records else "?"
    shots = analysis["grid"]["shots"]
    ts = records[0].get("timestamp") if records else None
    date = ts.split("T")[0] if isinstance(ts, str) and "T" in ts else (ts or "offline")
    return f"backend={b}, qubits={qubits}, date={date}, shots={shots}/circuit, delay={analysis['grid']['delay_s']*1e6:.2f} us"


def figure_15(analysis, records, path=None):
    res = analysis["results"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.8))

    # (a) theta0 positive control -- recovered vs injected, both topologies
    lo = min(config.THETA0_SWEEP)
    hi = max(config.THETA0_SWEEP)
    axA.plot([lo, hi], [lo, hi], "--", color=C_TRUTH, lw=1.2, label="ideal recovery")
    for topology, color, mk in [("pair", C_TARGET, "o"), ("star", C_STAR, "s")]:
        xs, ys, elo, ehi = [], [], [], []
        for key, r in sorted(res[topology].items(), key=lambda kv: kv[1]["theta0_star"]):
            xs.append(r["theta0_star"]); ys.append(r["theta0_hat"])
            lo_ci, hi_ci = r["bootstrap_ci"]["theta0"]
            elo.append(r["theta0_hat"] - lo_ci); ehi.append(hi_ci - r["theta0_hat"])
        axA.errorbar(xs, ys, yerr=[elo, ehi], fmt=mk, color=color, ms=6, mec="k", mew=0.4,
                     capsize=3, lw=1.2, label=f"{topology} topology")
    axA.set_xlabel(r"injected  $\theta_1^{(0)}$  (rad)")
    axA.set_ylabel(r"recovered  $\hat\theta_1^{(0)}$  (rad)")
    axA.set_title("(a) Positive control: known injected phase")
    axA.legend(fontsize=9, loc="upper left")

    # (b) recovered kappa_1j -- device characterization, no ground truth to compare
    labels, vals, elo, ehi = [], [], [], []
    for topology in ("pair", "star"):
        for j in NEIGHBOR_QUBITS[topology]:
            # average the recovered kappa across the (mostly-redundant) theta0 grid,
            # weighting by inverse CI width, to report one number per neighbor
            ks, ws = [], []
            for r in res[topology].values():
                kh = r["kappa_hat"][j]
                lo_ci, hi_ci = r["bootstrap_ci"]["kappa"][j]
                w = 1.0 / max((hi_ci - lo_ci), 1e-6) ** 2
                ks.append(kh); ws.append(w)
            ks, ws = np.array(ks), np.array(ws)
            kbar = float(np.sum(ks * ws) / np.sum(ws))
            spread = float(np.sqrt(1.0 / np.sum(ws)))
            labels.append(f"{topology}\nneighbor {j}")
            vals.append(kbar); elo.append(spread); ehi.append(spread)
    x = np.arange(len(labels))
    axB.errorbar(x, vals, yerr=[elo, ehi], fmt="D", color=C_MARGINAL, ms=8, mec="k",
                 mew=0.5, capsize=4, lw=1.4)
    axB.axhline(0.0, color="grey", lw=0.8, alpha=0.6)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=9)
    axB.set_ylabel(r"recovered  $\hat\kappa_{1j}$  (rad, over delay=%.2f us)" % (config.DELAY_S * 1e6))
    axB.set_title("(b) Real static ZZ coupling (characterization, no ground truth)")

    fig.suptitle("Fig 15 — Static ZZ crosstalk recovered on real hardware", fontsize=13)
    fig.text(0.5, -0.02, _backend_caption(analysis, records), ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, path or os.path.join(config.FIG_DIR, "fig15_hardware_zz"))
    plt.close(fig)


def _save(fig, stem):
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")


def main(raw_path=None):
    raw_path = raw_path or config.RAW_RESULTS_PATH
    if not os.path.exists(raw_path):
        raise SystemExit(f"no raw results at {raw_path}. Run submit->retrieve (or run_all.py).")
    with open(raw_path) as fh:
        records = json.load(fh)
    analysis = analyze(records)
    with open(config.ANALYSIS_PATH, "w") as fh:
        json.dump(analysis, fh, indent=2)
    figure_15(analysis, records)

    print(f"backend: {analysis['backend']}")
    for topology in ("pair", "star"):
        print(f"\n[{topology}] recovered theta0_hat vs injected (95% CI):")
        for key, r in sorted(analysis["results"][topology].items(), key=lambda kv: kv[1]["theta0_star"]):
            lo, hi = r["bootstrap_ci"]["theta0"]
            flag = ""
            if abs(r["theta0_star"]) < 1e-9 and abs(r["theta0_hat"]) > config.NULL_TOL:
                flag = "  <-- NULL CONTROL FAIL"
            print(f"  theta0*={r['theta0_star']:.3f}: theta0_hat={r['theta0_hat']:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]  v={r['v']:.3f}{flag}")
            for j, kh in r["kappa_hat"].items():
                klo, khi = r["bootstrap_ci"]["kappa"][j]
                print(f"      kappa_1{j}_hat={kh:+.4f} [{klo:+.4f},{khi:+.4f}]")
    print(f"\nfigures -> {config.FIG_DIR}")
    print(f"analysis -> {config.ANALYSIS_PATH}")
    return analysis


if __name__ == "__main__":
    if "--live" in sys.argv:
        config.use_live_paths()
    main()
