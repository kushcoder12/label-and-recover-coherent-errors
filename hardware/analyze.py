"""hardware/analyze.py — recover theta* from OWNED labels (IBM_HARDWARE_SPEC.md §4-§5).

No new physics: reuse the label-correlation estimator.
  - per circuit:   <Y> = (n0 - n1)/shots.
  - visibility:    v(L) = mean over the per-depth viscal circuits of <Y>  (ideal <Y>=1).
  - label corr.:   C(L) = mean_instances[ (sum_signs) * <Y> ]   (grows with depth: our signal).
  - marginal:      mean_instances[ <Y> ] ~ 0   (labels averaged away -> RC's blind spot).
  - recovery:      theta_hat = argmin_theta sum_i ( <Y>_i - v(L) sin(theta * sum_signs_i) )^2.
                   (Reduces to arcsin(C/v) at L=1; a proper 1-parameter fit at all L.)
  - CIs:           bootstrap over twirl instances (95%).
Positive control: theta_hat vs theta* on the diagonal; theta*=0 must recover ~0 (§6 guardrail).

Figures:
  Fig R1 — recovered vs injected theta* (diagonal = success), marginal flat at ~0.
  Fig R2 — label correlation C(L) vs depth (labeled rising, marginal flat, v(L) ceiling).
Every amplitude carries a bootstrap CI; nothing is plotted without it (§6).
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# Colorblind-safe (Okabe-Ito)
C_LABELED = "#0072B2"   # blue
C_MARGINAL = "#D55E00"  # vermillion
C_TRUTH = "#000000"
C_VIS = "#009E73"       # green


# ---------------------------------------------------------------------------
# Per-circuit reductions
# ---------------------------------------------------------------------------
def expval_Y(counts):
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    tot = n0 + n1
    return (n0 - n1) / tot if tot else 0.0


def _group(records):
    """Split raw records into signal[(theta,L)] -> list and viscal[L] -> list of <Y>."""
    signal = {}
    viscal = {}
    for r in records:
        y = expval_Y(r["counts"])
        if r["kind"] == "viscal":
            viscal.setdefault(r["L"], []).append(y)
        elif r["kind"] == "signal":
            key = (round(float(r["theta_star"]), 6), int(r["L"]))
            signal.setdefault(key, []).append((int(r["sum_signs"]), y))
    return signal, viscal


def visibility_by_depth(viscal):
    """v(L) and its bootstrap-free SEM from the per-depth viscal circuits (ideal <Y>=1)."""
    out = {}
    for L, ys in viscal.items():
        ys = np.asarray(ys, float)
        v = float(np.clip(ys.mean(), 1e-6, 1.0))
        sem = float(ys.std(ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0
        out[L] = (v, sem)
    return out


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def _fit_theta(m, y, v, grid=None):
    """LS fit of theta in y_i = v sin(theta * m_i). Global grid scan + parabolic refine."""
    m = np.asarray(m, float)
    y = np.asarray(y, float)
    if grid is None:
        grid = np.linspace(-0.6, 0.6, 4001)   # covers the theta* grid (<=0.3) with margin
    # residual sum of squares for each candidate theta (vectorized)
    pred = v * np.sin(np.outer(grid, m))       # (G, N)
    rss = np.sum((pred - y[None, :]) ** 2, axis=1)
    k = int(np.argmin(rss))
    th = grid[k]
    # parabolic refinement around the grid minimum
    if 0 < k < len(grid) - 1:
        x0, x1, x2 = grid[k - 1], grid[k], grid[k + 1]
        y0, y1, y2 = rss[k - 1], rss[k], rss[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-30:
            th = x1 + 0.5 * (y0 - y2) / denom * (x1 - x0)
    return float(th)


def label_correlation(pairs):
    """C = mean_i[ sum_signs_i * <Y>_i ]  (the depth-growing label-outcome correlation)."""
    m = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float)
    return float(np.mean(m * y))


def ideal_correlation(L, theta, v):
    """Exact expected C(L) = v * E_signs[ (Sum s) sin(theta * Sum s) ] under the sin model.

    Sum s = 2k - L with k ~ Binomial(L, 1/2). Grows ~ v*theta*L at small theta (linear
    response) and bends below it once |Sum s|*theta leaves the small-angle regime -- this
    is the honest reference the measured C(L) should track (not the linear guide)."""
    from math import comb, sin
    tot = 0.0
    for k in range(L + 1):
        m = 2 * k - L
        w = comb(L, k) / (2.0 ** L)
        tot += w * m * sin(theta * m)
    return v * tot


def marginal_mean(pairs):
    """mean_i[ <Y>_i ] — the standard-RC marginal; ~0 once labels are averaged away."""
    y = np.array([p[1] for p in pairs], float)
    return float(np.mean(y))


def _bootstrap(pairs, v, B=1000, seed=0):
    """Bootstrap over twirl instances -> (theta_hat, C, marginal) with 95% CIs."""
    rng = np.random.default_rng(seed)
    m = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float)
    n = len(pairs)
    th = _fit_theta(m, y, v)
    C = float(np.mean(m * y))
    mg = float(np.mean(y))
    th_bs, C_bs, mg_bs = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        mb, yb = m[idx], y[idx]
        th_bs[b] = _fit_theta(mb, yb, v)
        C_bs[b] = np.mean(mb * yb)
        mg_bs[b] = np.mean(yb)
    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {
        "theta_hat": th, "theta_ci": ci(th_bs),
        "C": C, "C_ci": ci(C_bs),
        "marginal": mg, "marginal_ci": ci(mg_bs),
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------
def analyze(records, B=1000, seed=None):
    seed = config.SEED if seed is None else seed
    signal, viscal = _group(records)
    vis = visibility_by_depth(viscal)
    # if no viscal present (e.g. minimal test data) assume ideal visibility
    default_v = 1.0

    results = {}
    for (theta, L), pairs in sorted(signal.items()):
        v = vis.get(L, (default_v, 0.0))[0]
        res = _bootstrap(pairs, v, B=B, seed=seed + L + int(round(theta * 1000)))
        res["v"] = v
        res["n_instances"] = len(pairs)
        results[f"theta{theta:.3f}_L{L}"] = {
            "theta_star": theta, "L": L, **res,
        }
    return {
        "results": results,
        "visibility": {str(L): {"v": v, "sem": s} for L, (v, s) in sorted(vis.items())},
        "grid": {"thetas": config.THETAS, "depths": config.DEPTHS,
                 "instances": config.INSTANCES_PER_POINT, "shots": config.SHOTS},
        "backend": records[0].get("backend", "unknown") if records else "unknown",
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _backend_caption(analysis, records):
    b = analysis.get("backend", "unknown")
    qubit = records[0].get("qubit", "?") if records else "?"
    shots = analysis["grid"]["shots"]
    inst = analysis["grid"]["instances"]
    ts = records[0].get("timestamp") if records else None
    date = (ts.split("T")[0] if isinstance(ts, str) and "T" in ts else (ts or "offline"))
    return (f"backend={b}, qubit={qubit}, date={date}, shots={shots}/circuit, "
            f"{inst} twirl instances/point")


def figure_R1(analysis, records, depths_to_show=None, path=None):
    """Fig R1 — recovered theta_hat vs injected theta* (the money figure)."""
    if depths_to_show is None:
        # a shallow and a deep depth to show recovery holds across depth
        avail = sorted({v["L"] for v in analysis["results"].values()})
        depths_to_show = [d for d in (4, 16) if d in avail] or avail[:2] or avail
    res = analysis["results"]
    thetas = sorted({v["theta_star"] for v in res.values()})

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    lo, hi = min(thetas), max(thetas)
    ax.plot([lo, hi], [lo, hi], "--", color=C_TRUTH, lw=1.3, zorder=1,
            label="ideal recovery (diagonal)")

    markers = ["o", "s", "^", "D", "v"]
    palette = [C_LABELED, "#56B4E9", "#CC79A7", "#E69F00"]
    for j, L in enumerate(depths_to_show):
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for th in thetas:
            key = f"theta{th:.3f}_L{L}"
            if key not in res:
                continue
            r = res[key]
            xs.append(th)
            ys.append(r["theta_hat"])
            yerr_lo.append(r["theta_hat"] - r["theta_ci"][0])
            yerr_hi.append(r["theta_ci"][1] - r["theta_hat"])
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt=markers[j % len(markers)],
                    color=palette[j % len(palette)], capsize=3, ms=6, lw=1.3,
                    label=f"labeled-RC recovery, L={L}")

    # marginal (label-blind) estimate: flat at ~0 for contrast — shown at the deepest depth
    Lm = depths_to_show[-1]
    mxs, mys, mlo, mhi = [], [], [], []
    for th in thetas:
        key = f"theta{th:.3f}_L{Lm}"
        if key in res:
            r = res[key]
            mxs.append(th); mys.append(r["marginal"])
            mlo.append(r["marginal"] - r["marginal_ci"][0])
            mhi.append(r["marginal_ci"][1] - r["marginal"])
    ax.errorbar(mxs, mys, yerr=[mlo, mhi], fmt="x", color=C_MARGINAL, capsize=3,
                ms=7, lw=1.3, label=f"marginal (labels discarded), L={Lm}")

    ax.set_xlabel(r"injected coherent phase per cycle  $\theta^*$  (rad)")
    ax.set_ylabel(r"recovered  $\hat{\theta}$  (rad)")
    ax.set_title("Fig R1 — Label recovery of coherent phase on hardware")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.text(0.5, -0.02, _backend_caption(analysis, records), ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, path or os.path.join(config.FIG_DIR, "figR1_recovered_vs_injected"))
    plt.close(fig)


def figure_R2(analysis, records, theta_pick=None, path=None):
    """Fig R2 — label correlation C(L) grows with depth; marginal flat; v(L) ceiling."""
    res = analysis["results"]
    thetas = sorted({v["theta_star"] for v in res.values()})
    nonzero = [t for t in thetas if t > 0]
    if theta_pick is None:
        theta_pick = 0.2 if 0.2 in nonzero else (nonzero[len(nonzero) // 2] if nonzero else thetas[-1])
    depths = sorted({v["L"] for v in res.values() if v["theta_star"] == theta_pick})

    C, Clo, Chi, Mg, Mlo, Mhi = [], [], [], [], [], []
    for L in depths:
        r = res[f"theta{theta_pick:.3f}_L{L}"]
        C.append(r["C"]); Clo.append(r["C"] - r["C_ci"][0]); Chi.append(r["C_ci"][1] - r["C"])
        Mg.append(r["marginal"]); Mlo.append(r["marginal"] - r["marginal_ci"][0])
        Mhi.append(r["marginal_ci"][1] - r["marginal"])

    vis = analysis.get("visibility", {})
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.errorbar(depths, C, yerr=[Clo, Chi], fmt="o-", color=C_LABELED, capsize=3, ms=6,
                lw=1.6, label=r"labeled  $C(L)=\langle(\Sigma s)\,\langle Y\rangle\rangle$")
    # exact expected C(L) under the sin model (tracks the data; bends below the linear v*theta*L)
    guide = [ideal_correlation(L, theta_pick, vis.get(str(L), {"v": 1.0})["v"]) for L in depths]
    ax.plot(depths, guide, ":", color=C_VIS, lw=1.6,
            label=r"ideal model  $v(L)\,\mathbb{E}[(\Sigma s)\sin(\theta^*\Sigma s)]$")
    ax.errorbar(depths, Mg, yerr=[Mlo, Mhi], fmt="s--", color=C_MARGINAL, capsize=3, ms=5,
                lw=1.3, label="marginal (labels discarded)")
    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.6)

    ax.set_xlabel("depth  $L$  (Trotter cycles)")
    ax.set_ylabel(r"label-outcome correlation  $C(L)$")
    ax.set_title(rf"Fig R2 — Label correlation grows with depth  ($\theta^*={theta_pick}$ rad)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.text(0.5, -0.02, _backend_caption(analysis, records), ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, path or os.path.join(config.FIG_DIR, "figR2_correlation_vs_depth"))
    plt.close(fig)


def _save(fig, stem):
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")


def make_all_figures(analysis, records):
    figure_R1(analysis, records)
    figure_R2(analysis, records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(raw_path=None):
    raw_path = raw_path or config.RAW_RESULTS_PATH
    if not os.path.exists(raw_path):
        raise SystemExit(f"no raw results at {raw_path}. Run submit->retrieve (or run_all.py).")
    with open(raw_path) as fh:
        records = json.load(fh)
    analysis = analyze(records)
    with open(config.ANALYSIS_PATH, "w") as fh:
        json.dump(analysis, fh, indent=2)
    make_all_figures(analysis, records)

    # console summary + positive-control checks (§6)
    print(f"backend: {analysis['backend']}")
    print("visibility v(L):", {L: round(d["v"], 4) for L, d in analysis["visibility"].items()})
    print("\n recovered theta_hat vs injected theta* (95% CI):")
    for key, r in analysis["results"].items():
        lo, hi = r["theta_ci"]
        flag = ""
        if abs(r["theta_star"]) < 1e-9 and abs(r["theta_hat"]) > config.NULL_TOL:
            flag = "  <-- NULL CONTROL FAIL (|theta_hat| > tol)"
        print(f"  theta*={r['theta_star']:.3f} L={r['L']:>2}: "
              f"theta_hat={r['theta_hat']:+.4f} [{lo:+.4f}, {hi:+.4f}]  v={r['v']:.3f}{flag}")
    print(f"\nfigures -> {config.FIG_DIR}")
    print(f"analysis -> {config.ANALYSIS_PATH}")
    return analysis


if __name__ == "__main__":
    import sys
    if "--live" in sys.argv:
        config.use_live_paths()   # analyze online_data/ -> online_figures/
    main()
