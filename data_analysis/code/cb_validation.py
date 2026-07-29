"""§2 Credibility anchor: reproduce the published cycle-benchmarking Pauli fidelities.

Standard cycle benchmarking (Erhard et al. 2019; Chen et al. 2023): for each target
Pauli the twirled dressed cycle is a Pauli channel, and the sign-corrected survival
of a measured Pauli O decays as
        f_O(L) = A_O * lambda_O**L,
where lambda_O is the Pauli fidelity (the SPAM-free decay base) and A_O absorbs SPAM.

We estimate f_O(L) from the sign-corrected per-shot eigenvalues (loader), pool over
all randomizations (and over the targets that measure O), fit (A, lambda), and compare
lambda to the paper's OWN stored fidelities.  Agreement proves our extraction + sign
convention + fit are correct, and yields the visibilities v = lambda_O used by §3.

Run:  python data_analysis/code/cb_validation.py
Writes: outputs/pauli_fidelities.json, figures/fig_cb_validation.{png,pdf}
"""

import json
import os
import sys

import numpy as np

import cb_loader as L
from audit import load_cache

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
FIGS = os.path.join(HERE, "..", "figures")
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments"))
try:
    import _style
    C = _style.setup()
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - style is optional
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"blue": "#0072B2", "vermil": "#D55E00", "green": "#009E73",
         "orange": "#E69F00", "grey": "#999999", "black": "#000000"}


def per_circuit_survivals(rec):
    """Sign-corrected survivals of the three measurable Paulis for one circuit.

    Returns dict {observable_body: survival} where observable_body is 'AI', 'IB', 'AB'
    with A,B the target letters, and survival in [-1, 1] (ideal +1).
    """
    A, B = rec["pauli_body"][0], rec["pauli_body"][1]
    sh = rec["shots"]
    sAI = rec["s0"] * rec["sm0"] / sh
    sIB = rec["s1"] * rec["sm1"] / sh
    sAB = rec["s0"] * rec["s1"] * rec["sm01"] / sh
    return {A + "I": sAI, "I" + B: sIB, A + B: sAB}


def collect_observable(cache, group, pauli):
    """Pool sign-corrected survivals for a 2-char Pauli across all contributing
    targets/randomizations, grouped by depth.

    Returns {L: np.array of per-circuit survivals}.
    """
    from collections import defaultdict
    byL = defaultdict(list)
    for tgt in L.CB_TARGETS:
        for rec in cache[group][tgt]["records"]:
            surv = per_circuit_survivals(rec)
            if pauli in surv:
                byL[rec["L"]].append(surv[pauli])
    return {LL: np.asarray(v) for LL, v in byL.items()}


def fit_exponential(Ls, means, ses):
    """Fit means(L) = A * lambda**L by weighted least squares.

    Grid-search lambda (A solved analytically per lambda), then parabolic refine.
    Weights = 1/se**2.  Returns (lambda, A, lambda_se).
    """
    Ls = np.asarray(Ls, float)
    means = np.asarray(means, float)
    w = 1.0 / np.clip(np.asarray(ses, float), 1e-6, None) ** 2

    def best_on_grid(grid):
        chi = []
        for lam in grid:
            x = lam ** Ls
            A = np.sum(w * means * x) / np.sum(w * x * x)
            chi.append(np.sum(w * (means - A * x) ** 2))
        return np.asarray(chi)

    grid = np.linspace(0.85, 1.0, 3001)
    chi = best_on_grid(grid)
    i = int(np.argmin(chi))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 2001)
    chif = best_on_grid(fine)
    j = int(np.argmin(chif))
    lam = fine[j]
    x = lam ** Ls
    A = np.sum(w * means * x) / np.sum(w * x * x)

    # lambda uncertainty: fit a parabola chi2 ~ chimin + a*(lam-lam_hat)^2 to the
    # coarse grid within a small window; delta chi2 = 1 -> se = 1/sqrt(a).
    win = (np.abs(grid - lam) <= 0.003)
    if win.sum() >= 5:
        t = grid[win] - lam
        p = np.polyfit(t, chi[win], 2)
        a = p[0]
        se = float(1.0 / np.sqrt(a)) if a > 0 else np.nan
    else:
        se = np.nan
    return float(lam), float(A), se


def extract_all_fidelities(cache, group="cb"):
    """Fit lambda for every non-identity 2-qubit Pauli. Returns {Pauli: dict}."""
    letters = "XYZ"
    paulis = []
    for a in "IXYZ":
        for b in "IXYZ":
            if a + b == "II":
                continue
            paulis.append(a + b)
    out = {}
    for P in paulis:
        byL = collect_observable(cache, group, P)
        if not byL:
            continue
        Ls = sorted(byL)
        means = np.array([byL[LL].mean() for LL in Ls])
        ses = np.array([byL[LL].std(ddof=1) / np.sqrt(len(byL[LL]))
                        if len(byL[LL]) > 1 else 1.0 for LL in Ls])
        lam, A, lam_se = fit_exponential(Ls, means, ses)
        out[P] = dict(lam=lam, A=A, lam_se=lam_se,
                      Ls=Ls, means=means.tolist(), ses=ses.tolist(),
                      n_per_depth=[int(len(byL[LL])) for LL in Ls])
    return out


def main():
    cache = load_cache()
    stored_all = L.load_stored_fidelities()
    stored = stored_all.get("cb", {})

    fit = extract_all_fidelities(cache, "cb")

    # comparison table
    rows = []
    for P in sorted(fit):
        s = stored.get(P, np.nan)
        rows.append((P, fit[P]["lam"], fit[P]["lam_se"], s,
                     fit[P]["lam"] - s if s == s else np.nan))
    print("\n%-4s %10s %10s %10s %10s" %
          ("O", "fit_lam", "lam_se", "paper_lam", "diff"))
    for P, lam, se, s, d in rows:
        print("%-4s %10.5f %10.2e %10.5f %+10.5f" % (P, lam, se, s, d))
    diffs = np.array([d for *_, d in rows if d == d])
    max_abs = float(np.max(np.abs(diffs)))
    rms = float(np.sqrt(np.mean(diffs ** 2)))
    print("\nmax|fit - paper| = %.2e   rms = %.2e   (over %d Paulis)"
          % (max_abs, rms, len(diffs)))

    # save
    os.makedirs(OUT, exist_ok=True)
    payload = {
        "description": "Pauli fidelities lambda_O extracted by multi-depth CB fit "
        "f_O(L)=A*lambda^L, compared to Chen et al. stored fidelities. These lambda "
        "are the visibilities v used for debiasing in the coherent estimator (§3).",
        "fit_vs_paper_max_abs_diff": max_abs,
        "fit_vs_paper_rms_diff": rms,
        "pauli_fidelities": {P: {"lambda_fit": fit[P]["lam"],
                                 "lambda_se": fit[P]["lam_se"],
                                 "lambda_paper": stored.get(P),
                                 "A_spam": fit[P]["A"]} for P in fit},
    }
    with open(os.path.join(OUT, "pauli_fidelities.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote outputs/pauli_fidelities.json")

    # ---- figure ----
    make_figure(fit, stored, cache)
    return payload


def make_figure(fit, stored, cache):
    os.makedirs(FIGS, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) extracted vs paper scatter
    Ps = [P for P in sorted(fit) if P in stored]
    x = np.array([stored[P] for P in Ps])
    y = np.array([fit[P]["lam"] for P in Ps])
    ye = np.array([fit[P]["lam_se"] for P in Ps])
    lim = [min(x.min(), y.min()) - 0.003, 1.001]
    ax1.plot(lim, lim, "-", color=C["grey"], lw=1, zorder=0)
    ax1.errorbar(x, y, yerr=ye, fmt="o", color=C["blue"], ms=6,
                 ecolor=C["grey"], capsize=2, lw=0, elinewidth=1)
    for P, xi, yi in zip(Ps, x, y):
        ax1.annotate(P, (xi, yi), fontsize=7, xytext=(3, 3),
                     textcoords="offset points", color=C["black"])
    ax1.set_xlabel(r"paper $\lambda_O$ (Chen et al. 2023)")
    ax1.set_ylabel(r"our multi-depth CB fit $\lambda_O$")
    ax1.set_title("(a) Pauli fidelities: our fit vs published")
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_aspect("equal")
    md = np.max(np.abs(y - x))
    ax1.text(0.05, 0.92, r"max$|\Delta\lambda|$ = %.1e" % md,
             transform=ax1.transAxes, fontsize=10)

    # (b) example decays for a few representative Paulis
    from collections import OrderedDict
    examples = ["ZZ", "XX", "XZ", "YY"]
    colors = [C["blue"], C["vermil"], C["green"], C["orange"]]
    for P, col in zip(examples, colors):
        byL = collect_observable(cache, "cb", P)
        Ls = np.array(sorted(byL))
        m = np.array([byL[LL].mean() for LL in Ls])
        se = np.array([byL[LL].std(ddof=1) / np.sqrt(len(byL[LL])) for LL in Ls])
        ax2.errorbar(Ls, m, yerr=se, fmt="o", color=col, ms=5, capsize=2,
                     lw=0, elinewidth=1, label=r"%s ($\lambda$=%.4f)" % (P, fit[P]["lam"]))
        xx = np.linspace(Ls.min(), Ls.max(), 200)
        ax2.plot(xx, fit[P]["A"] * fit[P]["lam"] ** xx, "-", color=col, lw=1.5)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("circuit depth $L$ (cycles)")
    ax2.set_ylabel(r"sign-corrected survival $f_O(L)$")
    ax2.set_title("(b) SPAM-free exponential decays")
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.02, 1.02)

    fig.savefig(os.path.join(FIGS, "fig_cb_validation.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "fig_cb_validation.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_cb_validation.{png,pdf}")


if __name__ == "__main__":
    main()
