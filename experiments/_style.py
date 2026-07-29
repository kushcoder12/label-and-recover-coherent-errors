"""Shared publication-quality figure style (CLAUDE.md §6).

PRX/PRL conventions: vector PDF + 300-dpi PNG, font >= 11, labelled axes with
units, legend, no chartjunk, colorblind-safe palette (Okabe-Ito).

Import `setup()` once per script, use the `C` palette dict for colors, and call
`save(fig, "figN_name")` to write both figures/figN_name.pdf and .png.
"""

import os

import matplotlib

matplotlib.use("Agg")            # headless, deterministic
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colorblind-safe palette (8 colors), keyed by role.
C = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#999999",
}

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")


def setup():
    """Apply global rcParams. Returns the palette dict C for convenience."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.constrained_layout.use": True,
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,          # editable text in vector PDF
        "ps.fonttype": 42,
    })
    return C


def save(fig, name):
    """Write fig to figures/<name>.pdf (vector) and figures/<name>.png (300 dpi)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, name + ".pdf")
    png = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")
