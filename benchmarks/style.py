"""One visual grammar for every exhibit in the paper.

Centralized so that a change to type size or color happens once. The palette is
chosen to survive grayscale printing and the common colour-vision deficiencies:
the four designs differ in lightness as well as hue, and each series is labeled
directly rather than through a legend.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

# Okabe-Ito, which is colour-vision safe and separates in grayscale.
DESIGN_STYLE = {
    "srs": {"color": "#767676", "marker": "o", "label": "Uniform"},
    "core_plus_tail": {
        "color": "#E69F00",
        "marker": "s",
        "label": "Core + uniform tail",
    },
    "pps": {"color": "#0072B2", "marker": "^", "label": "Clipped PPS"},
    "pps_oracle_norms": {
        "color": "#009E73",
        "marker": "D",
        "label": "Clipped PPS, $a_j^2$ known",
    },
}
DESIGN_ORDER = ["srs", "core_plus_tail", "pps", "pps_oracle_norms"]

RC = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
}


def use_style() -> None:
    """Apply the shared style to matplotlib."""
    mpl.rcParams.update(RC)
