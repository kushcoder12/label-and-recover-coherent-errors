"""PennyLane layer for label-and-recover coherent errors (arXiv:2607.26756).

The numpy engine in the repository's ``core/`` package is the reference
implementation. This package re-implements the method with PennyLane
transforms and automatic differentiation, and its tests cross-check every
result against ``core/``.
"""

from .frames import PAULI_LABELS, conjugate_pauli_pair, twirl_frames
from .transforms import pauli_twirl, sign_label_twirl, sample_signs
from .circuits import signal_tape_qnode, visibility_qnode, joint_label_qnode
from .estimator import (
    fit_theta,
    arcsin_estimate,
    label_correlation,
    marginal_mean,
    bootstrap_theta,
)
from .fisher import fisher_decomposition, exact_marginal_fisher
from .hardware_data import load_records, recover_from_records
from .depth import trotter_qnode, labeled_qfi, coherent_qfi
from .crosstalk import zz_probe_qnode, zz_visibility, sample_zz_records, recover_zz

__all__ = [
    "PAULI_LABELS",
    "conjugate_pauli_pair",
    "twirl_frames",
    "pauli_twirl",
    "sign_label_twirl",
    "sample_signs",
    "signal_tape_qnode",
    "visibility_qnode",
    "joint_label_qnode",
    "fit_theta",
    "arcsin_estimate",
    "label_correlation",
    "marginal_mean",
    "bootstrap_theta",
    "fisher_decomposition",
    "exact_marginal_fisher",
    "load_records",
    "recover_from_records",
    "trotter_qnode",
    "labeled_qfi",
    "coherent_qfi",
    "zz_probe_qnode",
    "zz_visibility",
    "sample_zz_records",
    "recover_zz",
]
