"""Estimators for the coherent-error angle from twirled shots.

Single-parameter E/R model (Sec 2.2): per shot a sign label s in {+1,-1} and a
binary outcome b in {+1,-1} with

    P(b | s, theta) = (1/2) (1 + b s sin theta).          (E2)

With the labels kept, theta is estimable at the quantum limit F_0 = 1 per shot;
with the labels forgotten (full twirl) it is unidentifiable.
"""

import numpy as np


def _loglik(theta, y):
    """Log-likelihood sum_i log(1 + y_i sin theta), y_i = b_i s_i in {+1,-1}."""
    return np.sum(np.log1p(y * np.sin(theta)))


def labeled_mle(shots):
    """MLE of theta from labeled shots.

    shots: iterable of (s, b) with s, b in {+1,-1}. The likelihood depends only on
    y = b s, and the score equation gives the closed form

        sin(theta_hat) = (n_+ - n_-) / N,   n_pm = #{y = +/-1},

    i.e. theta_hat = arcsin(mean(y)). Returns theta_hat in (-pi/2, pi/2).
    """
    shots = np.asarray(shots, dtype=float)
    if shots.size == 0:
        return 0.0
    y = shots[:, 0] * shots[:, 1]
    m = np.clip(np.mean(y), -1.0, 1.0)
    return float(np.arcsin(m))


def labeled_mle_golden(shots, lo=-np.pi / 2 + 1e-6, hi=np.pi / 2 - 1e-6, tol=1e-10):
    """Scipy-free golden-section maximization of the labeled log-likelihood.

    General fallback that does not assume the closed form; used to cross-check
    labeled_mle. Returns the maximizing theta.
    """
    shots = np.asarray(shots, dtype=float)
    y = shots[:, 0] * shots[:, 1]
    invphi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = _loglik(c, y), _loglik(d, y)
    while (b - a) > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = _loglik(c, y)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = _loglik(d, y)
    return float((a + b) / 2)


def marginal_estimate(shots, mu):
    """Attempt to estimate theta from outcomes b alone, ignoring labels.

    The marginal outcome mean is E[b] = mu sin theta, so sin(theta) = mean(b)/mu.
    At full twirl (mu = 0) this is undefined (0/0): theta is unidentifiable and the
    estimator diverges — the whole point of standard RC losing the information.
    Returns (theta_hat_or_nan, identifiable_flag).
    """
    shots = np.asarray(shots, dtype=float)
    b = shots[:, 1]
    if abs(mu) < 1e-12:
        return float("nan"), False
    val = np.mean(b) / mu
    if abs(val) > 1.0:
        return float(np.sign(val) * np.pi / 2), True
    return float(np.arcsin(val)), True


def sample_labeled_shots(theta_true, M, p=0.5, rng=None):
    """Draw M labeled shots (s, b) from the E/R model at theta_true, twirl P(+1)=p."""
    if rng is None:
        rng = np.random.default_rng()
    s = np.where(rng.random(M) < p, 1, -1)
    mean = s * np.sin(theta_true)                 # <b|s> = s sin theta
    b = np.where(rng.random(M) < (1 + mean) / 2, 1, -1)
    return np.column_stack([s, b])


def estimator_variance_vs_crb(theta_true, M, F0=1.0, trials=400, p=0.5, rng=None):
    """Compare empirical Var(theta_hat) of the labeled MLE to the Cramer-Rao bound 1/(M F0).

    Returns (var_empirical, crb, bias). At the quantum limit F0 = 1 the labeled MLE
    saturates the CRB, so var_empirical ~ 1/M.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ests = np.array([labeled_mle(sample_labeled_shots(theta_true, M, p, rng))
                     for _ in range(trials)])
    return float(np.var(ests)), float(1.0 / (M * F0)), float(np.mean(ests) - theta_true)


# ===========================================================================
# Noise-aware calibration model + estimators
# ===========================================================================
# Single-qubit calibration: prepare |+>, apply the coherent Z-error Rz(theta), one
# incoherent noise layer, measure Y; the RC twirl is the X-dressing P in {I, X}
# recorded as the sign label s (X flips the coherent sign AND conjugates the noise),
# and asymmetric readout is applied to the outcome. Under this model the FULL-TWIRL
# labeled correlation E[b s] is immune to any label-independent additive offset
# (readout, non-unital T1) and only shrinks multiplicatively, while the marginal /
# partial-twirl estimates are biased by such offsets.

_Xmat = np.array([[0, 1], [1, 0]], dtype=complex)
_Ymat = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)


def calib_probs(theta, s, profile):
    """Exact [P(b=+1), P(b=-1)] for the noisy single-qubit calibration model.

    s in {+1,-1} is the RC sign label (X-dressing for s=-1). Measures Y, then applies
    the asymmetric readout confusion of `profile`. Requires core.noise (imported lazily
    to avoid a hard dependency when only the clean model is used).
    """
    from .noise import apply_incoherent_layer, readout_confusion
    from .single_qubit_gates import Rz
    P = np.eye(2, dtype=complex) if s == 1 else _Xmat
    rho = np.outer(_PLUS, _PLUS.conj())
    rho = P @ rho @ P
    U = Rz(theta)
    rho = U @ rho @ U.conj().T
    rho = apply_incoherent_layer(rho, profile, 1, mode="dm")
    rho = P @ rho @ P
    ey = np.trace(rho @ _Ymat).real
    p_true = np.array([(1 + ey) / 2, (1 - ey) / 2])
    return readout_confusion(profile) @ p_true


def labeled_correlation(theta, profile, p=0.5):
    """Signal statistic C(theta) = E[b s] under the noisy model (twirl P(+1)=p)."""
    C = 0.0
    for s, pi in [(1, p), (-1, 1 - p)]:
        pr = calib_probs(theta, s, profile)
        C += pi * s * (pr[0] - pr[1])
    return float(C)


def marginal_mean(theta, profile, p=0.5):
    """Label-forgotten outcome mean E[b] under the noisy model."""
    m = 0.0
    for s, pi in [(1, p), (-1, 1 - p)]:
        pr = calib_probs(theta, s, profile)
        m += pi * (pr[0] - pr[1])
    return float(m)


def noisy_labeled_fisher(theta, profile, p=0.5, eps=1e-5):
    """Per-shot labeled Fisher information F_noisy = E_s[ (dm_s/dtheta)^2 / (1-m_s^2) ]."""
    from .fisher import binary_fisher
    F = 0.0
    for s, pi in [(1, p), (-1, 1 - p)]:
        m = calib_probs(theta, s, profile)[0] - calib_probs(theta, s, profile)[1]
        mp = calib_probs(theta + eps, s, profile)[0] - calib_probs(theta + eps, s, profile)[1]
        mm = calib_probs(theta - eps, s, profile)[0] - calib_probs(theta - eps, s, profile)[1]
        F += pi * binary_fisher(m, (mp - mm) / (2 * eps))
    return float(F)


def noise_aware_mle(shots, profile, lo=-np.pi / 2 + 1e-6, hi=np.pi / 2 - 1e-6, tol=1e-8):
    """MLE of theta using the EXACT noisy model likelihood (golden-section).

    shots: iterable of (s, b) with s,b in {+1,-1}. Because the model is correct, this
    estimator is (asymptotically) unbiased under the full noise profile.
    """
    shots = np.asarray(shots, dtype=float)
    s_arr, b_arr = shots[:, 0], shots[:, 1]

    def negll(theta):
        ll = 0.0
        for s in (1, -1):
            mask = s_arr == s
            if not mask.any():
                continue
            pr = calib_probs(theta, s, profile)          # [P(+1), P(-1)]
            bvals = b_arr[mask]
            ppos = np.clip(pr[0], 1e-12, 1)
            pneg = np.clip(pr[1], 1e-12, 1)
            ll += np.sum(np.where(bvals > 0, np.log(ppos), np.log(pneg)))
        return -ll

    invphi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = negll(c), negll(d)
    while (b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = negll(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = negll(d)
    return float((a + b) / 2)


def sample_noisy_shots(theta_true, M, profile, p=0.5, rng=None):
    """Draw M labeled shots (s, b) from the noisy calibration model."""
    if rng is None:
        rng = np.random.default_rng()
    s = np.where(rng.random(M) < p, 1, -1)
    out = np.empty((M, 2), dtype=int)
    out[:, 0] = s
    # cache probabilities per label
    pr_plus = calib_probs(theta_true, 1, profile)
    pr_minus = calib_probs(theta_true, -1, profile)
    for i in range(M):
        pr = pr_plus if s[i] == 1 else pr_minus
        out[i, 1] = 1 if rng.random() < pr[0] else -1
    return out
