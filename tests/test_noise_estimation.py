"""Noise-estimation / robustness verification (NOISE_MODULE.md §7).

- Noise-aware labeled MLE is unbiased under the full profile; naive is shrink-biased.
- Empirical Var(theta_hat) of the noise-aware MLE saturates the noisy CRB 1/(M F_noisy).
- Full-twirl labeled correlation is immune to the readout additive offset (cancels),
  whereas the marginal picks it up.
- exp7: the noiseless profile reproduces Oracle O2.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from core.noise import NoiseProfile
from core.estimator import (labeled_correlation, marginal_mean, noisy_labeled_fisher,
                            noise_aware_mle, labeled_mle, sample_noisy_shots)

_THETA = 0.3
_PROFILES = {
    "T2": NoiseProfile(1e-6, T1=1e9, T2=4e-6),
    "T1": NoiseProfile(1e-6, T1=3e-6, T2=6e-6),
    "depol": NoiseProfile(1e-6, T1=np.inf, T2=np.inf, r_1q=0.05),
    "readout": NoiseProfile(1e-6, T1=np.inf, T2=np.inf, e01=0.02, e10=0.08),
}


def test_noise_aware_unbiased_naive_biased():
    rng = np.random.default_rng(0)
    for name, pr in _PROFILES.items():
        aware, naive = [], []
        for _ in range(300):
            sh = sample_noisy_shots(_THETA, 4000, pr, rng=rng)
            aware.append(noise_aware_mle(sh, pr))
            naive.append(labeled_mle(sh))
        assert abs(np.mean(aware) - _THETA) < 5e-3, f"{name}: aware biased {np.mean(aware)-_THETA}"
        # naive is shrink-biased low (except readout has a smaller shrink); require clearly nonzero
        assert (np.mean(naive) - _THETA) < -5e-3, f"{name}: naive not shrink-biased"


def test_crb_saturation_under_noise():
    rng = np.random.default_rng(1)
    for name, pr in _PROFILES.items():
        F = noisy_labeled_fisher(_THETA, pr)
        M = 4000
        ests = np.array([noise_aware_mle(sample_noisy_shots(_THETA, M, pr, rng=rng), pr)
                         for _ in range(300)])
        ratio = np.var(ests) * M * F         # ~1 if CRB saturated
        assert 0.7 < ratio < 1.4, f"{name}: Var*M*F = {ratio:.3f} (CRB not saturated)"


def test_readout_offset_cancels_in_full_twirl():
    pr = NoiseProfile(1e-6, T1=np.inf, T2=np.inf, e01=0.02, e10=0.08)
    offset = pr.e10 - pr.e01                  # 0.06
    # marginal at partial twirl picks up the offset; full-twirl correlation does not
    marg_mu0 = marginal_mean(_THETA, pr, p=0.5)         # E[b] at mu=0 == offset
    assert abs(marg_mu0 - offset) < 1e-9, f"marginal(mu=0) {marg_mu0} != offset {offset}"
    C_full = labeled_correlation(_THETA, pr, p=0.5)      # offset-free (shrunk sin)
    C_clean = labeled_correlation(_THETA, NoiseProfile(1e-6), p=0.5)
    shrink = 1 - pr.e01 - pr.e10
    assert abs(C_full - shrink * C_clean) < 1e-9, "labeled correlation carries a non-shrink term"


def test_exp7_noiseless_recovers_O2():
    from exp7_saturation_full import verify_noiseless_recovers_O2
    assert verify_noiseless_recovers_O2(verbose=False) < 1e-9


def run_all():
    for t in [test_noise_aware_unbiased_naive_biased, test_crb_saturation_under_noise,
              test_readout_offset_cancels_in_full_twirl, test_exp7_noiseless_recovers_O2]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_noise_estimation: ALL PASS")


if __name__ == "__main__":
    run_all()
