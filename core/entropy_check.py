import numpy as np


def entropy(state, N, NA):
    a = state.reshape((2**NA), (2 ** (N - NA)))
    Lambda = np.linalg.svd(a, compute_uv=False)
    return -np.sum((Lambda**2) * np.log(Lambda**2 + 1e-20))
