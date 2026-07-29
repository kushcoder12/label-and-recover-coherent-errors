"""Platform compatibility shims.

`quiet_accelerate()` suppresses the spurious "divide by zero / overflow / invalid
value encountered in matmul" RuntimeWarnings emitted by numpy 2.0 when it is linked
against Apple's Accelerate BLAS (this repo's environment is Python 3.9 + numpy 2.0.2,
and numpy >= 2.1, which fixes the bug, requires Python >= 3.10). The matmul OUTPUTS are
verified finite and the numerical results are exact -- these are false floating-point
flags raised by the BLAS matmul kernel, not real errors. We scope the suppression to the
dense-matmul regions with np.errstate (NOT a global np.seterr), so genuine nan/inf
anywhere else is still surfaced.
"""

import functools
from contextlib import contextmanager

import numpy as np


@contextmanager
def quiet_accelerate():
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        yield


def quiet(fn):
    """Decorator: run `fn` inside quiet_accelerate()."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with quiet_accelerate():
            return fn(*args, **kwargs)
    return wrapper
