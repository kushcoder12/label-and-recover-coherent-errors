"""Make ``labeled_rc`` (this folder's parent) and the repo's ``core`` importable."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)  # pennylane_impl/
REPO_ROOT = os.path.dirname(PKG_ROOT)  # repository root, holds core/

for path in (PKG_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
