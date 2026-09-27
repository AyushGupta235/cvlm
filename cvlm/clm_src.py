"""Imports from the pinned CLM checkout (third_party/CLM).

``clm`` itself is pip-installed with --no-deps; ``train/adapters.py`` is not part of the
package, so it is imported from the checkout by path.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "CLM")

if not os.path.isdir(REPO):
    raise ImportError(f"CLM checkout missing at {REPO}; run scripts/setup_local.sh (or runpod/remote_setup.sh)")
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

import adapters  # noqa: E402  (third_party/CLM/train/adapters.py)
from clm import heads, schema  # noqa: E402

__all__ = ["adapters", "heads", "schema"]
