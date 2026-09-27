"""Small compatibility shims for supported third-party dependency versions."""

from __future__ import annotations

import scipy.integrate


if not hasattr(scipy.integrate, "simps"):
    # pycox 0.3 calls the removed positional ``simps(y, x)`` API.  SciPy
    # 1.14+ exposes ``simpson`` and makes ``x`` keyword-only.
    def _simps(y, x=None, dx=1.0, axis=-1, even=None):
        del even
        return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)

    scipy.integrate.simps = _simps
