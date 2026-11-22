"""The Referee's statistics: AUC and paired bootstrap confidence intervals.

Uses the C++ library in native/ (build with `make -C exo/native`) and falls back to NumPy
with identical results if it is not built. Both resample host stars by default: planets
around the same star are not independent, so resampling single objects would make the
interval too narrow and let noise through the NOISE and OVERFIT gates.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

_LIB = None
for name in ("libfastci.dylib", "libfastci.so"):
    p = Path(__file__).resolve().parent / "native" / name
    if p.exists():
        _LIB = ctypes.CDLL(str(p))
        break

if _LIB is not None:
    _D = ctypes.POINTER(ctypes.c_double)
    _I8 = ctypes.POINTER(ctypes.c_int8)
    _I64 = ctypes.POINTER(ctypes.c_int64)
    _LIB.fastci_auc.restype = ctypes.c_double
    _LIB.fastci_auc.argtypes = [_D, _I8, ctypes.c_int64]
    _LIB.fastci_paired_ci.restype = None
    _LIB.fastci_paired_ci.argtypes = [_D, _D, _I8, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint64, _D, _D]
    _LIB.fastci_paired_ci_star.restype = None
    _LIB.fastci_paired_ci_star.argtypes = [_D, _D, _I8, _I64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                           ctypes.c_uint64, _D, _D]

NATIVE = _LIB is not None


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _i8(a):
    return np.ascontiguousarray(a, dtype=np.int8)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC; tied predictions share their average rank (matches sklearn's roc_auc_score)."""
    y, p = _i8(y), _f64(p)
    if NATIVE:
        return float(_LIB.fastci_auc(p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     y.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), len(y)))
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_ci(y, c, b, groups=None, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI of auc(y, c) - auc(y, b), resampling the same items for both models.

    groups: cluster id per item (host star). When given, whole stars are resampled.
    """
    y, c, b = _i8(y), _f64(c), _f64(b)
    lo, hi = ctypes.c_double(), ctypes.c_double()
    if NATIVE:
        D = ctypes.POINTER(ctypes.c_double)
        args = (c.ctypes.data_as(D), b.ctypes.data_as(D), y.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        if groups is None:
            _LIB.fastci_paired_ci(*args, len(y), n, seed, ctypes.byref(lo), ctypes.byref(hi))
        else:
            codes, uniq = _codes(groups)
            _LIB.fastci_paired_ci_star(*args, codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(y),
                                       len(uniq), n, seed, ctypes.byref(lo), ctypes.byref(hi))
        return lo.value, hi.value
    return _paired_ci_numpy(y, c, b, groups, n, seed)


def _codes(groups):
    uniq, codes = np.unique(np.asarray(groups), return_inverse=True)
    return np.ascontiguousarray(codes, dtype=np.int64), uniq


def _paired_ci_numpy(y, c, b, groups, n, seed):
    g = np.random.default_rng(seed)
    if groups is not None:
        codes, uniq = _codes(groups)
        members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    out = []
    for _ in range(n):
        if groups is None:
            i = g.integers(0, len(y), len(y))
        else:
            i = np.concatenate([members[k] for k in g.integers(0, len(members), len(members))])
        if y[i].min() == y[i].max():
            continue
        out.append(auc(y[i], c[i]) - auc(y[i], b[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


if __name__ == "__main__":
    # Self-check: native and NumPy agree on AUC, and the bench shows why the hot path is in C++.
    import time
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1535)
    c = np.round(rng.random(1535) * .6 + y * .3, 2)
    b = np.round(rng.random(1535) * .6 + y * .25, 2)
    stars = rng.integers(0, 1328, 1535)
    print("native library:", NATIVE)
    print("auc", auc(y, c), "sklearn", roc_auc_score(y, c))
    t = time.time(); r1 = paired_ci(y, c, b, stars); t1 = time.time() - t
    nat, NATIVE = NATIVE, False
    t = time.time(); r2 = paired_ci(y, c, b, stars); t2 = time.time() - t
    NATIVE = nat
    print(f"star bootstrap CI  native {r1[0]:+.4f} {r1[1]:+.4f} in {t1:.2f}s | numpy {r2[0]:+.4f} {r2[1]:+.4f} in {t2:.2f}s")
    sys.exit(0 if abs(auc(y, c) - roc_auc_score(y, c)) < 1e-12 else 1)
