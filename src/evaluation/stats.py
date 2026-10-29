"""Statistical comparison across independent seeds (Table 3.19 / 4.13): paired t-test, Cohen's d, 95% CI."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats


def summarise(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    else:
        h = float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean - h, "ci95_high": mean + h}


def paired_comparison(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-tailed paired t-test of a vs b (same seeds). Cohen's d_z = mean(diff)/sd(diff).

    d_z explodes when the seed-to-seed spread of the difference is tiny; with n=10 it should be read
    next to the confidence interval, not on its own.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 2:
        return {"n": int(n), "note": "need >= 2 paired seeds"}
    diff = a - b
    sd = diff.std(ddof=1)
    t, p = stats.ttest_rel(a, b)
    dz = float(diff.mean() / sd) if sd > 0 else float("inf") if diff.mean() != 0 else 0.0
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"n": int(n), "mean_diff": float(diff.mean()), "ci95_low": float(diff.mean() - h),
            "ci95_high": float(diff.mean() + h), "t": float(t), "p": float(p), "df": int(n - 1),
            "cohens_dz": dz, "significant_0.05": bool(p < 0.05)}
