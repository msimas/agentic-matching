"""Fellegi-Sunter EM assumes fields are conditionally independent given match status.
Before a proposed attribute set is handed to EM, flag pairs of attributes that are too
associated with each other on the block's own record population (computed by pooling
both sides' computed values) using Cramer's V.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0
    r, c = table.shape
    return float(np.sqrt((chi2 / n) / (min(r, c) - 1)))


def check_correlations(
    values_df: pd.DataFrame, threshold: float = 0.8
) -> list[dict[str, Any]]:
    """values_df: one column per attribute name, one row per (pooled FNDDS+OFF) record.
    Returns flags for attribute pairs whose Cramer's V exceeds `threshold`."""
    flags: list[dict[str, Any]] = []
    cols = list(values_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            v = cramers_v(values_df[a].astype(str), values_df[b].astype(str))
            if v >= threshold:
                flags.append({"attribute_a": a, "attribute_b": b, "cramers_v": round(v, 3)})
    return sorted(flags, key=lambda f: -f["cramers_v"])
