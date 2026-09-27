"""Descriptive statistics and Pearson correlation analysis."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t

from .config import NUMERIC_COLUMNS, TASKS


def describe_target(df: pd.DataFrame, target: str) -> dict:
    """Compute descriptive statistics and a 95% confidence interval for the sample mean."""
    values = df[target].dropna()
    n = len(values)
    mean, median = values.mean(), values.median()
    sd = values.std(ddof=1)
    margin = t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {
        "target": target, "sample_size": n, "missing": len(df) - n,
        "mean": mean, "median": median, "std_dev": sd,
        "q1": values.quantile(0.25), "q3": values.quantile(0.75),
        "min": values.min(), "max": values.max(),
        "ci95_low": mean - margin, "ci95_high": mean + margin,
    }


def correlation_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute Pearson r and p-values for each candidate feature; valid pair counts may differ."""
    rows = []
    for feature in [x for x in TASKS[target]["inputs"] if x in NUMERIC_COLUMNS]:
        pair = df[[feature, target]].dropna()
        if len(pair) >= 3 and pair[feature].nunique() > 1:
            r, p = pearsonr(pair[feature], pair[target])
            rows.append({
                "input_feature": feature,
                "pair_count": len(pair),
                "pearson_r": r,
                "p_value": p,
            })
    return pd.DataFrame(rows).sort_values(
        "pearson_r", key=lambda col: col.abs(), ascending=False
    )
