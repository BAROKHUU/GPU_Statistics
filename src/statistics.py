"""Descriptive statistics and Pearson correlation analysis."""

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr, t

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
            result = pearsonr(pair[feature], pair[target])
            r, p = result.statistic, result.pvalue
            ci = result.confidence_interval(confidence_level=0.95)
            rows.append({
                "input_feature": feature,
                "pair_count": len(pair),
                "pearson_r": r,
                "ci95_low": ci.low,
                "ci95_high": ci.high,
                "p_value": p,
            })
    result = pd.DataFrame(rows).sort_values(
        "pearson_r", key=lambda col: col.abs(), ascending=False
    )
    # Bonferroni controls the family-wise Type-I error across tested features.
    result["p_bonferroni"] = (result["p_value"] * len(result)).clip(upper=1)
    result["significant_0.05"] = result["p_bonferroni"] < 0.05
    return result


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation-ready summary of all analysis variables."""
    rows = []
    for col in [x for x in NUMERIC_COLUMNS if x in df.columns]:
        values = df[col].dropna()
        q1, q3 = values.quantile(.25), values.quantile(.75)
        iqr = q3 - q1
        outliers = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
        rows.append({
            "variable": col, "n": len(values), "missing": df[col].isna().sum(),
            "mean": values.mean(), "std": values.std(ddof=1),
            "min": values.min(), "q1": q1, "median": values.median(), "q3": q3,
            "max": values.max(), "skew": values.skew(),
            "iqr_outliers": int(outliers),
        })
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the categorical variables used by the models."""
    rows = []
    for col in ("Manufacturer", "Notebook_GPU", "Integrated"):
        counts = df[col].value_counts(dropna=True)
        rows.append({
            "variable": col, "n": int(df[col].notna().sum()),
            "missing": int(df[col].isna().sum()), "levels": len(counts),
            "mode": counts.index[0] if len(counts) else "-",
            "mode_count": int(counts.iloc[0]) if len(counts) else 0,
            "mode_percent": 100 * counts.iloc[0] / counts.sum() if len(counts) else np.nan,
        })
    return pd.DataFrame(rows)


def manufacturer_test(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis test of equal target distributions across manufacturers."""
    subset = df[["Manufacturer", target]].dropna()
    grouped = [(name, values[target].to_numpy())
               for name, values in subset.groupby("Manufacturer") if len(values) >= 5]
    h, p = kruskal(*(values for _, values in grouped))
    n, k = sum(len(values) for _, values in grouped), len(grouped)
    epsilon_squared = max(0.0, (h - k + 1) / (n - k)) if n > k else np.nan
    result = pd.DataFrame([{
        "test": "Kruskal-Wallis", "target": target,
        "group_variable": "Manufacturer", "groups": k, "n": n,
        "statistic_H": h, "degrees_of_freedom": k - 1, "p_value": p,
        "epsilon_squared": epsilon_squared,
        "significant_0.05": p < 0.05,
    }])
    details = pd.DataFrame([
        {"Manufacturer": name, "n": len(values),
         "median": np.median(values), "mean_rank_input": "raw observations"}
        for name, values in grouped
    ])
    return result, details
