"""Generate and save statistical and regression plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Allow the script to run in a headless terminal session.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import NUMERIC_COLUMNS, TASKS


def _statistics_figure_number(target: str, position: int) -> int:
    """Number the three statistical figures for each target consecutively."""
    return (2 if target == "Max_Power" else 5) + position


def _extension_figure_number(target: str, position: int) -> int:
    """Use an independent sequence for optional regression figures."""
    return (1 if target == "Max_Power" else 3) + position


def _save_table(table: pd.DataFrame, title: str, path: Path,
                formats: dict[str, str] | None = None) -> None:
    """Render a dataframe as a numbered PNG table."""
    display = table.copy()
    for column, fmt in (formats or {}).items():
        if column in display:
            display[column] = display[column].map(
                lambda value: fmt.format(value) if pd.notna(value) else "-"
            )
    height = max(2.8, 0.38 * len(display) + 1.5)
    width = max(9.5, 1.25 * len(display.columns))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    tab = ax.table(cellText=display.values, colLabels=display.columns,
                   cellLoc="center", colLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.35)
    for (row, _), cell in tab.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9eaf7")
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dataset_overview_chart(df: pd.DataFrame, source_name: str, out: Path) -> None:
    """Save observation count, variable roles/types, and collection metadata."""
    rows = []
    roles = {
        "Name": "Identifier", "Release_Date": "Time",
        "Manufacturer": "Categorical predictor", "Notebook_GPU": "Binary predictor",
        "Integrated": "Binary predictor", "Max_Power": "Numeric target",
        "Release_Price": "Numeric target",
    }
    for col in ["Name", "Manufacturer", "Notebook_GPU", "Integrated",
                "Release_Date", *NUMERIC_COLUMNS]:
        if col not in df or col == "Release_Year" and "Release_Date" not in df:
            continue
        role = roles.get(col, "Numeric predictor")
        kind = "Quantitative (continuous)" if col in NUMERIC_COLUMNS else (
            "Date" if col == "Release_Date" else "Qualitative (nominal)")
        rows.append({"Variable": col, "Role": role, "Type": kind,
                     "Observed": int(df[col].notna().sum()),
                     "Missing": int(df[col].isna().sum())})
    title = (f"Table 1. Dataset overview — {len(df):,} observations, "
             f"{df.attrs['cleaning_audit']['raw_columns']} source variables ({source_name})")
    _save_table(pd.DataFrame(rows), title, out / "Table_01_dataset_overview.png")


def cleaning_report_chart(df: pd.DataFrame, out: Path) -> None:
    audit = df.attrs["cleaning_audit"]
    labels = {
        "raw_rows": "Rows read", "raw_columns": "Columns read",
        "duplicate_rows": "Exact duplicate rows found",
        "numeric_columns_converted": "Text numeric columns converted",
        "numeric_parse_failures": "Non-numeric/blank values -> missing",
        "bandwidth_units_converted": "MB/sec values converted to GB/sec",
        "invalid_dates": "Invalid/missing release dates",
        "unreleased_prices_removed": "Unreleased prices -> missing",
        "rows_deleted": "Rows deleted globally",
    }
    table = pd.DataFrame({"Cleaning operation / check": [labels[k] for k in labels],
                          "Count": [audit[k] for k in labels]})
    _save_table(table, "Table 2. Data cleaning and transformation audit",
                out / "Table_02_cleaning_audit.png")


def summary_table_charts(numeric: pd.DataFrame, categorical: pd.DataFrame,
                         out: Path) -> None:
    numeric.to_csv(out / "numeric_summary.csv", index=False)
    categorical.to_csv(out / "categorical_summary.csv", index=False)
    formats = {x: "{:.2f}" for x in
               ("mean", "std", "min", "q1", "median", "q3", "max", "skew")}
    _save_table(numeric, "Table 3. Descriptive statistics for quantitative variables",
                out / "Table_03_numeric_summary.png", formats)
    _save_table(categorical, "Table 4. Descriptive statistics for categorical variables",
                out / "Table_04_categorical_summary.png", {"mode_percent": "{:.1f}%"})


def missing_value_chart(df: pd.DataFrame, out: Path) -> None:
    """Show the number of missing values in numeric fields."""
    cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    missing = df[cols].isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot(kind="bar", ax=ax, color="#4c78a8", edgecolor="black")
    ax.set_title("Figure 1. Missing values after parsing and validation")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "Figure_01_missing_values.png", dpi=180)
    plt.close(fig)


def exploratory_charts(df: pd.DataFrame, target: str, out: Path) -> None:
    """Create distribution, manufacturer boxplots, and yearly trend charts."""
    label = TASKS[target]["label"]
    subset = df[df[target].notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    bins = 35
    if target == "Release_Price":
        # Log-spaced bins avoid visually merging most prices into one wide bin.
        bins = np.geomspace(subset[target].min(), subset[target].max(), 36)
    axes[0].hist(subset[target], bins=bins, color="#3182bd", edgecolor="white")
    axes[0].axvline(subset[target].median(), color="#c51b27", label="Median")
    axes[0].set(xlabel=label, ylabel="GPU count", title="Distribution")
    axes[0].legend()
    if target == "Release_Price":
        # Scale the axis only for visualization; do not remove expensive GPUs.
        axes[0].set_xscale("log")

    brands = subset.groupby("Manufacturer")[target].apply(
        lambda x: x.dropna().to_numpy()
    )
    axes[1].boxplot(brands.tolist(), tick_labels=brands.index.tolist(),
                    showfliers=False)
    axes[1].set(xlabel="Manufacturer", ylabel=label,
                title="By manufacturer (outliers hidden in plot)")

    yearly = subset.groupby("Release_Year")[target].agg(["median", "count"])
    yearly = yearly[yearly["count"] >= 5]
    axes[2].plot(yearly.index, yearly["median"], marker="o", color="#238b45")
    axes[2].set(xlabel="Release year", ylabel=f"Median {label}",
                title="Yearly median (>=5 GPUs/year)")
    axes[2].tick_params(axis="x", rotation=45)
    figure_number = _statistics_figure_number(target, 0)
    fig.suptitle(f"Figure {figure_number}. Sample distributions and trends — {label}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"Figure_{figure_number:02d}_{target}_descriptive_plots.png", dpi=180)
    plt.close(fig)


def correlation_heatmap(df: pd.DataFrame, target: str, out: Path) -> None:
    """Plot the correlation matrix between the target and numeric predictors."""
    cols = [col for col in TASKS[target]["inputs"]
            if col in NUMERIC_COLUMNS and col in df.columns]
    corr = df[cols + [target]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    figure_number = _statistics_figure_number(target, 1)
    ax.set_title(f"Figure {figure_number}. Pairwise Pearson correlation matrix — {target}")
    for i in range(len(corr.columns)):
        for j in range(len(corr.index)):
            value = corr.iloc[j, i]
            ax.text(i, j, f"{value:.2f}", ha="center", va="center",
                    color="black" if abs(value) < 0.7 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / f"Figure_{figure_number:02d}_{target}_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def feature_scatterplots(df: pd.DataFrame, target: str, out: Path, max_features: int = 3) -> None:
    """Plot the strongest numeric feature relationships with the target."""
    candidate_columns = [x for x in TASKS[target]["inputs"]
                         if x in NUMERIC_COLUMNS and x in df.columns]
    correlations = df[candidate_columns + [target]].corr(numeric_only=True)[target]
    top_features = correlations.drop(labels=[target], errors="ignore").abs().nlargest(max_features).index.tolist()
    if not top_features:
        return

    fig, axes = plt.subplots(1, len(top_features), figsize=(5 * len(top_features), 4.5))
    if len(top_features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, top_features):
        pair = df[[feature, target]].dropna()
        ax.scatter(pair[feature], pair[target], alpha=0.6, s=18, color="#2ca25f")
        if len(pair) > 1:
            slope, intercept = np.polyfit(pair[feature], pair[target], 1)
            xline = np.linspace(pair[feature].min(), pair[feature].max(), 100)
            ax.plot(xline, slope * xline + intercept, color="#c51b27", linewidth=1.5)
        ax.set_xlabel(feature)
        ax.set_ylabel(TASKS[target]["label"])
        ax.set_title(f"{feature} vs {TASKS[target]['label']}")
    figure_number = _statistics_figure_number(target, 2)
    fig.suptitle(f"Figure {figure_number}. Strongest pairwise numeric relationships — {TASKS[target]['label']}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"Figure_{figure_number:02d}_{target}_feature_scatter.png", dpi=180)
    plt.close(fig)


def model_chart(y_test, pred, target: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].scatter(y_test, pred, alpha=0.55, s=16, color="#3182bd")
    bound = max(float(y_test.max()), float(pred.max()))
    axes[0].plot([0, bound], [0, bound], "--", color="#c51b27")
    axes[0].set(xlabel="Observed", ylabel="Predicted", title="Test set")
    axes[1].hist(y_test.to_numpy() - pred, bins=30, color="#756bb1",
                 edgecolor="white")
    axes[1].set(xlabel="Observed - predicted", ylabel="GPU count",
                title="Test residuals")
    for ax in axes:
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    figure_number = _extension_figure_number(target, 1)
    fig.suptitle(f"Extension Figure {figure_number}. Hold-out regression diagnostics — {TASKS[target]['label']}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"Extension_Figure_{figure_number:02d}_{target}_regression.png", dpi=180)
    plt.close(fig)


def inference_chart(correlations: pd.DataFrame, group_test: pd.DataFrame,
                    target_stats: dict, target: str, out: Path) -> None:
    """Render inferential results with corrected p-values and effect sizes."""
    corr = correlations[["input_feature", "pair_count", "pearson_r",
                         "ci95_low", "ci95_high", "p_bonferroni",
                         "significant_0.05"]].copy()
    corr.columns = ["Feature", "n", "Pearson r", "CI low", "CI high",
                    "Adjusted p", "Significant"]
    row = group_test.iloc[0]
    footer = (f"Kruskal-Wallis across manufacturers: n={int(row['n'])}, "
              f"H({int(row['degrees_of_freedom'])})={row['statistic_H']:.2f}, "
              f"p={row['p_value']:.3g}, epsilon²={row['epsilon_squared']:.3f}")
    mean_ci = (f"Mean={target_stats['mean']:.2f}; 95% t-CI "
               f"[{target_stats['ci95_low']:.2f}, {target_stats['ci95_high']:.2f}]")
    table_number = 5 if target == "Max_Power" else 6
    title = f"Table {table_number}. Inferential tests — {TASKS[target]['label']}\n{mean_ci}; {footer}"
    _save_table(corr, title, out / f"Table_{table_number:02d}_{target}_inference.png",
                {"Pearson r": "{:.3f}", "CI low": "{:.3f}",
                 "CI high": "{:.3f}", "Adjusted p": "{:.3g}"})


def model_comparison_chart(cv_table: pd.DataFrame, test_metrics: pd.DataFrame,
                           target: str, out: Path) -> None:
    """Show cross-validation uncertainty and final test performance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(cv_table["model"], cv_table["CV_MAE"],
                yerr=cv_table["CV_MAE_std"], capsize=5, color="#4c78a8")
    axes[0].set(title="5-fold CV on training data", ylabel="MAE (lower is better)",
                xlabel="Candidate model")
    x = np.arange(len(test_metrics))
    width = .36
    axes[1].bar(x - width / 2, test_metrics["MAE"], width, label="MAE", color="#59a14f")
    axes[1].bar(x + width / 2, test_metrics["RMSE"], width, label="RMSE", color="#e15759")
    axes[1].set_xticks(x, test_metrics["model"], rotation=10)
    axes[1].set(title="Single hold-out test", ylabel="Error", xlabel="Model")
    axes[1].legend()
    figure_number = _extension_figure_number(target, 0)
    fig.suptitle(f"Extension Figure {figure_number}. Model comparison — {TASKS[target]['label']}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"Extension_Figure_{figure_number:02d}_{target}_model_comparison.png", dpi=180)
    plt.close(fig)
