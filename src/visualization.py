"""Generate and save statistical and regression plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Allow the script to run in a headless terminal session.
import matplotlib.pyplot as plt
import pandas as pd

from .config import TASKS


def exploratory_charts(df: pd.DataFrame, target: str, out: Path) -> None:
    """Create distribution, manufacturer boxplots, and yearly trend charts."""
    label = TASKS[target]["label"]
    subset = df[df[target].notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(subset[target], bins=35, color="#3182bd", edgecolor="white")
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
    fig.tight_layout()
    fig.savefig(out / f"01_{target}_statistics.png", dpi=180)
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
    fig.suptitle(TASKS[target]["label"])
    fig.tight_layout()
    fig.savefig(out / f"02_{target}_regression.png", dpi=180)
    plt.close(fig)
