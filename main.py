import argparse
from pathlib import Path

import pandas as pd

from src.analysis import analyze_regression_extension, analyze_statistics
from src.cleaning import load_and_clean
from src.config import NUMERIC_COLUMNS, TASKS
from src.statistics import categorical_summary, numeric_summary
from src.visualization import (
    cleaning_report_chart,
    dataset_overview_chart,
    missing_value_chart,
    summary_table_charts,
)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=project_dir / "data" / "All_GPUs.csv",
        help="Path to the CSV file; default is data/All_GPUs.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=project_dir / "results",
        help="Directory to save results; default is results/",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    statistics_out = args.output / "statistics"
    regression_out = args.output / "regression_extension"
    statistics_out.mkdir(parents=True, exist_ok=True)
    regression_out.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(args.data)
    pd.DataFrame({
        "column": NUMERIC_COLUMNS,
        "missing_values": df[NUMERIC_COLUMNS].isna().sum().values,
        "missing_rate": df[NUMERIC_COLUMNS].isna().mean().values,
    }).to_csv(statistics_out / "missing_values.csv", index=False)

    dataset_overview_chart(df, args.data.name, statistics_out)
    cleaning_report_chart(df, statistics_out)
    summary_table_charts(
        numeric_summary(df), categorical_summary(df), statistics_out
    )
    missing_value_chart(df, statistics_out)

    statistics_lines = [
        "PRIMARY ANALYSIS: DESCRIPTIVE AND INFERENTIAL STATISTICS",
        "=" * 55,
        f"Source: {args.data.name}; total GPUs: {len(df)}.",
        f"Price: removed {df.attrs['unreleased_price_count']} rows labeled "
        "[not released] from the target variable.",
        "Correlation describes the sample only and does not imply causation.",
        "The 95% confidence interval assumes the sample is representative/independent; the CSV may not be a random sample.",
        "Inference is limited to GPUs similar to the data (release years 1998-2017).",
        "Release prices are sparse and contain large outliers; do not infer current market prices from them.",
        "",
    ]
    for target in TASKS:
        statistics_lines.extend(analyze_statistics(df, target, statistics_out))
        statistics_lines.append("")

    extension_lines = [
        "OPTIONAL EXTENSION: REGRESSION PREDICTION",
        "=" * 41,
        "This section is supplementary and is not the main statistical analysis.",
        "",
    ]
    for target in TASKS:
        extension_lines.extend(
            analyze_regression_extension(df, target, regression_out)
        )
        extension_lines.append("")

    statistics_report = "\n".join(statistics_lines)
    extension_report = "\n".join(extension_lines)
    (statistics_out / "statistics_summary.txt").write_text(
        statistics_report, encoding="utf-8"
    )
    (regression_out / "regression_summary.txt").write_text(
        extension_report, encoding="utf-8"
    )
    report = f"{statistics_report}\n\n{extension_report}"
    (args.output / "summary.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
