import argparse
from pathlib import Path

import pandas as pd

from src.analysis import analyze_task
from src.cleaning import load_and_clean
from src.config import NUMERIC_COLUMNS, TASKS


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

    df = load_and_clean(args.data)
    pd.DataFrame({
        "column": NUMERIC_COLUMNS,
        "missing_values": df[NUMERIC_COLUMNS].isna().sum().values,
        "missing_rate": df[NUMERIC_COLUMNS].isna().mean().values,
    }).to_csv(args.output / "missing_values.csv", index=False)

    lines = [
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
        lines.extend(analyze_task(df, target, args.output))
        lines.append("")

    report = "\n".join(lines)
    (args.output / "summary.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
