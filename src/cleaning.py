"""Read and clean GPU data."""

from pathlib import Path
import numpy as np
import pandas as pd
from .config import NUMERIC_COLUMNS, COMMON_INPUTS


def read_number(series: pd.Series) -> pd.Series:
    """Extract numbers from values like '150 Watts', '$1,299.00', and '1024 MB'; '-' becomes NaN."""
    text = series.astype("string").str.strip()
    number = text.str.extract(r"(-?\d[\d,]*(?:\.\d+)?)", expand=False)
    return pd.to_numeric(number.str.replace(",", "", regex=False), errors="coerce")


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Read the CSV, standardize units, and clean the required columns."""
    df = pd.read_csv(csv_path)
    required = set(NUMERIC_COLUMNS + COMMON_INPUTS + ["Release_Date", "Name"])
    required.discard("Release_Year")  # This column is generated from Release_Date.
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError(f"CSV is missing required columns: {', '.join(absent)}")

    for col in NUMERIC_COLUMNS:
        if col != "Release_Year":
            original = df[col]
            df[col] = read_number(original)
            if col == "Memory_Bandwidth":
                # 4 rows in the data are MB/sec; the rest are GB/sec.
                in_mb_per_second = original.astype("string").str.contains(
                    "MB/sec", case=False, na=False, regex=False
                )
                df.loc[in_mb_per_second, col] /= 1000  # Decimal GB.

    dates = pd.to_datetime(
        df["Release_Date"].astype("string").str.strip(),
        format="%d-%b-%Y", errors="coerce",
    )
    df["Release_Year"] = dates.dt.year.astype(float)
    for col in ("Manufacturer", "Notebook_GPU", "Integrated"):
        cleaned = df[col].astype("string").str.strip().replace("", pd.NA)
        df[col] = cleaned.astype(object).where(cleaned.notna(), np.nan)

    # GPU prices labeled 'not released' are not observed launch prices.
    unreleased = df["Name"].astype("string").str.contains(
        "not released", case=False, na=False, regex=False
    )
    df.attrs["unreleased_price_count"] = int(
        (unreleased & df["Release_Price"].notna()).sum()
    )
    df.loc[unreleased, "Release_Price"] = np.nan
    return df
