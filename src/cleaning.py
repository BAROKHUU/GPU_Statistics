"""Read and clean the GPU dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMON_INPUTS, NUMERIC_COLUMNS


def read_number(series: pd.Series) -> pd.Series:
    """Extract numeric values from strings like '150 Watts', '$1,299.00', '1024 MB'; '-' becomes NaN."""
    text = series.astype("string").str.strip()
    number = text.str.extract(r"(-?\d[\d,]*(?:\.\d+)?)", expand=False)
    return pd.to_numeric(number.str.replace(",", "", regex=False), errors="coerce")


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Read the CSV, normalize units, and clean the required columns."""
    df = pd.read_csv(csv_path)
    required = set(NUMERIC_COLUMNS + COMMON_INPUTS + ["Release_Date", "Name"])
    required.discard("Release_Year")  # This column is created from Release_Date.
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError(f"CSV is missing columns: {', '.join(absent)}")

    for col in NUMERIC_COLUMNS:
        if col != "Release_Year":
            original = df[col]
            df[col] = read_number(original)
            if col == "Memory_Bandwidth":
                # Some rows use MB/sec while others use GB/sec.
                in_mb_per_second = original.astype("string").str.contains(
                    "MB/sec", case=False, na=False, regex=False
                )
                df.loc[in_mb_per_second, col] /= 1000  # Convert to decimal GB.

    dates = pd.to_datetime(
        df["Release_Date"].astype("string").str.strip(),
        format="%d-%b-%Y", errors="coerce",
    )
    df["Release_Year"] = dates.dt.year.astype(float)
    for col in ("Manufacturer", "Notebook_GPU", "Integrated"):
        cleaned = df[col].astype("string").str.strip().replace("", pd.NA)
        df[col] = cleaned.astype(object).where(cleaned.notna(), np.nan)

    # GPUs marked as 'not released' do not have a valid observed launch price.
    unreleased = df["Name"].astype("string").str.contains(
        "not released", case=False, na=False, regex=False
    )
    df.attrs["unreleased_price_count"] = int(
        (unreleased & df["Release_Price"].notna()).sum()
    )
    df.loc[unreleased, "Release_Price"] = np.nan
    return df
