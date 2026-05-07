from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


DEFAULT_CSV_PATH = Path(os.environ.get("RETURNS_CSV_PATH", "data/returns.csv"))

REASON_PREFIX = "Return reason: "
REASON_SUFFIX = " (%)"
CACHE_SCHEMA_VERSION = "v3"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".returns_cache"

REQUIRED_COLUMNS = [
    "Article variant",
    "Sold articles",
    "Returned articles",
    "Return rate (%)",
]

OPTIONAL_ANALYSIS_COLUMNS = [
    "NMV",
    "Country",
    "Category",
    "Article type",
    "Gender",
    "Season",
    "Zalando article variant",
    "Estimated return rate status",
    "Size-related return rate status",
    "Size-related return rate (%)",
]

NUMERIC_COLUMNS = [
    "NMV",
    "Sold articles",
    "Returned articles",
    "Return rate (%)",
    "Estimated return rate (%)",
    "Size-related return rate (%)",
    "Days online",
]


def reason_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith(REASON_PREFIX)]


def clean_reason_name(column: str) -> str:
    return column.replace(REASON_PREFIX, "").replace(REASON_SUFFIX, "")


def validate_returns_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues = []
    total_rows = len(df)

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            issues.append(
                {
                    "severity": "Error",
                    "area": "Schema",
                    "check": f"Missing required column: {column}",
                    "rows": total_rows,
                    "share": 100.0 if total_rows else 0.0,
                    "recommendation": "Add this column to the CSV file before analysis.",
                }
            )

    missing_optional_columns = set(df.attrs.get("missing_optional_columns", []))
    for column in OPTIONAL_ANALYSIS_COLUMNS:
        if column not in df.columns or column in missing_optional_columns:
            issues.append(
                {
                    "severity": "Info",
                    "area": "Schema",
                    "check": f"Missing optional column: {column}",
                    "rows": total_rows,
                    "share": 100.0 if total_rows else 0.0,
                    "recommendation": "The app will work, but some dimensions or reports will be less complete.",
                }
            )

    reason_cols = reason_columns(df)
    if not reason_cols:
        issues.append(
            {
                "severity": "Warning",
                "area": "Schema",
                "check": "No return reason columns",
                "rows": total_rows,
                "share": 100.0 if total_rows else 0.0,
                "recommendation": "Add 'Return reason: ... (%)' columns to analyze return reasons.",
            }
        )

    for column in ["Sold articles", "Returned articles", "Return rate (%)"]:
        if column in df.columns:
            missing = int(df[column].isna().sum())
            if missing:
                issues.append(
                    {
                        "severity": "Warning",
                        "area": "Values",
                        "check": f"Missing/non-numeric values in {column}",
                        "rows": missing,
                        "share": 100 * missing / total_rows if total_rows else 0.0,
                        "recommendation": "Check the decimal separator, blank fields, and text values.",
                    }
                )

    if {"Sold articles", "Returned articles"}.issubset(df.columns):
        negative = (df["Sold articles"].fillna(0) < 0) | (df["Returned articles"].fillna(0) < 0)
        returned_over_sold = df["Returned articles"].fillna(0) > df["Sold articles"].fillna(0)
        for label, mask, recommendation in [
            ("Negative sold/returned values", negative, "Remove negative values or fix the source export."),
            (
                "Returned articles greater than sold articles",
                returned_over_sold,
                "Verify aggregation, reporting period, or column mapping.",
            ),
        ]:
            count = int(mask.sum())
            if count:
                issues.append(
                    {
                        "severity": "Error" if label.startswith("Negative") else "Warning",
                        "area": "Values",
                        "check": label,
                        "rows": count,
                        "share": 100 * count / total_rows if total_rows else 0.0,
                        "recommendation": recommendation,
                    }
                )

    if reason_cols:
        reason_sum = df[reason_cols].fillna(0).sum(axis=1)
        reason_outside_range = (reason_sum < 95) | (reason_sum > 105)
        count = int(reason_outside_range.sum())
        if count:
            issues.append(
                {
                    "severity": "Warning",
                    "area": "Return reasons",
                    "check": "Return reason percentages do not sum to about 100%",
                    "rows": count,
                    "share": 100 * count / total_rows if total_rows else 0.0,
                    "recommendation": "Check whether the export includes all return reasons and the correct separator.",
                }
            )

    severity_order = {"Error": 0, "Warning": 1, "Info": 2}
    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        issues_df = pd.DataFrame(
            columns=["severity", "area", "check", "rows", "share", "recommendation"]
        )
    else:
        issues_df["_order"] = issues_df["severity"].map(severity_order).fillna(9)
        issues_df = issues_df.sort_values(["_order", "rows"], ascending=[True, False]).drop(columns="_order")

    summary = (
        issues_df.groupby("severity", dropna=False)
        .agg(checks=("check", "count"), impacted_rows=("rows", "sum"))
        .reset_index()
        if not issues_df.empty
        else pd.DataFrame(columns=["severity", "checks", "impacted_rows"])
    )
    return summary, issues_df


def _path_cache_key(path: str | Path) -> str:
    source_path = Path(path).expanduser()
    stat = source_path.stat()
    resolved = source_path.resolve(strict=False)
    payload = f"{CACHE_SCHEMA_VERSION}|{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def parquet_cache_path(path: str | Path) -> Path:
    return CACHE_DIR / f"{_path_cache_key(path)}.parquet"


def load_returns_path_with_parquet(path: str | Path) -> pd.DataFrame:
    cache_file = parquet_cache_path(path)
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            cache_file.unlink(missing_ok=True)

    df = load_returns_csv(path)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp_file = cache_file.with_suffix(".tmp.parquet")
        df.to_parquet(temp_file, index=False)
        temp_file.replace(cache_file)
    except Exception:
        temp_file = cache_file.with_suffix(".tmp.parquet")
        temp_file.unlink(missing_ok=True)

    return df


def load_returns_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    df = pd.read_csv(source, sep=";", dtype=str, keep_default_na=False)
    df.columns = [column.strip() for column in df.columns]

    missing_required = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_required:
        missing = ", ".join(missing_required)
        raise ValueError(f"Missing required columns: {missing}")

    missing_optional = [column for column in OPTIONAL_ANALYSIS_COLUMNS if column not in df.columns]
    for column in missing_optional:
        df[column] = 0 if column in {"NMV", "Size-related return rate (%)"} else "Unknown"
    if missing_optional:
        df.attrs["missing_optional_columns"] = missing_optional

    for column in NUMERIC_COLUMNS + reason_columns(df):
        if column in df.columns:
            df[column] = (
                df[column]
                .replace({"": np.nan, "N/A": np.nan, "n/a": np.nan})
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Date first on offer" in df.columns:
        df["Date first on offer"] = pd.to_datetime(
            df["Date first on offer"], format="%d.%m.%Y", errors="coerce"
        )

    for column in reason_columns(df):
        estimated_column = f"Estimated returns - {clean_reason_name(column)}"
        df[estimated_column] = df["Returned articles"].fillna(0) * df[column].fillna(0) / 100

    if {"Sold articles", "Returned articles"}.issubset(df.columns):
        df["Weighted return rate (%)"] = np.where(
            df["Sold articles"].fillna(0) > 0,
            100 * df["Returned articles"].fillna(0) / df["Sold articles"].fillna(0),
            np.nan,
        )

    if {"Return reason: Item is too big (%)", "Return reason: Item is too small (%)"}.issubset(
        df.columns
    ):
        df["Size reason share (%)"] = (
            df["Return reason: Item is too big (%)"].fillna(0)
            + df["Return reason: Item is too small (%)"].fillna(0)
        )

    text_columns = df.select_dtypes(include="object").columns
    for column in text_columns:
        df[column] = df[column].replace("", "Unknown")

    return df
