from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


DEFAULT_CSV_PATH = Path(os.environ.get("RETURNS_CSV_PATH", "data/returns.csv"))

ARTICLE_VARIANT_COLUMN = "Article variant"
ZALANDO_ARTICLE_VARIANT_COLUMN = "Zalando article variant"
NA_ARTICLE_VARIANT = "N/A"
REASON_PREFIX = "Return reason: "
REASON_SUFFIX = " (%)"
ESTIMATED_REASON_PREFIX = "Estimated returns - "
NO_DETAILS_REASON = "No details"
CACHE_SCHEMA_VERSION = "v4"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".returns_cache"

REQUIRED_COLUMNS = [
    ARTICLE_VARIANT_COLUMN,
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
    ZALANDO_ARTICLE_VARIANT_COLUMN,
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


def estimated_reason_column(reason: str) -> str:
    return f"{ESTIMATED_REASON_PREFIX}{reason}"


def find_reason_column(df: pd.DataFrame, reason: str) -> str | None:
    wanted = str(reason).strip().casefold()
    for column in reason_columns(df):
        if clean_reason_name(column).strip().casefold() == wanted:
            return column
    return None


def has_return_reason(df: pd.DataFrame, reason: str) -> bool:
    return find_reason_column(df, reason) is not None


def is_na_article_variant(value: object) -> bool:
    return str(value).strip().casefold() == NA_ARTICLE_VARIANT.casefold()


def disambiguate_na_article_variants(df: pd.DataFrame) -> pd.DataFrame:
    if ARTICLE_VARIANT_COLUMN not in df.columns:
        return df

    na_mask = df[ARTICLE_VARIANT_COLUMN].map(is_na_article_variant)
    if not na_mask.any():
        return df

    result = df.copy()
    fallback = pd.Series([f"row {index + 1}" for index in range(len(result))], index=result.index)
    if ZALANDO_ARTICLE_VARIANT_COLUMN in result.columns:
        suffix = result[ZALANDO_ARTICLE_VARIANT_COLUMN].astype(str).str.strip()
        suffix = suffix.where(suffix.ne("") & suffix.str.casefold().ne("unknown"), fallback)
    else:
        suffix = fallback

    result.loc[na_mask, ARTICLE_VARIANT_COLUMN] = NA_ARTICLE_VARIANT + " (" + suffix[na_mask] + ")"
    return result


def exclude_return_reason(df: pd.DataFrame, reason: str = NO_DETAILS_REASON) -> pd.DataFrame:
    reason_column = find_reason_column(df, reason)
    if reason_column is None:
        return df.copy()

    excluded_reason = clean_reason_name(reason_column)
    result = df.copy()
    result.attrs.update(df.attrs)

    old_returned = result["Returned articles"].fillna(0)
    excluded_pct = result[reason_column].fillna(0).clip(lower=0, upper=100)
    excluded_returns = old_returned * excluded_pct / 100
    result["Returned articles"] = (old_returned - excluded_returns).clip(lower=0)

    remaining_reason_columns = [column for column in reason_columns(result) if column != reason_column]
    remaining_pct_sum = result[remaining_reason_columns].fillna(0).sum(axis=1) if remaining_reason_columns else 0
    for column in remaining_reason_columns:
        result[column] = np.where(
            remaining_pct_sum > 0,
            100 * result[column].fillna(0) / remaining_pct_sum,
            0.0,
        )

    result = result.drop(
        columns=[reason_column, estimated_reason_column(excluded_reason)],
        errors="ignore",
    )

    for column in remaining_reason_columns:
        if column in result.columns:
            result[estimated_reason_column(clean_reason_name(column))] = (
                result["Returned articles"].fillna(0) * result[column].fillna(0) / 100
            )

    if "Sold articles" in result.columns:
        result["Return rate (%)"] = np.where(
            result["Sold articles"].fillna(0) > 0,
            100 * result["Returned articles"].fillna(0) / result["Sold articles"].fillna(0),
            np.nan,
        )
        if "Weighted return rate (%)" in result.columns:
            result["Weighted return rate (%)"] = result["Return rate (%)"]
        if "Estimated return rate (%)" in result.columns:
            result["Estimated return rate (%)"] = result["Return rate (%)"]

    too_big_column = "Return reason: Item is too big (%)"
    too_small_column = "Return reason: Item is too small (%)"
    if {too_big_column, too_small_column}.issubset(result.columns):
        result["Size reason share (%)"] = (
            result[too_big_column].fillna(0) + result[too_small_column].fillna(0)
        )

    too_big_estimate = estimated_reason_column("Item is too big")
    too_small_estimate = estimated_reason_column("Item is too small")
    if {"Sold articles", too_big_estimate, too_small_estimate}.issubset(result.columns):
        size_returns = result[too_big_estimate].fillna(0) + result[too_small_estimate].fillna(0)
        result["Size-related return rate (%)"] = np.where(
            result["Sold articles"].fillna(0) > 0,
            100 * size_returns / result["Sold articles"].fillna(0),
            np.nan,
        )

    result.attrs["excluded_return_reason"] = excluded_reason
    result.attrs["excluded_return_reason_returns"] = float(excluded_returns.sum())
    return result


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
        returned = df["Returned articles"].fillna(0) if "Returned articles" in df.columns else pd.Series(0, index=df.index)
        reason_outside_range = ((reason_sum < 95) | (reason_sum > 105)) & (returned > 0)
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


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def load_returns_path_with_parquet(path: str | Path) -> pd.DataFrame:
    cache_file = parquet_cache_path(path)
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            _safe_unlink(cache_file)

    df = load_returns_csv(path)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
    except Exception:
        pass

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

    df = disambiguate_na_article_variants(df)

    return df
