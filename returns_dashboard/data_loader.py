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
CACHE_SCHEMA_VERSION = "v2"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".returns_cache"

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
