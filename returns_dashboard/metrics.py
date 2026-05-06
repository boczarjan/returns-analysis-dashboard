from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import clean_reason_name, reason_columns


def weighted_return_rate(df: pd.DataFrame) -> float:
    sold = df["Sold articles"].sum()
    if sold <= 0:
        return 0.0
    return 100 * df["Returned articles"].sum() / sold


def kpi_summary(df: pd.DataFrame) -> dict[str, float]:
    sold = float(df["Sold articles"].sum())
    returned = float(df["Returned articles"].sum())
    nmv = float(df["NMV"].sum()) if "NMV" in df.columns else 0.0
    average_return_rate = (
        float(df["Return rate (%)"].mean(skipna=True)) if "Return rate (%)" in df.columns else 0.0
    )
    return {
        "sold": sold,
        "returned": returned,
        "return_rate": 100 * returned / sold if sold else 0.0,
        "average_return_rate": average_return_rate,
        "nmv": nmv,
        "variants": float(df["Article variant"].nunique()) if "Article variant" in df.columns else 0,
    }


def aggregate_by(df: pd.DataFrame, group_cols: str | list[str]) -> pd.DataFrame:
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            sold=("Sold articles", "sum"),
            returned=("Returned articles", "sum"),
            nmv=("NMV", "sum"),
            variants=("Article variant", "nunique"),
        )
        .reset_index()
    )
    grouped["return_rate"] = np.where(grouped["sold"] > 0, 100 * grouped["returned"] / grouped["sold"], 0)
    grouped["return_share"] = np.where(
        grouped["returned"].sum() > 0,
        100 * grouped["returned"] / grouped["returned"].sum(),
        0,
    )
    return grouped.sort_values("returned", ascending=False)


def reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_returned = df["Returned articles"].sum()
    for column in reason_columns(df):
        reason = clean_reason_name(column)
        estimated_returns = (df["Returned articles"].fillna(0) * df[column].fillna(0) / 100).sum()
        rows.append(
            {
                "reason": reason,
                "estimated_returns": estimated_returns,
                "share_of_returns": 100 * estimated_returns / total_returned if total_returned else 0,
                "avg_reason_pct": df[column].mean(skipna=True),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["reason", "estimated_returns", "share_of_returns", "avg_reason_pct"])
    return pd.DataFrame(rows).sort_values("estimated_returns", ascending=False)


def reason_by_dimension(df: pd.DataFrame, dimension: str, min_returned: int = 1) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(dimension, dropna=False):
        returned = group["Returned articles"].sum()
        if returned < min_returned:
            continue
        for column in reason_columns(df):
            reason = clean_reason_name(column)
            estimated_returns = (group["Returned articles"].fillna(0) * group[column].fillna(0) / 100).sum()
            rows.append(
                {
                    dimension: value,
                    "reason": reason,
                    "estimated_returns": estimated_returns,
                    "returned": returned,
                    "reason_share": 100 * estimated_returns / returned if returned else 0,
                }
            )
    if not rows:
        return pd.DataFrame(columns=[dimension, "reason", "estimated_returns", "returned", "reason_share"])
    return pd.DataFrame(rows)


def product_ranking(df: pd.DataFrame, min_sold: int = 20) -> pd.DataFrame:
    group_cols = [
        "Article variant",
        "Zalando article variant",
        "Category",
        "Article type",
        "Gender",
        "Season",
    ]
    existing_group_cols = [column for column in group_cols if column in df.columns]
    grouped = aggregate_by(df, existing_group_cols)
    grouped = grouped[grouped["sold"] >= min_sold].copy()
    grouped["return_gap_vs_dataset"] = grouped["return_rate"] - weighted_return_rate(df)
    return grouped.sort_values(["returned", "return_rate"], ascending=[False, False])
