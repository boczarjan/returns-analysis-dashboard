from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import (
    clean_reason_name,
    disambiguate_na_article_variants,
    estimated_reason_column,
    find_reason_column,
    reason_columns,
)


def weighted_return_rate(df: pd.DataFrame) -> float:
    sold = df["Sold articles"].sum()
    if sold <= 0:
        return 0.0
    return 100 * df["Returned articles"].sum() / sold


def kpi_summary(df: pd.DataFrame) -> dict[str, float]:
    df = disambiguate_na_article_variants(df)
    sold = float(df["Sold articles"].sum())
    returned = float(df["Returned articles"].sum())
    nmv = float(df["NMV"].sum()) if "NMV" in df.columns else 0.0
    avg_net_price = nmv / sold if sold else 0.0
    estimated_returned_nmv = avg_net_price * returned
    average_return_rate = (
        float(df["Return rate (%)"].mean(skipna=True)) if "Return rate (%)" in df.columns else 0.0
    )
    return {
        "sold": sold,
        "returned": returned,
        "return_rate": 100 * returned / sold if sold else 0.0,
        "average_return_rate": average_return_rate,
        "nmv": nmv,
        "avg_net_price": avg_net_price,
        "estimated_returned_nmv": estimated_returned_nmv,
        "variants": float(df["Article variant"].nunique()) if "Article variant" in df.columns else 0,
    }


def aggregate_by(df: pd.DataFrame, group_cols: str | list[str]) -> pd.DataFrame:
    df = disambiguate_na_article_variants(df)
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
    grouped["avg_net_price"] = np.where(grouped["sold"] > 0, grouped["nmv"] / grouped["sold"], 0)
    grouped["estimated_returned_nmv"] = grouped["avg_net_price"] * grouped["returned"]
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
    df = disambiguate_na_article_variants(df)
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
    if "Category" in df.columns and "Category" in grouped.columns:
        category_prices = aggregate_by(df, "Category")[["Category", "avg_net_price"]].rename(
            columns={"avg_net_price": "category_avg_net_price"}
        )
        grouped = grouped.merge(category_prices, on="Category", how="left")
    else:
        grouped["category_avg_net_price"] = 0.0
    grouped["price_gap_vs_category"] = grouped["avg_net_price"] - grouped["category_avg_net_price"].fillna(0)
    grouped["price_index_vs_category"] = np.where(
        grouped["category_avg_net_price"].fillna(0) > 0,
        100 * grouped["avg_net_price"] / grouped["category_avg_net_price"],
        0.0,
    )
    return grouped.sort_values(["returned", "return_rate"], ascending=[False, False])


def product_reason_ranking(
    df: pd.DataFrame,
    reason: str,
    min_sold: int = 20,
    min_returned: int = 1,
) -> pd.DataFrame:
    df = disambiguate_na_article_variants(df)
    columns = [
        "Article variant",
        "Zalando article variant",
        "Category",
        "Article type",
        "sold",
        "returned",
        "return_rate",
        "category_return_rate",
        "dataset_return_rate",
        "return_rate_gap_vs_category",
        "return_rate_gap_vs_dataset",
        "avg_net_price",
        "category_avg_net_price",
        "price_gap_vs_category",
        "price_index_vs_category",
        "estimated_returned_nmv",
        "selected_reason_returns",
        "reason_returned_nmv",
        "reason_share_of_returns",
        "category_reason_share",
        "dataset_reason_share",
        "avg_product_reason_share",
        "reason_gap_vs_category",
        "reason_gap_vs_dataset",
        "reason_gap_vs_product_average",
        "selected_reason_return_rate",
        "category_selected_reason_return_rate",
        "dataset_selected_reason_return_rate",
    ]
    reason_column = find_reason_column(df, reason)
    if reason_column is None:
        return pd.DataFrame(columns=columns)

    products = product_ranking(df, min_sold=min_sold).copy()
    if products.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    estimated_column = estimated_reason_column(clean_reason_name(reason_column))
    if estimated_column in work.columns:
        work["_selected_reason_returns"] = work[estimated_column].fillna(0)
    else:
        work["_selected_reason_returns"] = (
            work["Returned articles"].fillna(0) * work[reason_column].fillna(0) / 100
        )

    reason_by_variant = work.groupby("Article variant", dropna=False)["_selected_reason_returns"].sum()
    products["selected_reason_returns"] = (
        products["Article variant"].map(reason_by_variant).fillna(0).astype(float)
    )
    products = products[products["returned"] >= min_returned].copy()
    if products.empty:
        return pd.DataFrame(columns=columns)

    products["reason_share_of_returns"] = np.where(
        products["returned"] > 0,
        100 * products["selected_reason_returns"] / products["returned"],
        0.0,
    )
    products["selected_reason_return_rate"] = np.where(
        products["sold"] > 0,
        100 * products["selected_reason_returns"] / products["sold"],
        0.0,
    )
    products["reason_returned_nmv"] = products["avg_net_price"] * products["selected_reason_returns"]

    dataset_sold = work["Sold articles"].sum()
    dataset_returned = work["Returned articles"].sum()
    dataset_reason_returns = work["_selected_reason_returns"].sum()
    dataset_return_rate = 100 * dataset_returned / dataset_sold if dataset_sold else 0.0
    dataset_reason_share = 100 * dataset_reason_returns / dataset_returned if dataset_returned else 0.0
    dataset_reason_return_rate = 100 * dataset_reason_returns / dataset_sold if dataset_sold else 0.0

    if "Category" in work.columns and "Category" in products.columns:
        category_benchmarks = (
            work.groupby("Category", dropna=False)
            .agg(
                category_sold=("Sold articles", "sum"),
                category_returned=("Returned articles", "sum"),
                category_reason_returns=("_selected_reason_returns", "sum"),
            )
            .reset_index()
        )
        category_benchmarks["category_return_rate"] = np.where(
            category_benchmarks["category_sold"] > 0,
            100 * category_benchmarks["category_returned"] / category_benchmarks["category_sold"],
            0.0,
        )
        category_benchmarks["category_reason_share"] = np.where(
            category_benchmarks["category_returned"] > 0,
            100 * category_benchmarks["category_reason_returns"] / category_benchmarks["category_returned"],
            0.0,
        )
        category_benchmarks["category_selected_reason_return_rate"] = np.where(
            category_benchmarks["category_sold"] > 0,
            100 * category_benchmarks["category_reason_returns"] / category_benchmarks["category_sold"],
            0.0,
        )
        products = products.merge(
            category_benchmarks[
                [
                    "Category",
                    "category_return_rate",
                    "category_reason_share",
                    "category_selected_reason_return_rate",
                ]
            ],
            on="Category",
            how="left",
        )
    else:
        products["category_return_rate"] = 0.0
        products["category_reason_share"] = 0.0
        products["category_selected_reason_return_rate"] = 0.0

    avg_product_reason_share = products["reason_share_of_returns"].mean()
    products["dataset_return_rate"] = dataset_return_rate
    products["dataset_reason_share"] = dataset_reason_share
    products["dataset_selected_reason_return_rate"] = dataset_reason_return_rate
    products["avg_product_reason_share"] = avg_product_reason_share if not np.isnan(avg_product_reason_share) else 0.0
    products["reason_gap_vs_category"] = (
        products["reason_share_of_returns"] - products["category_reason_share"].fillna(0)
    )
    products["reason_gap_vs_dataset"] = products["reason_share_of_returns"] - dataset_reason_share
    products["reason_gap_vs_product_average"] = (
        products["reason_share_of_returns"] - products["avg_product_reason_share"]
    )
    products["return_rate_gap_vs_category"] = (
        products["return_rate"] - products["category_return_rate"].fillna(0)
    )
    products["return_rate_gap_vs_dataset"] = products["return_rate"] - dataset_return_rate

    available_columns = [column for column in columns if column in products.columns]
    return products[available_columns].sort_values(
        ["reason_share_of_returns", "selected_reason_returns"],
        ascending=[False, False],
    )
