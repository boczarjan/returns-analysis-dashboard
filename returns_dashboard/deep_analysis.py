from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import clean_reason_name, reason_columns
from .metrics import aggregate_by, product_ranking, reason_summary, weighted_return_rate


SIZE_TOO_BIG = "Estimated returns - Item is too big"
SIZE_TOO_SMALL = "Estimated returns - Item is too small"


def _reason_estimate_columns(df: pd.DataFrame) -> list[str]:
    return [f"Estimated returns - {clean_reason_name(column)}" for column in reason_columns(df)]


def _dominant_reason(group: pd.DataFrame) -> tuple[str, float, float]:
    reason_totals = {}
    for column in reason_columns(group):
        reason = clean_reason_name(column)
        reason_totals[reason] = (group["Returned articles"].fillna(0) * group[column].fillna(0) / 100).sum()
    if not reason_totals:
        return "Unknown", 0.0, 0.0
    reason, estimated_returns = max(reason_totals.items(), key=lambda item: item[1])
    returned = group["Returned articles"].sum()
    return reason, float(estimated_returns), 100 * estimated_returns / returned if returned else 0.0


def recommendation_for_reason(reason: str, size_balance: float = 0.0) -> str:
    reason_lower = reason.lower()
    if "too small" in reason_lower:
        return "Check the size chart, fit communication, and feedback about undersized fit."
    if "too big" in reason_lower:
        return "Check the fit description, model photos, and risk of oversized fit."
    if "described" in reason_lower:
        return "Verify the description, photos, material, color, and expectations set on the product page."
    if "damaged" in reason_lower:
        return "Check quality control, packaging, and whether the issue repeats by supplier."
    if "wrong item" in reason_lower:
        return "Check variant mapping, picking, and product code consistency."
    if "too expensive" in reason_lower:
        return "Compare the price with the category and check whether the value proposition is clear."
    if "late" in reason_lower:
        return "Check fulfillment SLA and the countries where delays occur most often."
    if "no details" in reason_lower:
        return "Treat this as a data gap: look for an additional pattern by country, category, and sizing."
    if "don't like" in reason_lower or "like" in reason_lower:
        return "Review styling, photos, quality expectations, and product fit with the category."
    if size_balance > 25:
        return "The issue skews toward too-small sizing; start with fit communication."
    if size_balance < -25:
        return "The issue skews toward too-big sizing; start with fit communication."
    return "Review the product in the detailed profile and compare it with the category and markets."


def action_list(df: pd.DataFrame, min_sold: int = 30) -> pd.DataFrame:
    products = product_ranking(df, min_sold=min_sold).copy()
    if products.empty:
        return products

    dataset_rr = weighted_return_rate(df)
    category_rr = aggregate_by(df, "Category")[["Category", "return_rate"]].rename(
        columns={"return_rate": "category_return_rate"}
    )
    type_rr = aggregate_by(df, "Article type")[["Article type", "return_rate"]].rename(
        columns={"return_rate": "type_return_rate"}
    )
    products = products.merge(category_rr, on="Category", how="left").merge(type_rr, on="Article type", how="left")

    variant_group = df.groupby("Article variant", dropna=False)
    volume = variant_group.agg(
        sold_raw=("Sold articles", "sum"),
        returned_raw=("Returned articles", "sum"),
    )

    estimate_columns = [column for column in _reason_estimate_columns(df) if column in df.columns]
    if estimate_columns:
        reason_totals = variant_group[estimate_columns].sum()
        dominant_estimated = reason_totals.max(axis=1)
        dominant_reason = reason_totals.idxmax(axis=1).str.replace("Estimated returns - ", "", regex=False)
    else:
        dominant_estimated = pd.Series(0.0, index=volume.index)
        dominant_reason = pd.Series("Unknown", index=volume.index)

    too_big = variant_group[SIZE_TOO_BIG].sum() if SIZE_TOO_BIG in df.columns else pd.Series(0.0, index=volume.index)
    too_small = variant_group[SIZE_TOO_SMALL].sum() if SIZE_TOO_SMALL in df.columns else pd.Series(0.0, index=volume.index)
    size_balance = np.where(
        volume["returned_raw"] > 0,
        100 * (too_small.reindex(volume.index).fillna(0) - too_big.reindex(volume.index).fillna(0)) / volume["returned_raw"],
        0.0,
    )

    status_columns = [
        column
        for column in ["Estimated return rate status", "Size-related return rate status"]
        if column in df.columns
    ]
    if status_columns:
        unstable_mask = (
            df[status_columns]
            .astype(str)
            .apply(lambda column: column.str.contains("unstable", case=False, na=False))
            .any(axis=1)
        )
        unstable = unstable_mask.groupby(df["Article variant"], dropna=False).max().reindex(volume.index).fillna(False)
    else:
        unstable = pd.Series(False, index=volume.index)

    details = volume.reset_index()
    details["dominant_reason"] = dominant_reason.reindex(volume.index).fillna("Unknown").values
    details["dominant_reason_returns"] = dominant_estimated.reindex(volume.index).fillna(0).values
    details["dominant_reason_share"] = np.where(
        details["returned_raw"] > 0,
        100 * details["dominant_reason_returns"] / details["returned_raw"],
        0.0,
    )
    details["size_balance"] = size_balance
    details["confidence_flag"] = np.where(
        (details["sold_raw"] < 50) | unstable.reindex(volume.index).fillna(False).to_numpy(),
        "Low volume/status unstable",
        "OK",
    )
    details["recommended_action"] = [
        recommendation_for_reason(reason, balance)
        for reason, balance in zip(details["dominant_reason"], details["size_balance"], strict=False)
    ]

    products = products.merge(
        details[
            [
                "Article variant",
                "dominant_reason",
                "dominant_reason_returns",
                "dominant_reason_share",
                "size_balance",
                "confidence_flag",
                "recommended_action",
            ]
        ],
        on="Article variant",
        how="left",
    )
    products["gap_vs_dataset"] = products["return_rate"] - dataset_rr
    products["gap_vs_category"] = products["return_rate"] - products["category_return_rate"]
    products["gap_vs_type"] = products["return_rate"] - products["type_return_rate"]
    products["excess_returns_vs_dataset"] = np.maximum(
        products["returned"] - products["sold"] * dataset_rr / 100,
        0,
    )
    products["priority_score"] = (
        products["excess_returns_vs_dataset"] * 1.4
        + products["dominant_reason_returns"].fillna(0) * 0.8
        + np.maximum(products["gap_vs_category"].fillna(0), 0) * 2
    )
    columns = [
        "Article variant",
        "Zalando article variant",
        "Category",
        "Article type",
        "sold",
        "returned",
        "return_rate",
        "category_return_rate",
        "gap_vs_category",
        "gap_vs_dataset",
        "gap_vs_type",
        "excess_returns_vs_dataset",
        "dominant_reason",
        "dominant_reason_returns",
        "dominant_reason_share",
        "size_balance",
        "priority_score",
        "confidence_flag",
        "recommended_action",
    ]
    return products[columns].sort_values("priority_score", ascending=False)


def benchmark_products(df: pd.DataFrame, min_sold: int = 30) -> pd.DataFrame:
    products = product_ranking(df, min_sold=min_sold).copy()
    if products.empty:
        return products
    dataset_rr = weighted_return_rate(df)
    category_rr = aggregate_by(df, "Category")[["Category", "return_rate"]].rename(
        columns={"return_rate": "category_return_rate"}
    )
    type_rr = aggregate_by(df, "Article type")[["Article type", "return_rate"]].rename(
        columns={"return_rate": "type_return_rate"}
    )
    products = products.merge(category_rr, on="Category", how="left").merge(type_rr, on="Article type", how="left")
    products["dataset_return_rate"] = dataset_rr
    products["gap_vs_dataset"] = products["return_rate"] - dataset_rr
    products["gap_vs_category"] = products["return_rate"] - products["category_return_rate"]
    products["gap_vs_type"] = products["return_rate"] - products["type_return_rate"]
    return products.sort_values("gap_vs_category", ascending=False)


def detect_product_anomalies(df: pd.DataFrame, min_sold: int = 30) -> pd.DataFrame:
    products = benchmark_products(df, min_sold=min_sold).copy()
    if products.empty:
        return products

    dataset_rr = weighted_return_rate(df)
    returned_q3 = products["returned"].quantile(0.75)
    gap_q3 = products["gap_vs_category"].clip(lower=0).quantile(0.75)
    high_gap = products["gap_vs_category"].fillna(0) >= max(gap_q3, 5)
    high_volume = products["returned"].fillna(0) >= returned_q3
    very_high_vs_dataset = products["gap_vs_dataset"].fillna(0) >= 10

    reason_map = {}
    reason_share_map = {}
    for variant, group in df.groupby("Article variant", dropna=False):
        reason, _, share = _dominant_reason(group)
        reason_map[variant] = reason
        reason_share_map[variant] = share

    products["dominant_reason"] = products["Article variant"].map(reason_map).fillna("Unknown")
    products["dominant_reason_share"] = products["Article variant"].map(reason_share_map).fillna(0.0)
    concentrated_reason = products["dominant_reason_share"] >= 55

    status_columns = [
        column
        for column in ["Estimated return rate status", "Size-related return rate status"]
        if column in df.columns
    ]
    if status_columns:
        unstable_mask = (
            df[status_columns]
            .astype(str)
            .apply(lambda column: column.str.contains("unstable", case=False, na=False))
            .any(axis=1)
        )
        unstable = unstable_mask.groupby(df["Article variant"], dropna=False).max()
        products["unstable_status"] = products["Article variant"].map(unstable).fillna(False)
    else:
        products["unstable_status"] = False

    products["anomaly_flags"] = [
        ", ".join(
            flag
            for flag, active in [
                ("high gap vs category", gap),
                ("high return volume", volume),
                ("very high vs dataset", dataset),
                ("dominant reason concentration", reason),
                ("unstable data status", unstable_flag),
            ]
            if active
        )
        for gap, volume, dataset, reason, unstable_flag in zip(
            high_gap,
            high_volume,
            very_high_vs_dataset,
            concentrated_reason,
            products["unstable_status"],
            strict=False,
        )
    ]
    products["anomaly_score"] = (
        np.maximum(products["gap_vs_category"].fillna(0), 0) * 2.0
        + np.maximum(products["gap_vs_dataset"].fillna(0), 0) * 1.2
        + products["returned"].fillna(0) * 0.35
        + np.where(concentrated_reason, 20, 0)
        - np.where(products["unstable_status"], 10, 0)
    )
    products = products[products["anomaly_flags"].astype(bool)].copy()
    return products.sort_values("anomaly_score", ascending=False)


def _reason_gap_vs_category(df: pd.DataFrame, product_df: pd.DataFrame, category: str) -> pd.DataFrame:
    product_reasons = reason_summary(product_df)
    if not category or "Category" not in df.columns:
        product_reasons["category_share_of_returns"] = 0.0
        product_reasons["gap_vs_category"] = product_reasons["share_of_returns"]
        return product_reasons

    category_df = df[df["Category"].eq(category)]
    category_reasons = reason_summary(category_df)
    if category_reasons.empty:
        product_reasons["category_share_of_returns"] = 0.0
        product_reasons["gap_vs_category"] = product_reasons["share_of_returns"]
        return product_reasons

    category_reasons = category_reasons[["reason", "share_of_returns"]].rename(
        columns={"share_of_returns": "category_share_of_returns"}
    )
    comparison = product_reasons.merge(category_reasons, on="reason", how="left")
    comparison["category_share_of_returns"] = comparison["category_share_of_returns"].fillna(0.0)
    comparison["gap_vs_category"] = comparison["share_of_returns"] - comparison["category_share_of_returns"]
    return comparison.sort_values("gap_vs_category", ascending=False)


def _similar_better_products(df: pd.DataFrame, article_variant: str, category: str, article_type: str) -> pd.DataFrame:
    ranking = product_ranking(df, min_sold=10).copy()
    if ranking.empty:
        return ranking
    target = ranking[ranking["Article variant"].astype(str).eq(str(article_variant))]
    if target.empty:
        return ranking.iloc[0:0]

    target_row = target.iloc[0]
    peers = ranking[~ranking["Article variant"].astype(str).eq(str(article_variant))].copy()
    if category and "Category" in peers.columns:
        peers = peers[peers["Category"].eq(category)]
    if article_type and "Article type" in peers.columns:
        same_type = peers[peers["Article type"].eq(article_type)]
        if not same_type.empty:
            peers = same_type

    peers = peers[peers["return_rate"] < target_row["return_rate"]].copy()
    if peers.empty:
        return peers
    peers["return_rate_advantage"] = target_row["return_rate"] - peers["return_rate"]
    peers["returned_delta_if_like_peer"] = np.maximum(
        target_row["returned"] - target_row["sold"] * peers["return_rate"] / 100,
        0,
    )
    return peers.sort_values(["return_rate_advantage", "sold"], ascending=[False, False]).head(10)


def product_profile(df: pd.DataFrame, article_variant: str) -> dict[str, pd.DataFrame | dict[str, float | str]]:
    product_df = df[df["Article variant"].astype(str).eq(str(article_variant))].copy()
    if product_df.empty:
        return {}

    summary = aggregate_by(product_df, "Article variant").iloc[0].to_dict()
    category = product_df["Category"].mode().iat[0] if "Category" in product_df and not product_df["Category"].mode().empty else ""
    article_type = (
        product_df["Article type"].mode().iat[0]
        if "Article type" in product_df and not product_df["Article type"].mode().empty
        else ""
    )
    dataset_rr = weighted_return_rate(df)
    category_rr = weighted_return_rate(df[df["Category"].eq(category)]) if category else 0.0
    type_rr = weighted_return_rate(df[df["Article type"].eq(article_type)]) if article_type else 0.0
    reason, dominant_estimated, dominant_share = _dominant_reason(product_df)
    too_big = product_df[SIZE_TOO_BIG].sum() if SIZE_TOO_BIG in product_df.columns else 0.0
    too_small = product_df[SIZE_TOO_SMALL].sum() if SIZE_TOO_SMALL in product_df.columns else 0.0
    returned = product_df["Returned articles"].sum()
    size_balance = 100 * (too_small - too_big) / returned if returned else 0.0

    summary.update(
        {
            "category": category,
            "article_type": article_type,
            "dataset_return_rate": dataset_rr,
            "category_return_rate": category_rr,
            "type_return_rate": type_rr,
            "dominant_reason": reason,
            "dominant_reason_returns": dominant_estimated,
            "dominant_reason_share": dominant_share,
            "size_balance": size_balance,
            "recommended_action": recommendation_for_reason(reason, size_balance),
        }
    )

    return {
        "summary": summary,
        "reasons": reason_summary(product_df),
        "reason_gap_vs_category": _reason_gap_vs_category(df, product_df, category),
        "similar_products": _similar_better_products(df, article_variant, category, article_type),
        "countries": aggregate_by(product_df, "Country"),
        "seasons": aggregate_by(product_df, "Season"),
        "raw": product_df,
    }


def product_segments(df: pd.DataFrame, min_sold: int = 10) -> pd.DataFrame:
    products = product_ranking(df, min_sold=min_sold).copy()
    if products.empty:
        return products
    sold_threshold = products["sold"].median()
    rr_threshold = weighted_return_rate(df)
    conditions = [
        (products["sold"] >= sold_threshold) & (products["return_rate"] >= rr_threshold),
        (products["sold"] >= sold_threshold) & (products["return_rate"] < rr_threshold),
        (products["sold"] < sold_threshold) & (products["return_rate"] >= rr_threshold),
        (products["sold"] < sold_threshold) & (products["return_rate"] < rr_threshold),
    ]
    labels = [
        "High volume / high returns",
        "High volume / low returns",
        "Low volume / high returns",
        "Low volume / low returns",
    ]
    products["segment"] = np.select(conditions, labels, default="Other")
    products["sold_threshold"] = sold_threshold
    products["return_rate_threshold"] = rr_threshold
    return products


def pareto_products(df: pd.DataFrame) -> pd.DataFrame:
    products = aggregate_by(df, "Article variant").sort_values("returned", ascending=False).reset_index(drop=True)
    total_returned = products["returned"].sum()
    products["rank"] = np.arange(1, len(products) + 1)
    products["cumulative_returns"] = products["returned"].cumsum()
    products["cumulative_return_share"] = np.where(
        total_returned > 0,
        100 * products["cumulative_returns"] / total_returned,
        0,
    )
    products["cumulative_variant_share"] = 100 * products["rank"] / len(products) if len(products) else 0
    return products


def pareto_breakpoints(pareto_df: pd.DataFrame, thresholds: tuple[int, ...] = (50, 80, 90)) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        reached = pareto_df[pareto_df["cumulative_return_share"] >= threshold]
        if reached.empty:
            continue
        row = reached.iloc[0]
        rows.append(
            {
                "return_share_threshold": threshold,
                "variants_needed": int(row["rank"]),
                "variant_share": row["cumulative_variant_share"],
                "returns_covered": row["cumulative_returns"],
            }
        )
    return pd.DataFrame(rows)


def data_quality_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_rows = len(df)
    total_returned = df["Returned articles"].sum()
    no_details = 0.0
    no_details_column = "Estimated returns - No details"
    if no_details_column in df.columns:
        no_details = df[no_details_column].sum()

    status_columns = [
        column
        for column in ["Estimated return rate status", "Size-related return rate status"]
        if column in df.columns
    ]
    if status_columns:
        unstable_mask = pd.Series(False, index=df.index)
        for column in status_columns:
            unstable_mask = unstable_mask | df[column].astype(str).str.contains("unstable", case=False, na=False)
        unstable_rows = int(unstable_mask.sum())
    else:
        unstable_mask = pd.Series(False, index=df.index)
        unstable_rows = 0
    low_volume_rows = (df["Sold articles"].fillna(0) < 10).sum()

    summary = pd.DataFrame(
        [
            {
                "metric": "Rows after filters",
                "value": total_rows,
                "share": 100.0,
                "interpretation": "The data range currently analyzed in the dashboard.",
            },
            {
                "metric": "Rows with sold < 10",
                "value": low_volume_rows,
                "share": 100 * low_volume_rows / total_rows if total_rows else 0,
                "interpretation": "A high share means a higher risk of noise in the return rate.",
            },
            {
                "metric": "Rows with unstable status",
                "value": unstable_rows,
                "share": 100 * unstable_rows / total_rows if total_rows else 0,
                "interpretation": "These rows should be treated more cautiously in product decisions.",
            },
            {
                "metric": "Estimated returns with no details",
                "value": no_details,
                "share": 100 * no_details / total_returned if total_returned else 0,
                "interpretation": "A high share limits the precision of return reason recommendations.",
            },
        ]
    )

    risk_rows = df.copy()
    risk_rows["low_volume"] = risk_rows["Sold articles"].fillna(0) < 10
    risk_rows["unstable_status"] = unstable_mask.reindex(risk_rows.index).fillna(False)
    risk_rows["data_quality_risk"] = risk_rows["low_volume"] | risk_rows["unstable_status"]
    risk_rows = risk_rows[risk_rows["data_quality_risk"]].sort_values("Returned articles", ascending=False)
    return summary, risk_rows


def season_article_type_analysis(df: pd.DataFrame, top_n_types: int = 12) -> pd.DataFrame:
    data = aggregate_by(df, ["Season", "Article type"])
    top_types = aggregate_by(df, "Article type").head(top_n_types)["Article type"].tolist()
    return data[data["Article type"].isin(top_types)].copy()


def simulate_reason_reduction(df: pd.DataFrame, reasons: list[str], reduction_pct: float) -> dict[str, float]:
    total_sold = df["Sold articles"].sum()
    total_returned = df["Returned articles"].sum()
    impacted_returns = 0.0

    for reason in reasons:
        column = f"Estimated returns - {reason}"
        if column in df.columns:
            impacted_returns += df[column].sum()

    reduced_returns = impacted_returns * reduction_pct / 100
    new_returned = max(total_returned - reduced_returns, 0)
    return {
        "current_returned": float(total_returned),
        "current_return_rate": 100 * total_returned / total_sold if total_sold else 0.0,
        "impacted_returns": float(impacted_returns),
        "reduced_returns": float(reduced_returns),
        "new_returned": float(new_returned),
        "new_return_rate": 100 * new_returned / total_sold if total_sold else 0.0,
        "return_rate_delta": (100 * total_returned / total_sold - 100 * new_returned / total_sold) if total_sold else 0.0,
    }
