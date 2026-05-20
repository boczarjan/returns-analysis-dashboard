from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import clean_reason_name, reason_columns
from .metrics import aggregate_by, product_ranking, reason_summary, weighted_return_rate


SIZE_TOO_BIG = "Estimated returns - Item is too big"
SIZE_TOO_SMALL = "Estimated returns - Item is too small"
QUALITY_REASON_KEYWORDS = (
    "damaged",
    "defect",
    "fault",
    "quality",
    "wrong item",
    "incomplete",
    "missing",
    "broken",
    "stain",
)
PREVENTABLE_REASON_GROUPS = {
    "Size / fit": ("too small", "too big", "size", "fit"),
    "Description / expectations": ("described", "description", "expected", "different"),
    "Visual / material mismatch": ("don't like", "like", "photo", "picture", "color", "colour", "material"),
    "Quality / supplier": QUALITY_REASON_KEYWORDS,
    "Fulfillment / wrong item": ("wrong item", "late", "delivery", "shipping", "arrived", "missing", "incomplete"),
    "Price / value": ("too expensive", "price", "value"),
}


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _reason_estimate_columns(df: pd.DataFrame) -> list[str]:
    return [f"Estimated returns - {clean_reason_name(column)}" for column in reason_columns(df)]


def _reason_estimate_column(df: pd.DataFrame, reason: str) -> str | None:
    column = f"Estimated returns - {reason}"
    return column if column in df.columns else None


def _reason_estimate_columns_matching(df: pd.DataFrame, keywords: tuple[str, ...]) -> list[str]:
    matches = []
    for column in reason_columns(df):
        reason = clean_reason_name(column)
        if any(keyword in reason.lower() for keyword in keywords):
            estimate = f"Estimated returns - {reason}"
            if estimate in df.columns:
                matches.append(estimate)
    return matches


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


def _dominant_estimated_reason(group: pd.DataFrame, estimate_columns: list[str]) -> tuple[str, float, float]:
    if not estimate_columns:
        return "Unknown", 0.0, 0.0
    totals = group[estimate_columns].fillna(0).sum()
    if totals.empty:
        return "Unknown", 0.0, 0.0
    column = str(totals.idxmax())
    reason = column.replace("Estimated returns - ", "", 1)
    estimated = float(totals.max())
    returned = float(group["Returned articles"].sum())
    return reason, estimated, 100 * estimated / returned if returned else 0.0


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def _product_age(df: pd.DataFrame) -> pd.DataFrame:
    age = pd.DataFrame({"Article variant": df["Article variant"].astype(str).unique()})
    age["days_online"] = np.nan
    age["date_first_on_offer"] = pd.NaT

    if "Days online" in df.columns:
        days = pd.to_numeric(df["Days online"], errors="coerce")
        age_by_variant = days.groupby(df["Article variant"].astype(str), dropna=False).median()
        age["days_online"] = age["Article variant"].map(age_by_variant)

    if "Date first on offer" in df.columns:
        dates = pd.to_datetime(df["Date first on offer"], errors="coerce")
        first_date = dates.groupby(df["Article variant"].astype(str), dropna=False).min()
        age["date_first_on_offer"] = age["Article variant"].map(first_date)
        missing_age = age["days_online"].isna() & age["date_first_on_offer"].notna()
        if missing_age.any():
            today = pd.Timestamp.today().normalize()
            age.loc[missing_age, "days_online"] = (
                today - age.loc[missing_age, "date_first_on_offer"]
            ).dt.days

    return age


def _lifecycle_stage(days_online: object, new_days: int = 90) -> str:
    if pd.isna(days_online):
        return "Unknown"
    days = float(days_online)
    if days <= new_days:
        return f"New <= {new_days}d"
    if days <= 180:
        return "Ramping 91-180d"
    if days <= 365:
        return "Established 181-365d"
    return "Mature >365d"


def _priority_from_score(score: float, high: float = 80, medium: float = 40) -> str:
    if score >= high:
        return "High"
    if score >= medium:
        return "Medium"
    return "Low"


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


def lifecycle_products(df: pd.DataFrame, min_sold: int = 10, new_days: int = 90) -> pd.DataFrame:
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
        "avg_net_price",
        "estimated_returned_nmv",
        "days_online",
        "date_first_on_offer",
        "lifecycle_stage",
        "dominant_reason",
        "dominant_reason_returns",
        "dominant_reason_share",
        "size_balance",
        "confidence_flag",
        "recommended_action",
        "early_warning_score",
        "early_warning_priority",
    ]
    if "Article variant" not in df.columns:
        return _empty(columns)

    products = benchmark_products(df, min_sold=min_sold).copy()
    if products.empty:
        return _empty(columns)

    age = _product_age(df)
    if age["days_online"].isna().all() and age["date_first_on_offer"].isna().all():
        return _empty(columns)

    actions = action_list(df, min_sold=min_sold)
    action_columns = [
        column
        for column in [
            "Article variant",
            "dominant_reason",
            "dominant_reason_returns",
            "dominant_reason_share",
            "size_balance",
            "confidence_flag",
            "recommended_action",
        ]
        if column in actions.columns
    ]
    if action_columns:
        products = products.merge(actions[action_columns], on="Article variant", how="left")
    products = products.merge(age, on="Article variant", how="left")

    products["lifecycle_stage"] = products["days_online"].map(lambda value: _lifecycle_stage(value, new_days))
    positive_gap = np.maximum(products.get("gap_vs_category", 0).fillna(0), 0)
    size_pressure = np.maximum(products.get("size_balance", 0).abs().fillna(0) - 25, 0)
    age_weight = np.where(
        products["days_online"].fillna(new_days + 1) <= new_days,
        1 + (new_days - products["days_online"].fillna(new_days)).clip(lower=0) / max(new_days, 1),
        0.65,
    )
    products["early_warning_score"] = (
        positive_gap * 2.0
        + products["returned"].fillna(0) * 0.55
        + products.get("dominant_reason_share", 0).fillna(0) * 0.25
        + size_pressure * 0.5
    ) * age_weight
    products["early_warning_priority"] = products["early_warning_score"].map(_priority_from_score)

    for column in columns:
        if column not in products.columns:
            products[column] = "" if column.endswith(("reason", "action", "flag", "priority")) else 0.0
    return products[columns].sort_values(["days_online", "early_warning_score"], ascending=[True, False])


def new_product_early_warning(
    df: pd.DataFrame,
    max_days_online: int = 90,
    min_sold: int = 10,
    min_returned: int = 1,
) -> pd.DataFrame:
    products = lifecycle_products(df, min_sold=min_sold, new_days=max_days_online)
    if products.empty:
        return products
    warnings = products[
        (products["days_online"].fillna(max_days_online + 1) <= max_days_online)
        & (products["returned"].fillna(0) >= min_returned)
        & (
            (products["gap_vs_category"].fillna(0) > 0)
            | (products["dominant_reason_share"].fillna(0) >= 45)
            | (products["size_balance"].abs().fillna(0) >= 35)
        )
    ].copy()
    return warnings.sort_values("early_warning_score", ascending=False)


def country_report(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Country",
        "sold",
        "returned",
        "return_rate",
        "return_share",
        "gap_vs_dataset",
        "variants",
        "top_reason",
        "top_reason_returns",
        "top_reason_share",
        "size_returns",
        "size_share_of_returns",
        "size_balance",
        "high_risk_variants",
        "estimated_returned_nmv",
        "recommended_focus",
    ]
    if "Country" not in df.columns:
        return _empty(columns)

    countries = aggregate_by(df, "Country").copy()
    if countries.empty:
        return _empty(columns)

    dataset_rr = weighted_return_rate(df)
    countries["gap_vs_dataset"] = countries["return_rate"] - dataset_rr

    too_big_col = _reason_estimate_column(df, "Item is too big")
    too_small_col = _reason_estimate_column(df, "Item is too small")
    product_country = aggregate_by(df, ["Country", "Article variant"])
    country_rr = countries.set_index("Country")["return_rate"]
    product_country["country_return_rate"] = product_country["Country"].map(country_rr).fillna(0)
    product_country["gap_vs_country"] = product_country["return_rate"] - product_country["country_return_rate"]
    risk_counts = (
        product_country[
            (product_country["returned"].fillna(0) >= 2)
            & (product_country["gap_vs_country"].fillna(0) >= 5)
        ]
        .groupby("Country", dropna=False)["Article variant"]
        .nunique()
    )

    rows = []
    for country, group in df.groupby("Country", dropna=False):
        top_reason, top_returns, top_share = _dominant_reason(group)
        too_big = group[too_big_col].sum() if too_big_col else 0.0
        too_small = group[too_small_col].sum() if too_small_col else 0.0
        returned = group["Returned articles"].sum()
        size_returns = too_big + too_small
        size_balance = 100 * (too_small - too_big) / size_returns if size_returns else 0.0
        rows.append(
            {
                "Country": country,
                "top_reason": top_reason,
                "top_reason_returns": top_returns,
                "top_reason_share": top_share,
                "size_returns": float(size_returns),
                "size_share_of_returns": 100 * size_returns / returned if returned else 0.0,
                "size_balance": float(size_balance),
                "high_risk_variants": float(risk_counts.get(country, 0)),
                "recommended_focus": recommendation_for_reason(top_reason, size_balance),
            }
        )

    details = pd.DataFrame(rows)
    report = countries.merge(details, on="Country", how="left")
    for column in columns:
        if column not in report.columns:
            report[column] = 0.0
    return report[columns].sort_values(["returned", "gap_vs_dataset"], ascending=[False, False])


def size_fit_intelligence(df: pd.DataFrame, min_sold: int = 10) -> pd.DataFrame:
    columns = [
        "Article variant",
        "Zalando article variant",
        "Category",
        "Article type",
        "Gender",
        "Season",
        "sold",
        "returned",
        "return_rate",
        "category_return_rate",
        "gap_vs_category",
        "too_big_returns",
        "too_small_returns",
        "size_returns",
        "size_return_rate",
        "size_share_of_returns",
        "too_big_share",
        "too_small_share",
        "size_balance",
        "fit_issue",
        "fit_risk_score",
        "recommended_action",
    ]
    if SIZE_TOO_BIG not in df.columns or SIZE_TOO_SMALL not in df.columns:
        return _empty(columns)

    products = benchmark_products(df, min_sold=min_sold).copy()
    if products.empty:
        return _empty(columns)

    grouped = (
        df.groupby("Article variant", dropna=False)
        .agg(
            too_big_returns=(SIZE_TOO_BIG, "sum"),
            too_small_returns=(SIZE_TOO_SMALL, "sum"),
        )
        .reset_index()
    )
    products = products.merge(grouped, on="Article variant", how="left")
    products["too_big_returns"] = products["too_big_returns"].fillna(0)
    products["too_small_returns"] = products["too_small_returns"].fillna(0)
    products["size_returns"] = products["too_big_returns"] + products["too_small_returns"]
    products["size_return_rate"] = np.where(
        products["sold"] > 0,
        100 * products["size_returns"] / products["sold"],
        0.0,
    )
    products["size_share_of_returns"] = np.where(
        products["returned"] > 0,
        100 * products["size_returns"] / products["returned"],
        0.0,
    )
    products["too_big_share"] = np.where(
        products["size_returns"] > 0,
        100 * products["too_big_returns"] / products["size_returns"],
        0.0,
    )
    products["too_small_share"] = np.where(
        products["size_returns"] > 0,
        100 * products["too_small_returns"] / products["size_returns"],
        0.0,
    )
    products["size_balance"] = np.where(
        products["size_returns"] > 0,
        100 * (products["too_small_returns"] - products["too_big_returns"]) / products["size_returns"],
        0.0,
    )
    products["fit_issue"] = np.select(
        [
            products["size_balance"] >= 25,
            products["size_balance"] <= -25,
            products["size_share_of_returns"] >= 35,
        ],
        ["Too small skew", "Too big skew", "High size share"],
        default="Balanced / low signal",
    )
    products["fit_risk_score"] = (
        products["size_returns"].fillna(0) * 0.8
        + products["size_return_rate"].fillna(0) * 1.6
        + products["size_share_of_returns"].fillna(0) * 0.35
        + np.maximum(products["gap_vs_category"].fillna(0), 0) * 1.2
        + np.maximum(products["size_balance"].abs().fillna(0) - 25, 0) * 0.55
    )
    products["recommended_action"] = [
        recommendation_for_reason("Item is too small" if balance > 0 else "Item is too big", balance)
        if size_returns > 0
        else "Keep monitoring; sizing signal is weak in the current filters."
        for balance, size_returns in zip(products["size_balance"], products["size_returns"], strict=False)
    ]

    for column in columns:
        if column not in products.columns:
            products[column] = "" if column in {"Gender", "Season", "fit_issue", "recommended_action"} else 0.0
    return products[columns].sort_values("fit_risk_score", ascending=False)


def size_fit_dimension_report(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    columns = [
        dimension,
        "sold",
        "returned",
        "return_rate",
        "size_returns",
        "size_return_rate",
        "size_share_of_returns",
        "too_big_returns",
        "too_small_returns",
        "size_balance",
        "variants",
    ]
    if dimension not in df.columns or SIZE_TOO_BIG not in df.columns or SIZE_TOO_SMALL not in df.columns:
        return _empty(columns)
    report = (
        df.groupby(dimension, dropna=False)
        .agg(
            sold=("Sold articles", "sum"),
            returned=("Returned articles", "sum"),
            too_big_returns=(SIZE_TOO_BIG, "sum"),
            too_small_returns=(SIZE_TOO_SMALL, "sum"),
            variants=("Article variant", "nunique"),
        )
        .reset_index()
    )
    report["return_rate"] = np.where(report["sold"] > 0, 100 * report["returned"] / report["sold"], 0.0)
    report["size_returns"] = report["too_big_returns"] + report["too_small_returns"]
    report["size_return_rate"] = np.where(report["sold"] > 0, 100 * report["size_returns"] / report["sold"], 0.0)
    report["size_share_of_returns"] = np.where(
        report["returned"] > 0,
        100 * report["size_returns"] / report["returned"],
        0.0,
    )
    report["size_balance"] = np.where(
        report["size_returns"] > 0,
        100 * (report["too_small_returns"] - report["too_big_returns"]) / report["size_returns"],
        0.0,
    )
    return report[columns].sort_values("size_returns", ascending=False)


def quality_supplier_report(df: pd.DataFrame, dimension: str = "Category") -> pd.DataFrame:
    columns = [
        dimension,
        "sold",
        "returned",
        "return_rate",
        "variants",
        "quality_returns",
        "quality_return_rate",
        "quality_share_of_returns",
        "gap_vs_dataset_quality_share",
        "top_quality_reason",
        "top_quality_reason_returns",
        "risk_score",
        "recommended_action",
    ]
    if dimension not in df.columns:
        return _empty(columns)
    quality_columns = _reason_estimate_columns_matching(df, QUALITY_REASON_KEYWORDS)
    if not quality_columns:
        return _empty(columns)

    dataset_quality_returns = df[quality_columns].fillna(0).sum(axis=1).sum()
    dataset_returned = df["Returned articles"].sum()
    dataset_quality_share = 100 * dataset_quality_returns / dataset_returned if dataset_returned else 0.0

    rows = []
    for value, group in df.groupby(dimension, dropna=False):
        sold = float(group["Sold articles"].sum())
        returned = float(group["Returned articles"].sum())
        quality_returns = float(group[quality_columns].fillna(0).sum(axis=1).sum())
        top_reason, top_returns, _ = _dominant_estimated_reason(group, quality_columns)
        quality_share = 100 * quality_returns / returned if returned else 0.0
        quality_rate = 100 * quality_returns / sold if sold else 0.0
        gap = quality_share - dataset_quality_share
        rows.append(
            {
                dimension: value,
                "sold": sold,
                "returned": returned,
                "return_rate": 100 * returned / sold if sold else 0.0,
                "variants": float(group["Article variant"].nunique()),
                "quality_returns": quality_returns,
                "quality_return_rate": quality_rate,
                "quality_share_of_returns": quality_share,
                "gap_vs_dataset_quality_share": gap,
                "top_quality_reason": top_reason,
                "top_quality_reason_returns": top_returns,
                "risk_score": quality_returns * 0.9 + max(gap, 0) * 3 + quality_rate * 2,
                "recommended_action": (
                    "Review supplier, QC, packaging, and variant mapping for this group."
                    if quality_returns > 0
                    else "No material quality/supplier signal in the current filters."
                ),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values("risk_score", ascending=False)


def pricing_risk_report(
    df: pd.DataFrame,
    min_sold: int = 30,
    premium_threshold: int = 110,
    return_gap_threshold: int = 0,
) -> pd.DataFrame:
    products = benchmark_products(df, min_sold=min_sold).copy()
    if products.empty or "nmv" not in products.columns or products["nmv"].fillna(0).sum() <= 0:
        return _empty(
            [
                "Article variant",
                "Zalando article variant",
                "Category",
                "Article type",
                "sold",
                "returned",
                "return_rate",
                "category_return_rate",
                "gap_vs_category",
                "avg_net_price",
                "category_avg_net_price",
                "price_index_vs_category",
                "estimated_returned_nmv",
                "pricing_risk_score",
                "risk_type",
                "recommended_action",
            ]
        )

    risk = products[
        (products["price_index_vs_category"].fillna(0) >= premium_threshold)
        & (products["gap_vs_category"].fillna(0) >= return_gap_threshold)
    ].copy()
    if risk.empty:
        return risk

    max_value = max(float(risk["estimated_returned_nmv"].fillna(0).max()), 1.0)
    value_component = 100 * risk["estimated_returned_nmv"].fillna(0) / max_value
    risk["pricing_risk_score"] = (
        np.maximum(risk["price_index_vs_category"].fillna(0) - 100, 0) * 0.8
        + np.maximum(risk["gap_vs_category"].fillna(0), 0) * 2.0
        + value_component * 0.7
    )
    risk["risk_type"] = np.select(
        [
            (risk["price_index_vs_category"] >= 130) & (risk["gap_vs_category"] >= 10),
            risk["price_index_vs_category"] >= 130,
            risk["gap_vs_category"] >= 10,
        ],
        ["Premium price and high return gap", "Premium price", "High return gap"],
        default="Watch price/value",
    )
    risk["recommended_action"] = (
        "Check price/value communication, competitive price position, photos, material promises, and reason mix."
    )
    return risk.sort_values("pricing_risk_score", ascending=False)


def preventable_returns_report(df: pd.DataFrame, dimension: str = "Article variant", min_sold: int = 10) -> pd.DataFrame:
    columns = [
        dimension,
        "sold",
        "returned",
        "return_rate",
        "variants",
        "preventable_returns",
        "preventable_return_rate",
        "preventable_share_of_returns",
        "top_preventable_area",
        "top_preventable_returns",
        "preventable_score",
        "recommended_action",
    ]
    if dimension not in df.columns:
        return _empty(columns)

    group_columns = {
        area: _reason_estimate_columns_matching(df, keywords)
        for area, keywords in PREVENTABLE_REASON_GROUPS.items()
    }
    group_columns = {area: cols for area, cols in group_columns.items() if cols}
    if not group_columns:
        return _empty(columns)

    rows = []
    for value, group in df.groupby(dimension, dropna=False):
        sold = float(group["Sold articles"].sum())
        if sold < min_sold:
            continue
        returned = float(group["Returned articles"].sum())
        area_totals = {
            area: float(group[cols].fillna(0).sum(axis=1).sum())
            for area, cols in group_columns.items()
        }
        top_area, top_returns = max(area_totals.items(), key=lambda item: item[1])
        preventable_returns = float(sum(area_totals.values()))
        if returned:
            preventable_returns = min(preventable_returns, returned)
        action_map = {
            "Size / fit": "Add fit notes, review size chart, and compare too-small vs too-big skew.",
            "Description / expectations": "Rewrite PDP copy around material, dimensions, use case, and expectations.",
            "Visual / material mismatch": "Review photos, color accuracy, scale cues, detail shots, and material promises.",
            "Quality / supplier": "Review QC, supplier batch, packaging, and repeat damage/defect signals.",
            "Fulfillment / wrong item": "Audit picking, variant mapping, warehouse flow, and delivery promise.",
            "Price / value": "Check competitive price, discount context, and value communication.",
        }
        rows.append(
            {
                dimension: value,
                "sold": sold,
                "returned": returned,
                "return_rate": 100 * returned / sold if sold else 0.0,
                "variants": float(group["Article variant"].nunique()),
                "preventable_returns": preventable_returns,
                "preventable_return_rate": 100 * preventable_returns / sold if sold else 0.0,
                "preventable_share_of_returns": 100 * preventable_returns / returned if returned else 0.0,
                "top_preventable_area": top_area,
                "top_preventable_returns": top_returns,
                "preventable_score": preventable_returns * 0.9 + max(100 * preventable_returns / sold if sold else 0, 0) * 2,
                "recommended_action": action_map.get(top_area, "Review preventable reason mix and assign an owner."),
            }
        )
    if not rows:
        return _empty(columns)
    return pd.DataFrame(rows)[columns].sort_values("preventable_score", ascending=False)


def predictive_return_risk(df: pd.DataFrame, min_sold: int = 10) -> pd.DataFrame:
    columns = [
        "Article variant",
        "Zalando article variant",
        "Category",
        "Article type",
        "sold",
        "returned",
        "return_rate",
        "category_return_rate",
        "type_return_rate",
        "dataset_return_rate",
        "gap_vs_category",
        "gap_vs_type",
        "gap_vs_dataset",
        "avg_net_price",
        "price_index_vs_category",
        "estimated_returned_nmv",
        "dominant_reason",
        "dominant_reason_share",
        "size_balance",
        "days_online",
        "predicted_return_rate",
        "predicted_returns_next_30d",
        "predictive_risk_score",
        "predictive_risk_level",
        "risk_drivers",
        "recommended_action",
    ]
    products = benchmark_products(df, min_sold=min_sold).copy()
    if products.empty:
        return _empty(columns)

    actions = action_list(df, min_sold=min_sold)
    action_cols = [
        column
        for column in ["Article variant", "dominant_reason", "dominant_reason_share", "size_balance", "recommended_action"]
        if column in actions.columns
    ]
    if action_cols:
        products = products.merge(actions[action_cols], on="Article variant", how="left")
    age = _product_age(df)
    products = products.merge(age[["Article variant", "days_online"]], on="Article variant", how="left")

    dataset_rr = weighted_return_rate(df)
    products["dataset_return_rate"] = dataset_rr
    positive_category_gap = np.maximum(products["gap_vs_category"].fillna(0), 0)
    positive_type_gap = np.maximum(products["gap_vs_type"].fillna(0), 0)
    price_index = _numeric_series(products, "price_index_vs_category")
    reason_pressure = _numeric_series(products, "dominant_reason_share")
    size_balance = _numeric_series(products, "size_balance")
    returned_nmv = _numeric_series(products, "estimated_returned_nmv")
    price_pressure = np.maximum(price_index - 100, 0)
    size_pressure = np.maximum(size_balance.abs() - 25, 0)
    age_pressure = np.where(products["days_online"].fillna(999) <= 90, 12, 0)
    value_scale = 100 * returned_nmv / max(float(returned_nmv.max()), 1.0)

    products["predicted_return_rate"] = (
        products["return_rate"].fillna(0) * 0.45
        + products["category_return_rate"].fillna(dataset_rr) * 0.25
        + products["type_return_rate"].fillna(dataset_rr) * 0.15
        + dataset_rr * 0.15
        + positive_category_gap * 0.20
        + price_pressure * 0.03
        + size_pressure * 0.04
    )
    observed_days = products["days_online"].fillna(30).clip(lower=7)
    daily_sold = products["sold"].fillna(0) / observed_days
    products["predicted_returns_next_30d"] = daily_sold * 30 * products["predicted_return_rate"] / 100
    products["predictive_risk_score"] = (
        positive_category_gap * 2.2
        + positive_type_gap * 1.1
        + products["predicted_returns_next_30d"].fillna(0) * 1.5
        + reason_pressure * 0.25
        + size_pressure * 0.6
        + price_pressure * 0.2
        + value_scale * 0.45
        + age_pressure
    )
    products["predictive_risk_level"] = products["predictive_risk_score"].map(_priority_from_score)

    drivers = []
    for row in products.itertuples(index=False):
        row_drivers = []
        if getattr(row, "gap_vs_category", 0) > 5:
            row_drivers.append("category gap")
        if getattr(row, "gap_vs_type", 0) > 5:
            row_drivers.append("type gap")
        if getattr(row, "price_index_vs_category", 0) > 115:
            row_drivers.append("price premium")
        if abs(getattr(row, "size_balance", 0) or 0) > 35:
            row_drivers.append("size skew")
        if getattr(row, "dominant_reason_share", 0) > 50:
            row_drivers.append("reason concentration")
        if getattr(row, "days_online", 999) <= 90:
            row_drivers.append("new product")
        drivers.append(", ".join(row_drivers) if row_drivers else "baseline risk")
    products["risk_drivers"] = drivers
    default_action = "Review benchmark gaps, reason mix, and product page audit pack."
    if "recommended_action" in products.columns:
        products["recommended_action"] = products["recommended_action"].fillna(default_action)
    else:
        products["recommended_action"] = default_action

    for column in columns:
        if column not in products.columns:
            products[column] = "" if column in {"risk_drivers", "recommended_action", "dominant_reason", "predictive_risk_level"} else 0.0
    return products[columns].sort_values("predictive_risk_score", ascending=False)


def size_guidance_report(df: pd.DataFrame, min_sold: int = 10) -> pd.DataFrame:
    columns = [
        "Article variant",
        "Zalando article variant",
        "Category",
        "Article type",
        "Gender",
        "Season",
        "sold",
        "returned",
        "return_rate",
        "size_returns",
        "size_return_rate",
        "size_share_of_returns",
        "too_big_returns",
        "too_small_returns",
        "size_balance",
        "guidance",
        "confidence",
        "guidance_score",
        "recommended_copy",
    ]
    size_df = size_fit_intelligence(df, min_sold=min_sold).copy()
    if size_df.empty:
        return _empty(columns)

    guidance = []
    copy = []
    confidence = []
    for row in size_df.itertuples(index=False):
        balance = float(getattr(row, "size_balance", 0) or 0)
        size_share = float(getattr(row, "size_share_of_returns", 0) or 0)
        sold = float(getattr(row, "sold", 0) or 0)
        if balance >= 35 and size_share >= 20:
            guidance.append("Recommend size up")
            copy.append("Runs small for many customers. Consider sizing up for a more comfortable fit.")
        elif balance <= -35 and size_share >= 20:
            guidance.append("Recommend size down")
            copy.append("Runs large for many customers. Consider sizing down for a closer fit.")
        elif size_share >= 35:
            guidance.append("Improve fit explanation")
            copy.append("Fit feedback is mixed. Add fit notes, measurements, and model-size context.")
        else:
            guidance.append("Monitor")
            copy.append("Sizing signal is not strong enough for a customer-facing size note yet.")
        if sold >= 100 and size_share >= 25:
            confidence.append("High")
        elif sold >= 30 and size_share >= 15:
            confidence.append("Medium")
        else:
            confidence.append("Low")
    size_df["guidance"] = guidance
    size_df["confidence"] = confidence
    size_df["recommended_copy"] = copy
    size_df["guidance_score"] = (
        size_df["size_returns"].fillna(0) * 0.8
        + size_df["size_share_of_returns"].fillna(0) * 0.7
        + np.maximum(size_df["size_balance"].abs().fillna(0) - 25, 0) * 0.9
    )

    for column in columns:
        if column not in size_df.columns:
            size_df[column] = ""
    return size_df[columns].sort_values("guidance_score", ascending=False)


def trend_report(current_df: pd.DataFrame, baseline_df: pd.DataFrame, dimension: str = "Article variant") -> pd.DataFrame:
    columns = [
        dimension,
        "sold_current",
        "sold_baseline",
        "sold_delta",
        "returned_current",
        "returned_baseline",
        "returned_delta",
        "return_rate_current",
        "return_rate_baseline",
        "return_rate_delta",
        "estimated_returned_nmv_current",
        "estimated_returned_nmv_baseline",
        "estimated_returned_nmv_delta",
        "trend_score",
        "trend_direction",
    ]
    if dimension not in current_df.columns or dimension not in baseline_df.columns:
        return _empty(columns)

    current = aggregate_by(current_df, dimension).rename(
        columns={
            "sold": "sold_current",
            "returned": "returned_current",
            "return_rate": "return_rate_current",
            "estimated_returned_nmv": "estimated_returned_nmv_current",
        }
    )
    baseline = aggregate_by(baseline_df, dimension).rename(
        columns={
            "sold": "sold_baseline",
            "returned": "returned_baseline",
            "return_rate": "return_rate_baseline",
            "estimated_returned_nmv": "estimated_returned_nmv_baseline",
        }
    )
    report = current.merge(
        baseline[
            [
                dimension,
                "sold_baseline",
                "returned_baseline",
                "return_rate_baseline",
                "estimated_returned_nmv_baseline",
            ]
        ],
        on=dimension,
        how="outer",
    )
    for column in columns:
        if column not in report.columns:
            report[column] = 0.0
    for column in [
        "sold_current",
        "sold_baseline",
        "returned_current",
        "returned_baseline",
        "return_rate_current",
        "return_rate_baseline",
        "estimated_returned_nmv_current",
        "estimated_returned_nmv_baseline",
    ]:
        report[column] = report[column].fillna(0)
    report["sold_delta"] = report["sold_current"] - report["sold_baseline"]
    report["returned_delta"] = report["returned_current"] - report["returned_baseline"]
    report["return_rate_delta"] = report["return_rate_current"] - report["return_rate_baseline"]
    report["estimated_returned_nmv_delta"] = (
        report["estimated_returned_nmv_current"] - report["estimated_returned_nmv_baseline"]
    )
    report["trend_score"] = (
        np.maximum(report["return_rate_delta"], 0) * 2.5
        + np.maximum(report["returned_delta"], 0) * 0.8
        + np.maximum(report["estimated_returned_nmv_delta"], 0) / max(
            float(report["estimated_returned_nmv_delta"].clip(lower=0).max()),
            1.0,
        ) * 50
    )
    report["trend_direction"] = np.select(
        [
            report["return_rate_delta"] >= 3,
            report["return_rate_delta"] <= -3,
            report["returned_delta"] > 0,
            report["returned_delta"] < 0,
        ],
        ["Worse RR", "Better RR", "More returns", "Fewer returns"],
        default="Stable",
    )
    return report[columns].sort_values(["trend_score", "returned_current"], ascending=[False, False])


def forecast_report(df: pd.DataFrame, horizon_days: int = 30, min_sold: int = 10) -> pd.DataFrame:
    columns = [
        "Article variant",
        "Zalando article variant",
        "Category",
        "Article type",
        "sold",
        "returned",
        "return_rate",
        "days_online",
        "daily_sold",
        "daily_returns",
        "forecast_sold",
        "forecast_returns",
        "forecast_return_rate",
        "forecast_returned_nmv",
        "forecast_risk_score",
        "forecast_risk_level",
        "forecast_note",
    ]
    products = predictive_return_risk(df, min_sold=min_sold).copy()
    if products.empty:
        return _empty(columns)
    observed_days = products["days_online"].fillna(30).clip(lower=7)
    products["daily_sold"] = products["sold"].fillna(0) / observed_days
    products["daily_returns"] = products["returned"].fillna(0) / observed_days
    products["forecast_sold"] = products["daily_sold"] * horizon_days
    products["forecast_returns"] = products["forecast_sold"] * products["predicted_return_rate"].fillna(0) / 100
    products["forecast_return_rate"] = products["predicted_return_rate"].fillna(products["return_rate"])
    products["forecast_returned_nmv"] = products["forecast_returns"] * _numeric_series(products, "avg_net_price")
    products["forecast_risk_score"] = (
        products["forecast_returns"].fillna(0) * 1.5
        + np.maximum(products["forecast_return_rate"].fillna(0) - weighted_return_rate(df), 0) * 2
        + products["forecast_returned_nmv"].fillna(0) / max(float(products["forecast_returned_nmv"].fillna(0).max()), 1.0) * 50
    )
    products["forecast_risk_level"] = products["forecast_risk_score"].map(_priority_from_score)
    products["forecast_note"] = np.where(
        products["days_online"].isna(),
        "Uses 30 observed days fallback because product age is missing.",
        f"Forecast horizon: {horizon_days} days.",
    )
    for column in columns:
        if column not in products.columns:
            products[column] = "" if column.endswith(("level", "note")) else 0.0
    return products[columns].sort_values("forecast_risk_score", ascending=False)


def product_audit_pack(df: pd.DataFrame, article_variant: str) -> dict[str, pd.DataFrame | dict[str, float | str]]:
    profile = product_profile(df, article_variant)
    if not profile:
        return {}

    summary = profile["summary"]
    raw = profile["raw"]
    reasons = profile["reason_gap_vs_category"].copy()
    countries = profile["countries"].copy()
    returned = float(summary.get("returned", 0.0))

    def reason_signal(keywords: tuple[str, ...]) -> tuple[float, float, str]:
        if reasons.empty:
            return 0.0, 0.0, "Unknown"
        mask = reasons["reason"].astype(str).str.lower().apply(
            lambda value: any(keyword in value for keyword in keywords)
        )
        matched = reasons[mask]
        if matched.empty:
            return 0.0, 0.0, "Unknown"
        matched = matched.sort_values(["gap_vs_category", "estimated_returns"], ascending=[False, False])
        row = matched.iloc[0]
        return float(row.get("gap_vs_category", 0.0)), float(row.get("estimated_returns", 0.0)), str(row["reason"])

    def priority(gap: float = 0.0, estimated: float = 0.0, absolute_signal: float = 0.0) -> str:
        if estimated >= max(returned * 0.25, 5) or gap >= 15 or absolute_signal >= 40:
            return "High"
        if estimated >= max(returned * 0.1, 2) or gap >= 7 or absolute_signal >= 25:
            return "Medium"
        return "Low"

    size_balance = float(summary.get("size_balance", 0.0))
    size_reason = "Item is too small" if size_balance > 0 else "Item is too big"
    desc_gap, desc_returns, desc_reason = reason_signal(("described", "description", "expected", "different"))
    image_gap, image_returns, image_reason = reason_signal(("don't like", "like", "photo", "picture", "color", "colour", "material"))
    quality_gap, quality_returns, quality_reason = reason_signal(QUALITY_REASON_KEYWORDS)
    price_gap, price_returns, price_reason = reason_signal(("too expensive", "price", "value"))
    fulfill_gap, fulfill_returns, fulfill_reason = reason_signal(("late", "delivery", "shipping", "arrived"))

    top_country = "Unknown"
    top_country_share = 0.0
    if not countries.empty and returned:
        top_country_row = countries.sort_values("returned", ascending=False).iloc[0]
        top_country = str(top_country_row.get("Country", "Unknown"))
        top_country_share = 100 * float(top_country_row.get("returned", 0.0)) / returned

    checklist_rows = [
        {
            "area": "Fit and size",
            "priority": priority(absolute_signal=abs(size_balance)),
            "signal": f"{size_reason} skew: {size_balance:+.1f} p.p.",
            "recommended_check": recommendation_for_reason(size_reason, size_balance),
        },
        {
            "area": "Description accuracy",
            "priority": priority(desc_gap, desc_returns),
            "signal": f"{desc_reason}: {desc_returns:.0f} est. returns, gap {desc_gap:+.1f} p.p.",
            "recommended_check": "Compare title, bullets, material, dimensions, and expectations set on the PDP.",
        },
        {
            "area": "Images and styling",
            "priority": priority(image_gap, image_returns),
            "signal": f"{image_reason}: {image_returns:.0f} est. returns, gap {image_gap:+.1f} p.p.",
            "recommended_check": "Review hero image, detail shots, color accuracy, model styling, and visual fit cues.",
        },
        {
            "area": "Quality / supplier",
            "priority": priority(quality_gap, quality_returns),
            "signal": f"{quality_reason}: {quality_returns:.0f} est. returns, gap {quality_gap:+.1f} p.p.",
            "recommended_check": "Check supplier batch, QC notes, packaging, picking accuracy, and repeat defects.",
        },
        {
            "area": "Price / value",
            "priority": priority(price_gap, price_returns, max(float(summary.get("price_index_vs_category", 0)) - 100, 0)),
            "signal": (
                f"Price index {float(summary.get('price_index_vs_category', 0)):.1f}; "
                f"{price_reason}: {price_returns:.0f} est. returns"
            ),
            "recommended_check": "Check price position, discount context, competitor set, and whether value is visible on the PDP.",
        },
        {
            "area": "Fulfillment promise",
            "priority": priority(fulfill_gap, fulfill_returns),
            "signal": f"{fulfill_reason}: {fulfill_returns:.0f} est. returns, gap {fulfill_gap:+.1f} p.p.",
            "recommended_check": "Check delivery promise, late-return concentration, and countries with fulfillment pressure.",
        },
        {
            "area": "Market localization",
            "priority": priority(absolute_signal=top_country_share if top_country_share >= 55 and len(countries) > 1 else 0),
            "signal": f"Top country {top_country}: {top_country_share:.1f}% of variant returns.",
            "recommended_check": "Review whether the issue is market-specific before applying a global product change.",
        },
    ]
    checklist = pd.DataFrame(checklist_rows)
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    checklist["_order"] = checklist["priority"].map(priority_order).fillna(9)
    checklist = checklist.sort_values(["_order", "area"]).drop(columns="_order").reset_index(drop=True)

    action_plan = checklist[checklist["priority"].isin(["High", "Medium"])].copy()
    action_plan.insert(0, "Article variant", article_variant)
    action_plan["owner_hint"] = action_plan["area"].map(
        {
            "Fit and size": "Product / fit",
            "Description accuracy": "Content",
            "Images and styling": "Creative / content",
            "Quality / supplier": "Quality / buying",
            "Price / value": "Pricing / merchandising",
            "Fulfillment promise": "Operations",
            "Market localization": "Country manager",
        }
    ).fillna("Business owner")

    return {
        "summary": summary,
        "checklist": checklist,
        "action_plan": action_plan,
        "raw": raw,
    }


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
    category_df = df[df["Category"].eq(category)] if category else df.iloc[0:0]
    type_df = df[df["Article type"].eq(article_type)] if article_type else df.iloc[0:0]
    category_sold = category_df["Sold articles"].sum() if not category_df.empty else 0.0
    type_sold = type_df["Sold articles"].sum() if not type_df.empty else 0.0
    category_avg_price = category_df["NMV"].sum() / category_sold if category_sold else 0.0
    type_avg_price = type_df["NMV"].sum() / type_sold if type_sold else 0.0
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
            "category_avg_net_price": category_avg_price,
            "type_avg_net_price": type_avg_price,
            "price_gap_vs_category": summary.get("avg_net_price", 0.0) - category_avg_price,
            "price_index_vs_category": (
                100 * summary.get("avg_net_price", 0.0) / category_avg_price if category_avg_price else 0.0
            ),
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
