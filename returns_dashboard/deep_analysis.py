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
        return "Sprawdź tabelę rozmiarów, komunikację fitu i opinie o zaniżonej rozmiarówce."
    if "too big" in reason_lower:
        return "Sprawdź opis fitu, zdjęcia na modelu i ryzyko zawyżonej rozmiarówki."
    if "described" in reason_lower:
        return "Zweryfikuj opis, zdjęcia, materiał, kolor i oczekiwania ustawiane na karcie produktu."
    if "damaged" in reason_lower:
        return "Sprawdź kontrolę jakości, pakowanie i powtarzalność problemu u dostawcy."
    if "wrong item" in reason_lower:
        return "Sprawdź mapowanie wariantów, kompletację i zgodność kodów produktu."
    if "too expensive" in reason_lower:
        return "Porównaj cenę z kategorią i sprawdź, czy value proposition jest jasne."
    if "late" in reason_lower:
        return "Sprawdź SLA fulfillmentu i kraje, w których opóźnienie występuje najczęściej."
    if "no details" in reason_lower:
        return "Traktuj jako lukę danych: szukaj dodatkowego wzorca po kraju, kategorii i rozmiarówce."
    if "don't like" in reason_lower or "like" in reason_lower:
        return "Zweryfikuj stylizację, zdjęcia, oczekiwania jakościowe i spójność produktu z kategorią."
    if size_balance > 25:
        return "Problem przechyla się w stronę za małych rozmiarów; zacznij od komunikacji fitu."
    if size_balance < -25:
        return "Problem przechyla się w stronę za dużych rozmiarów; zacznij od komunikacji fitu."
    return "Sprawdź produkt w profilu szczegółowym i porównaj go z kategorią oraz rynkami."


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
        "dominant_reason",
        "dominant_reason_returns",
        "dominant_reason_share",
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
            "recommended_action": recommendation_for_reason(reason),
        }
    )

    return {
        "summary": summary,
        "reasons": reason_summary(product_df),
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
                "interpretation": "Zakres danych aktualnie analizowany w dashboardzie.",
            },
            {
                "metric": "Rows with sold < 10",
                "value": low_volume_rows,
                "share": 100 * low_volume_rows / total_rows if total_rows else 0,
                "interpretation": "Wysoki udział oznacza większe ryzyko szumu w return rate.",
            },
            {
                "metric": "Rows with unstable status",
                "value": unstable_rows,
                "share": 100 * unstable_rows / total_rows if total_rows else 0,
                "interpretation": "Te wiersze warto traktować ostrożniej w decyzjach produktowych.",
            },
            {
                "metric": "Estimated returns with no details",
                "value": no_details,
                "share": 100 * no_details / total_returned if total_returned else 0,
                "interpretation": "Wysoki udział ogranicza precyzję rekomendacji powodów zwrotu.",
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


def recommendation_for_reason(reason: str, size_balance: float = 0.0) -> str:
    reason_lower = reason.lower()
    if "too small" in reason_lower:
        return "Sprawdź tabelę rozmiarów, komunikację fitu i opinie o zaniżonej rozmiarówce."
    if "too big" in reason_lower:
        return "Sprawdź opis fitu, zdjęcia na modelu i ryzyko zawyżonej rozmiarówki."
    if "described" in reason_lower:
        return "Zweryfikuj opis, zdjęcia, materiał, kolor i oczekiwania ustawiane na karcie produktu."
    if "damaged" in reason_lower:
        return "Sprawdź kontrolę jakości, pakowanie i powtarzalność problemu u dostawcy."
    if "wrong item" in reason_lower:
        return "Sprawdź mapowanie wariantów, kompletację i zgodność kodów produktu."
    if "too expensive" in reason_lower:
        return "Porównaj cenę z kategorią i sprawdź, czy value proposition jest jasne."
    if "late" in reason_lower:
        return "Sprawdź SLA fulfillmentu i kraje, w których opóźnienie występuje najczęściej."
    if "no details" in reason_lower:
        return "Traktuj jako lukę danych: szukaj dodatkowego wzorca po kraju, kategorii i rozmiarówce."
    if "don't like" in reason_lower or "like" in reason_lower:
        return "Zweryfikuj stylizację, zdjęcia, oczekiwania jakościowe i spójność produktu z kategorią."
    if size_balance > 25:
        return "Problem przechyla się w stronę za małych rozmiarów; zacznij od komunikacji fitu."
    if size_balance < -25:
        return "Problem przechyla się w stronę za dużych rozmiarów; zacznij od komunikacji fitu."
    return "Sprawdź produkt w profilu szczegółowym i porównaj go z kategorią oraz rynkami."


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

    low_volume_rows = int((df["Sold articles"].fillna(0) < 10).sum())
    summary = pd.DataFrame(
        [
            {
                "metric": "Rows after filters",
                "value": total_rows,
                "share": 100.0,
                "interpretation": "Zakres danych aktualnie analizowany w dashboardzie.",
            },
            {
                "metric": "Rows with sold < 10",
                "value": low_volume_rows,
                "share": 100 * low_volume_rows / total_rows if total_rows else 0,
                "interpretation": "Wysoki udział oznacza większe ryzyko szumu w return rate.",
            },
            {
                "metric": "Rows with unstable status",
                "value": unstable_rows,
                "share": 100 * unstable_rows / total_rows if total_rows else 0,
                "interpretation": "Te wiersze warto traktować ostrożniej w decyzjach produktowych.",
            },
            {
                "metric": "Estimated returns with no details",
                "value": no_details,
                "share": 100 * no_details / total_returned if total_returned else 0,
                "interpretation": "Wysoki udział ogranicza precyzję rekomendacji powodów zwrotu.",
            },
        ]
    )

    risk_rows = df.copy()
    risk_rows["low_volume"] = risk_rows["Sold articles"].fillna(0) < 10
    risk_rows["unstable_status"] = unstable_mask.reindex(risk_rows.index).fillna(False)
    risk_rows["data_quality_risk"] = risk_rows["low_volume"] | risk_rows["unstable_status"]
    risk_rows = risk_rows[risk_rows["data_quality_risk"]].sort_values("Returned articles", ascending=False)
    return summary, risk_rows
