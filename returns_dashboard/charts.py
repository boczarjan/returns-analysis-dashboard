from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORWAY = [
    "#31572c",
    "#4f772d",
    "#90a955",
    "#f9c74f",
    "#f8961e",
    "#f3722c",
    "#577590",
    "#43aa8b",
    "#277da1",
]


def apply_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        colorway=COLORWAY,
        margin=dict(l=16, r=16, t=48, b=24),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#263238"),
        title=dict(font=dict(size=18, color="#102027")),
        hoverlabel=dict(bgcolor="#102027", font_size=12, font_color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf2ef", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf2ef", zeroline=False)
    return fig


def reason_bar(reason_df: pd.DataFrame) -> go.Figure:
    data = reason_df.sort_values("estimated_returns", ascending=True)
    fig = px.bar(
        data,
        x="estimated_returns",
        y="reason",
        orientation="h",
        color="share_of_returns",
        color_continuous_scale=["#d8f3dc", "#52b788", "#1b4332"],
        labels={
            "estimated_returns": "Estimated returned items",
            "reason": "Reason",
            "share_of_returns": "Share of returns (%)",
        },
        title="Top return reasons",
        hover_data={"share_of_returns": ":.1f", "estimated_returns": ":,.0f"},
    )
    return apply_layout(fig, 430)


def country_bar(country_df: pd.DataFrame) -> go.Figure:
    data = country_df.sort_values("returned", ascending=True)
    fig = px.bar(
        data,
        x="returned",
        y="Country",
        orientation="h",
        color="return_rate",
        color_continuous_scale=["#caf0f8", "#00b4d8", "#03045e"],
        labels={"returned": "Returned items", "return_rate": "Return rate (%)"},
        title="Returns by country",
        hover_data={"sold": ":,.0f", "returned": ":,.0f", "return_rate": ":.1f"},
    )
    return apply_layout(fig, 430)


def return_rate_scatter(df: pd.DataFrame, label_col: str, title: str) -> go.Figure:
    if df.empty:
        return apply_layout(go.Figure(), 430)
    custom_data = [label_col] if label_col in df.columns else None
    fig = px.scatter(
        df,
        x="sold",
        y="return_rate",
        size="returned",
        color="returned",
        hover_name=label_col,
        color_continuous_scale=["#fee8c8", "#fdbb84", "#e34a33"],
        labels={"sold": "Sold items", "return_rate": "Return rate (%)", "returned": "Returns"},
        title=title,
        hover_data={"sold": ":,.0f", "returned": ":,.0f", "return_rate": ":.1f"},
        custom_data=custom_data,
    )
    fig.update_traces(marker=dict(sizemode="area", sizeref=max(df["returned"].max() / 60, 1)))
    if "return_rate" in df.columns and not df["return_rate"].dropna().empty:
        fig.add_hline(
            y=float(df["return_rate"].median()),
            line_dash="dash",
            line_color="#63736d",
            annotation_text="median RR",
            annotation_position="top left",
        )
    if "sold" in df.columns and not df["sold"].dropna().empty:
        fig.add_vline(
            x=float(df["sold"].median()),
            line_dash="dot",
            line_color="#8a9a94",
            annotation_text="median volume",
            annotation_position="top right",
        )
    return apply_layout(fig, 430)


def average_price_return_rate_scatter(df: pd.DataFrame, label_col: str = "Article variant") -> go.Figure:
    if df.empty or "avg_net_price" not in df.columns:
        return apply_layout(go.Figure(), 520)

    size_col = "estimated_returned_nmv"
    if size_col not in df.columns or df[size_col].fillna(0).max() <= 0:
        size_col = "returned"
    color_col = "price_index_vs_category" if "price_index_vs_category" in df.columns else "return_rate"
    hover_data = {
        "Category": True,
        "Article type": True,
        "sold": ":,.0f",
        "returned": ":,.0f",
        "return_rate": ":.1f",
        "avg_net_price": ":,.2f",
        "category_avg_net_price": ":,.2f",
        "price_index_vs_category": ":.1f",
        "estimated_returned_nmv": ":,.0f",
    }
    hover_data = {column: value for column, value in hover_data.items() if column in df.columns}
    custom_data = [label_col] if label_col in df.columns else None
    fig = px.scatter(
        df,
        x="avg_net_price",
        y="return_rate",
        size=size_col,
        color=color_col,
        hover_name=label_col,
        color_continuous_scale=["#74c69d", "#f9c74f", "#d00000"],
        labels={
            "avg_net_price": "Average net price",
            "return_rate": "Return rate (%)",
            "price_index_vs_category": "Price index vs category",
            "estimated_returned_nmv": "Estimated returned NMV",
            "returned": "Returns",
        },
        title="Average net price vs return rate",
        hover_data=hover_data,
        custom_data=custom_data,
    )
    if "category_avg_net_price" in df.columns and not df["category_avg_net_price"].dropna().empty:
        fig.add_vline(x=float(df["category_avg_net_price"].median()), line_dash="dash", line_color="#63736d")
    return apply_layout(fig, 540)


def lifecycle_stage_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return apply_layout(go.Figure(), 520)
    color_col = "early_warning_priority" if "early_warning_priority" in df.columns else "lifecycle_stage"
    custom_data = ["Article variant"] if "Article variant" in df.columns else None
    hover_data = {
        "Category": True,
        "Article type": True,
        "sold": ":,.0f",
        "returned": ":,.0f",
        "return_rate": ":.1f",
        "gap_vs_category": ":.1f",
        "days_online": ":,.0f",
        "dominant_reason": True,
        "early_warning_score": ":.0f",
    }
    hover_data = {column: value for column, value in hover_data.items() if column in df.columns}
    fig = px.scatter(
        df,
        x="days_online",
        y="return_rate",
        size="returned",
        color=color_col,
        hover_name="Article variant" if "Article variant" in df.columns else None,
        labels={
            "days_online": "Days online",
            "return_rate": "Return rate (%)",
            "returned": "Returns",
            "early_warning_priority": "Priority",
            "lifecycle_stage": "Lifecycle stage",
        },
        title="Lifecycle: age vs return rate",
        hover_data=hover_data,
        custom_data=custom_data,
    )
    return apply_layout(fig, 540)


def country_report_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return apply_layout(go.Figure(), 460)
    data = df.sort_values("returned", ascending=True).tail(20)
    fig = px.bar(
        data,
        x="returned",
        y="Country",
        orientation="h",
        color="gap_vs_dataset",
        color_continuous_scale=["#52b788", "#f9c74f", "#d00000"],
        labels={"returned": "Returned items", "gap_vs_dataset": "Gap vs dataset (p.p.)"},
        title="Country report: return volume and deviation",
        hover_data={
            "sold": ":,.0f",
            "return_rate": ":.1f",
            "top_reason": True,
            "top_reason_share": ":.1f",
            "high_risk_variants": ":,.0f",
        },
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#63736d")
    return apply_layout(fig, 470)


def size_fit_intelligence_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return apply_layout(go.Figure(), 520)
    fig = px.scatter(
        df,
        x="size_balance",
        y="size_return_rate",
        size="size_returns",
        color="fit_issue",
        hover_name="Article variant",
        labels={
            "size_balance": "Size skew: too big < 0 / too small > 0",
            "size_return_rate": "Size return rate (%)",
            "size_returns": "Size returns",
            "fit_issue": "Fit issue",
        },
        title="Size & fit intelligence",
        hover_data={
            "Category": True,
            "Article type": True,
            "sold": ":,.0f",
            "returned": ":,.0f",
            "return_rate": ":.1f",
            "size_share_of_returns": ":.1f",
            "gap_vs_category": ":.1f",
        },
        custom_data=["Article variant"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#63736d")
    return apply_layout(fig, 540)


def quality_supplier_chart(df: pd.DataFrame, dimension: str) -> go.Figure:
    if df.empty or dimension not in df.columns:
        return apply_layout(go.Figure(), 460)
    data = df.sort_values("quality_returns", ascending=True).tail(20)
    fig = px.bar(
        data,
        x="quality_returns",
        y=dimension,
        orientation="h",
        color="quality_share_of_returns",
        color_continuous_scale=["#d8f3dc", "#f9c74f", "#d00000"],
        labels={
            "quality_returns": "Quality/supplier returns",
            "quality_share_of_returns": "Quality share of returns (%)",
            dimension: dimension,
        },
        title=f"Quality / supplier report by {dimension}",
        hover_data={
            "sold": ":,.0f",
            "returned": ":,.0f",
            "return_rate": ":.1f",
            "top_quality_reason": True,
            "risk_score": ":.0f",
        },
    )
    return apply_layout(fig, 500)


def stacked_reasons(reason_dim_df: pd.DataFrame, dimension: str, top_n: int = 10) -> go.Figure:
    top_dimensions = (
        reason_dim_df.groupby(dimension)["estimated_returns"].sum().sort_values(ascending=False).head(top_n).index
    )
    data = reason_dim_df[reason_dim_df[dimension].isin(top_dimensions)]
    fig = px.bar(
        data,
        x=dimension,
        y="estimated_returns",
        color="reason",
        labels={"estimated_returns": "Estimated returned items", dimension: dimension, "reason": "Reason"},
        title=f"Return reason structure: {dimension}",
        hover_data={"reason_share": ":.1f", "estimated_returns": ":,.0f", "returned": ":,.0f"},
    )
    fig.update_layout(barmode="stack")
    return apply_layout(fig, 500)


def reason_heatmap(reason_dim_df: pd.DataFrame, dimension: str, top_n: int = 12) -> go.Figure:
    top_dimensions = (
        reason_dim_df.groupby(dimension)["estimated_returns"].sum().sort_values(ascending=False).head(top_n).index
    )
    data = reason_dim_df[reason_dim_df[dimension].isin(top_dimensions)]
    matrix = data.pivot_table(index=dimension, columns="reason", values="reason_share", aggfunc="sum").fillna(0)
    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale=["#f7fcf5", "#74c69d", "#1b4332"],
        labels=dict(x="Reason", y=dimension, color="Share (%)"),
        title=f"Return reason share heatmap: {dimension}",
    )
    fig.update_traces(hovertemplate=f"{dimension}: %{{y}}<br>Reason: %{{x}}<br>Share: %{{z:.1f}}%<extra></extra>")
    return apply_layout(fig, 500)


def product_reason_ranking_chart(
    df: pd.DataFrame,
    metric_col: str,
    reason: str,
    top_n: int = 30,
) -> go.Figure:
    if df.empty or metric_col not in df.columns:
        return apply_layout(go.Figure(), 520)

    label_map = {
        "reason_share_of_returns": "Reason share of returns (%)",
        "reason_gap_vs_category": "Gap vs category (p.p.)",
        "reason_gap_vs_dataset": "Gap vs full data (p.p.)",
        "reason_gap_vs_product_average": "Gap vs avg product (p.p.)",
        "selected_reason_returns": "Estimated reason returns",
        "selected_reason_return_rate": "Reason return rate (%)",
    }
    data = df.sort_values(metric_col, ascending=True).tail(top_n)
    hover_data = {
        "Category": True,
        "Article type": True,
        "sold": ":,.0f",
        "returned": ":,.0f",
        "return_rate": ":.1f",
        "avg_net_price": ":,.2f",
        "price_index_vs_category": ":.1f",
        "estimated_returned_nmv": ":,.0f",
        "selected_reason_returns": ":,.0f",
        "reason_returned_nmv": ":,.0f",
        "reason_share_of_returns": ":.1f",
        "category_reason_share": ":.1f",
        "dataset_reason_share": ":.1f",
        "avg_product_reason_share": ":.1f",
    }
    hover_data = {column: value for column, value in hover_data.items() if column in data.columns}
    fig = px.bar(
        data,
        x=metric_col,
        y="Article variant",
        orientation="h",
        color=metric_col,
        color_continuous_scale=["#d8f3dc", "#f9c74f", "#d00000"],
        labels={metric_col: label_map.get(metric_col, metric_col), "Article variant": "Article variant"},
        title=f"Top variants for return reason: {reason}",
        hover_name="Article variant",
        hover_data=hover_data,
        custom_data=["Article variant"],
    )
    if "gap" in metric_col:
        fig.add_vline(x=0, line_dash="dash", line_color="#63736d")
    return apply_layout(fig, 560)


def size_reason_chart(df: pd.DataFrame, dimension: str) -> go.Figure:
    data = (
        df.groupby(dimension, dropna=False)
        .agg(
            sold=("Sold articles", "sum"),
            returned=("Returned articles", "sum"),
            too_big=("Estimated returns - Item is too big", "sum"),
            too_small=("Estimated returns - Item is too small", "sum"),
        )
        .reset_index()
    )
    data = data[data["returned"] > 0].sort_values("returned", ascending=False).head(12)
    long = data.melt(
        id_vars=[dimension, "sold", "returned"],
        value_vars=["too_big", "too_small"],
        var_name="size_reason",
        value_name="estimated_returns",
    )
    long["size_reason"] = long["size_reason"].replace({"too_big": "Too big", "too_small": "Too small"})
    fig = px.bar(
        long,
        x=dimension,
        y="estimated_returns",
        color="size_reason",
        barmode="group",
        labels={"estimated_returns": "Estimated returned items", "size_reason": "Problem"},
        title=f"Sizing issues by: {dimension}",
        hover_data={"sold": ":,.0f", "returned": ":,.0f"},
    )
    return apply_layout(fig, 430)


def treemap(df: pd.DataFrame) -> go.Figure:
    fig = px.treemap(
        df,
        path=["Category", "Article type"],
        values="returned",
        color="return_rate",
        color_continuous_scale=["#d8f3dc", "#f9c74f", "#d00000"],
        labels={"returned": "Returns", "return_rate": "Return rate (%)"},
        title="Category and article type map",
        hover_data={"sold": ":,.0f", "returned": ":,.0f", "return_rate": ":.1f"},
    )
    return apply_layout(fig, 520)


def action_list_chart(df: pd.DataFrame) -> go.Figure:
    data = df.head(40).copy()
    fig = px.scatter(
        data,
        x="gap_vs_category",
        y="dominant_reason_returns",
        size="returned",
        color="priority_score",
        hover_name="Article variant",
        color_continuous_scale=["#d8f3dc", "#f9c74f", "#d00000"],
        labels={
            "gap_vs_category": "Gap vs category (p.p.)",
            "dominant_reason_returns": "Dominant reason returns",
            "priority_score": "Priority score",
            "returned": "Returns",
        },
        title="Action priorities: impact vs category deviation",
        hover_data={
            "Article type": True,
            "sold": ":,.0f",
            "returned": ":,.0f",
            "return_rate": ":.1f",
            "dominant_reason": True,
        },
        custom_data=["Article variant"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#63736d")
    return apply_layout(fig, 470)


def benchmark_gap_chart(df: pd.DataFrame, gap_col: str = "gap_vs_category") -> go.Figure:
    data = df.sort_values(gap_col, ascending=True).tail(25)
    fig = px.bar(
        data,
        x=gap_col,
        y="Article variant",
        orientation="h",
        color=gap_col,
        color_continuous_scale=["#74c69d", "#f9c74f", "#d00000"],
        labels={gap_col: "Gap (p.p.)", "Article variant": "Article variant"},
        title="Largest return rate deviations from benchmark",
        hover_data={
            "Category": True,
            "Article type": True,
            "sold": ":,.0f",
            "returned": ":,.0f",
            "return_rate": ":.1f",
            "category_return_rate": ":.1f",
            "type_return_rate": ":.1f",
        },
        custom_data=["Article variant"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#63736d")
    return apply_layout(fig, 560)


def segmentation_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="sold",
        y="return_rate",
        size="returned",
        color="segment",
        hover_name="Article variant",
        labels={"sold": "Sold items", "return_rate": "Return rate (%)", "segment": "Segment"},
        title="Variant segmentation: volume vs return rate",
        hover_data={"Category": True, "Article type": True, "returned": ":,.0f"},
        custom_data=["Article variant"],
    )
    if not df.empty:
        fig.add_vline(x=float(df["sold_threshold"].iloc[0]), line_dash="dash", line_color="#63736d")
        fig.add_hline(y=float(df["return_rate_threshold"].iloc[0]), line_dash="dash", line_color="#63736d")
    return apply_layout(fig, 540)


def pareto_chart(df: pd.DataFrame) -> go.Figure:
    data = df.head(200)
    fig = go.Figure()
    fig.add_bar(
        x=data["rank"],
        y=data["returned"],
        name="Returns",
        marker_color="#90a955",
        hovertemplate="Rank %{x}<br>Returns %{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=data["rank"],
        y=data["cumulative_return_share"],
        name="Cumulative return share",
        mode="lines",
        yaxis="y2",
        line=dict(color="#d00000", width=3),
        hovertemplate="Rank %{x}<br>Cumulative share %{y:.1f}%<extra></extra>",
    )
    fig.update_layout(
        title="Returns Pareto by variant",
        xaxis_title="Variant ranking",
        yaxis=dict(title="Returns"),
        yaxis2=dict(title="Cumulative share (%)", overlaying="y", side="right", range=[0, 100]),
    )
    for threshold in [50, 80, 90]:
        fig.add_hline(y=threshold, yref="y2", line_dash="dot", line_color="#63736d")
    return apply_layout(fig, 520)


def quality_bar(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    fig = px.bar(
        data,
        x="metric",
        y="share",
        color="share",
        color_continuous_scale=["#d8f3dc", "#f9c74f", "#d00000"],
        labels={"metric": "Metric", "share": "Share (%)"},
        title="Data quality risks",
        hover_data={"value": ":,.0f", "interpretation": True},
    )
    return apply_layout(fig, 420)


def season_heatmap(df: pd.DataFrame) -> go.Figure:
    matrix = df.pivot_table(index="Season", columns="Article type", values="return_rate", aggfunc="mean").fillna(0)
    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale=["#d8f3dc", "#f9c74f", "#d00000"],
        labels=dict(x="Article type", y="Season", color="Return rate (%)"),
        title="Return rate by season and article type",
    )
    fig.update_traces(hovertemplate="Season: %{y}<br>Type: %{x}<br>Return rate: %{z:.1f}%<extra></extra>")
    return apply_layout(fig, 520)


def simulation_chart(result: dict[str, float]) -> go.Figure:
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Current returns", "Reduction", "After simulation"],
            y=[result["current_returned"], -result["reduced_returns"], result["new_returned"]],
            text=[
                f'{result["current_returned"]:,.0f}',
                f'-{result["reduced_returns"]:,.0f}',
                f'{result["new_returned"]:,.0f}',
            ],
            connector={"line": {"color": "#63736d"}},
            increasing={"marker": {"color": "#d00000"}},
            decreasing={"marker": {"color": "#52b788"}},
            totals={"marker": {"color": "#31572c"}},
        )
    )
    fig.update_layout(title="Simulated impact of return reason reduction", yaxis_title="Returned items")
    return apply_layout(fig, 430)
