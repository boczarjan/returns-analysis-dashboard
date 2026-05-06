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
            "estimated_returns": "Estymowane zwrocone sztuki",
            "reason": "Powód",
            "share_of_returns": "Udział w zwrotach (%)",
        },
        title="Największe powody zwrotów",
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
        labels={"returned": "Zwrócone sztuki", "return_rate": "Return rate (%)"},
        title="Zwroty według kraju",
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
        labels={"sold": "Sprzedane sztuki", "return_rate": "Return rate (%)", "returned": "Zwroty"},
        title=title,
        hover_data={"sold": ":,.0f", "returned": ":,.0f", "return_rate": ":.1f"},
        custom_data=custom_data,
    )
    fig.update_traces(marker=dict(sizemode="area", sizeref=max(df["returned"].max() / 60, 1)))
    return apply_layout(fig, 430)


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
        labels={"estimated_returns": "Estymowane zwrocone sztuki", dimension: dimension, "reason": "Powód"},
        title=f"Struktura powodów zwrotów: {dimension}",
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
        labels=dict(x="Powód", y=dimension, color="Udział (%)"),
        title=f"Heatmapa udziału powodów zwrotów: {dimension}",
    )
    fig.update_traces(hovertemplate=f"{dimension}: %{{y}}<br>Powód: %{{x}}<br>Udział: %{{z:.1f}}%<extra></extra>")
    return apply_layout(fig, 500)


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
    long["size_reason"] = long["size_reason"].replace({"too_big": "Za duży", "too_small": "Za mały"})
    fig = px.bar(
        long,
        x=dimension,
        y="estimated_returns",
        color="size_reason",
        barmode="group",
        labels={"estimated_returns": "Estymowane zwrocone sztuki", "size_reason": "Problem"},
        title=f"Problemy rozmiarowe według: {dimension}",
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
        labels={"returned": "Zwroty", "return_rate": "Return rate (%)"},
        title="Mapa kategorii i typów artykułów",
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
            "gap_vs_category": "Gap vs kategoria (p.p.)",
            "dominant_reason_returns": "Zwroty dominującego powodu",
            "priority_score": "Priority score",
            "returned": "Zwroty",
        },
        title="Priorytety działań: wpływ vs odchylenie od kategorii",
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
        title="Największe odchylenia return rate od benchmarku",
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
        labels={"sold": "Sprzedane sztuki", "return_rate": "Return rate (%)", "segment": "Segment"},
        title="Segmentacja wariantów: wolumen vs return rate",
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
        name="Zwroty",
        marker_color="#90a955",
        hovertemplate="Rank %{x}<br>Zwroty %{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=data["rank"],
        y=data["cumulative_return_share"],
        name="Skumulowany udział zwrotów",
        mode="lines",
        yaxis="y2",
        line=dict(color="#d00000", width=3),
        hovertemplate="Rank %{x}<br>Skumulowany udział %{y:.1f}%<extra></extra>",
    )
    fig.update_layout(
        title="Pareto zwrotów po wariantach",
        xaxis_title="Ranking wariantów",
        yaxis=dict(title="Zwroty"),
        yaxis2=dict(title="Skumulowany udział (%)", overlaying="y", side="right", range=[0, 100]),
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
        labels={"metric": "Metryka", "share": "Udział (%)"},
        title="Ryzyka jakości danych",
        hover_data={"value": ":,.0f", "interpretation": True},
    )
    return apply_layout(fig, 420)


def season_heatmap(df: pd.DataFrame) -> go.Figure:
    matrix = df.pivot_table(index="Season", columns="Article type", values="return_rate", aggfunc="mean").fillna(0)
    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale=["#d8f3dc", "#f9c74f", "#d00000"],
        labels=dict(x="Typ artykułu", y="Sezon", color="Return rate (%)"),
        title="Return rate według sezonu i typu artykułu",
    )
    fig.update_traces(hovertemplate="Sezon: %{y}<br>Typ: %{x}<br>Return rate: %{z:.1f}%<extra></extra>")
    return apply_layout(fig, 520)


def simulation_chart(result: dict[str, float]) -> go.Figure:
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Obecne zwroty", "Redukcja", "Po symulacji"],
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
    fig.update_layout(title="Symulowany wpływ redukcji powodów zwrotów", yaxis_title="Zwrócone sztuki")
    return apply_layout(fig, 430)
