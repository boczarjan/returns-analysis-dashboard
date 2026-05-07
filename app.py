from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from returns_dashboard.charts import (
    action_list_chart,
    benchmark_gap_chart,
    country_bar,
    pareto_chart,
    quality_bar,
    reason_bar,
    reason_heatmap,
    return_rate_scatter,
    season_heatmap,
    segmentation_chart,
    simulation_chart,
    size_reason_chart,
    stacked_reasons,
    treemap,
)
from returns_dashboard.data_loader import (
    DEFAULT_CSV_PATH,
    load_returns_csv,
    load_returns_path_with_parquet,
    reason_columns,
    validate_returns_data,
)
from returns_dashboard.deep_analysis import (
    action_list,
    benchmark_products,
    data_quality_report,
    detect_product_anomalies,
    pareto_breakpoints,
    pareto_products,
    product_profile,
    product_segments,
    season_article_type_analysis,
    simulate_reason_reduction,
)
from returns_dashboard.metrics import aggregate_by, kpi_summary, product_ranking, reason_by_dimension, reason_summary
from returns_dashboard.pdf_report import build_pdf_report


st.set_page_config(
    page_title="Returns Analysis",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
  --ink: #102027;
  --muted: #63736d;
  --panel: rgba(255,255,255,.9);
  --line: rgba(49,87,44,.16);
  --green: #31572c;
  --green-soft: #e9f5ec;
  --amber: #f9c74f;
  --amber-soft: #fff4c7;
  --red: #c1121f;
  --red-soft: #ffe1df;
  --blue-soft: #e5f1ff;
}

.stApp {
  background:
    linear-gradient(135deg, rgba(214,245,221,.92) 0%, rgba(255,250,235,.9) 42%, rgba(234,244,255,.94) 100%);
  color: var(--ink);
}

div[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
  background: rgba(247, 252, 245, .96);
  border-right: 1px solid var(--line);
}

.hero {
  padding: 30px 34px 26px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(120deg, rgba(255,255,255,.88), rgba(248,255,249,.76));
  box-shadow: 0 18px 50px rgba(16, 32, 39, .08);
  margin-bottom: 14px;
}

.hero h1 {
  font-family: Inter, Segoe UI, sans-serif;
  font-size: clamp(2rem, 4vw, 4.2rem);
  line-height: .98;
  letter-spacing: 0;
  margin: 0 0 12px;
  color: #102027;
}

.hero p {
  max-width: 960px;
  margin: 0;
  color: var(--muted);
  font-size: 1.04rem;
}

.context-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  padding: 10px 12px;
  margin: 0 0 16px;
  background: rgba(255,255,255,.72);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.context-pill {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  padding: 5px 9px;
  border-radius: 999px;
  background: var(--green-soft);
  color: var(--ink);
  font-size: .82rem;
  font-weight: 650;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(130px, 1fr));
  gap: 12px;
  margin: 8px 0 18px;
}

.kpi {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 15px 15px 13px;
  box-shadow: 0 10px 28px rgba(16, 32, 39, .06);
}

.kpi span {
  display: block;
  color: var(--muted);
  font-size: .76rem;
  font-weight: 750;
  text-transform: uppercase;
  letter-spacing: 0;
  margin-bottom: 7px;
}

.kpi strong {
  display: block;
  color: var(--ink);
  font-size: clamp(1.25rem, 1.7vw, 1.9rem);
  line-height: 1.05;
}

.kpi small {
  display: block;
  color: var(--muted);
  margin-top: 8px;
  font-size: .78rem;
  line-height: 1.2;
}

.section-title {
  font-family: Inter, Segoe UI, sans-serif;
  font-size: 1.22rem;
  font-weight: 800;
  color: var(--ink);
  margin: 12px 0 6px;
}

.insight-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(160px, 1fr));
  gap: 12px;
  margin: 0 0 18px;
}

.insight-card, .action-card {
  background: rgba(255,255,255,.82);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 15px;
  box-shadow: 0 10px 26px rgba(16,32,39,.06);
}

.insight-card b, .action-card b {
  display: block;
  font-size: 1rem;
  color: var(--ink);
  margin-bottom: 4px;
}

.insight-card span, .action-card span {
  color: var(--muted);
  font-size: .86rem;
}

.action-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(160px, 1fr));
  gap: 10px;
  margin: 4px 0 16px;
}

.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 4px 8px;
  font-size: .74rem;
  font-weight: 750;
  margin: 0 4px 6px 0;
}

.badge-high { background: var(--red-soft); color: var(--red); }
.badge-medium { background: var(--amber-soft); color: #755500; }
.badge-low { background: var(--green-soft); color: var(--green); }
.badge-info { background: var(--blue-soft); color: #14507a; }
.badge-muted { background: #edf1ef; color: #51615b; }

div[data-testid="stMetric"] {
  background: rgba(255,255,255,.78);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}

@media (max-width: 1100px) {
  .action-grid { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
  .insight-grid { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
}

@media (max-width: 900px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
  .hero { padding: 22px 20px; }
  .action-grid, .insight-grid { grid-template-columns: 1fr; }
}
</style>
"""


NAV_GROUPS = {
    "Dashboard": ["Executive summary", "Overview"],
    "Priorities": ["Action list", "Watchlist", "Anomalies", "Benchmarks"],
    "Products": ["Product profile", "Segmentation", "Pareto", "Products"],
    "Return reasons": ["Return reasons", "Sizing", "Seasons", "Simulation"],
    "Data and export": ["Data quality", "Data"],
}


TABLE_PREVIEW_ROWS = 500
VARIANT_SEARCH_LIMIT = 75
WATCHLIST_COLUMNS = [
    "Article variant",
    "priority",
    "status",
    "owner",
    "due_date",
    "problem_type",
    "dominant_reason",
    "return_rate",
    "returned",
    "recommended_action",
    "notes",
]


def file_revision(path: str | Path) -> str:
    source_path = Path(path)
    stat = source_path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


@st.cache_data(show_spinner=False)
def cached_load_from_path(path: str, revision: str) -> pd.DataFrame:
    return load_returns_path_with_parquet(Path(path))


@st.cache_data(show_spinner=False)
def cached_load_from_upload(uploaded_file) -> pd.DataFrame:
    return load_returns_csv(uploaded_file)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_unique_values(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False, max_entries=64)
def cached_kpi_summary(df: pd.DataFrame) -> dict[str, float]:
    return kpi_summary(df)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_aggregate_by(df: pd.DataFrame, group_cols: str | tuple[str, ...]) -> pd.DataFrame:
    cols = list(group_cols) if isinstance(group_cols, tuple) else group_cols
    return aggregate_by(df, cols)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    return reason_summary(df)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_reason_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    return reason_by_dimension(df, dimension)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_product_ranking(df: pd.DataFrame, min_sold: int) -> pd.DataFrame:
    return product_ranking(df, min_sold=min_sold)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_action_list(df: pd.DataFrame, min_sold: int) -> pd.DataFrame:
    return action_list(df, min_sold=min_sold)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_benchmark_products(df: pd.DataFrame, min_sold: int) -> pd.DataFrame:
    return benchmark_products(df, min_sold=min_sold)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_product_anomalies(df: pd.DataFrame, min_sold: int) -> pd.DataFrame:
    return detect_product_anomalies(df, min_sold=min_sold)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_product_profile(df: pd.DataFrame, article_variant: str) -> dict:
    return product_profile(df, article_variant)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_product_segments(df: pd.DataFrame, min_sold: int) -> pd.DataFrame:
    return product_segments(df, min_sold=min_sold)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_pareto_products(df: pd.DataFrame) -> pd.DataFrame:
    return pareto_products(df)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_pareto_breakpoints(pareto_df: pd.DataFrame) -> pd.DataFrame:
    return pareto_breakpoints(pareto_df)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_data_quality_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return data_quality_report(df)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_validation_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return validate_returns_data(df)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_season_article_type_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return season_article_type_analysis(df)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_simulate_reason_reduction(
    df: pd.DataFrame,
    reasons: tuple[str, ...],
    reduction_pct: float,
) -> dict[str, float]:
    return simulate_reason_reduction(df, list(reasons), reduction_pct)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_build_pdf_report(df: pd.DataFrame) -> bytes:
    return build_pdf_report(df)


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def format_pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f} p.p."


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


def dataframe_signature(df: pd.DataFrame) -> str:
    columns = [
        column
        for column in [
            "Article variant",
            "Country",
            "Category",
            "Article type",
            "Season",
            "Sold articles",
            "Returned articles",
            "Return rate (%)",
        ]
        if column in df.columns
    ]
    if not columns:
        return str(len(df))
    hashed = pd.util.hash_pandas_object(df[columns], index=True).sum()
    return f"{len(df)}:{int(hashed)}"


def encode_variant_key(article_variant: str) -> str:
    encoded = base64.urlsafe_b64encode(str(article_variant).encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def decode_variant_key(value: str | None) -> str | None:
    if not value:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception:
        return None


def variant_url(article_variant: object) -> str:
    label = str(article_variant)
    return f"./?variant_key={encode_variant_key(label)}#{label}"


def variant_anchor(article_variant: object) -> str:
    safe_label = html.escape(str(article_variant))
    return f'<a href="{variant_url(article_variant)}">{safe_label}</a>'


def with_variant_links(df: pd.DataFrame) -> pd.DataFrame:
    linked = df.copy()
    if "Article variant" in linked.columns:
        linked["Article variant"] = linked["Article variant"].astype(str).map(variant_url)
    return linked


def variant_column_config(extra: dict | None = None) -> dict:
    config = {
        "Article variant": st.column_config.LinkColumn(
            "Article variant",
            help="Click the code to open this variant analysis.",
            display_text=r"#(.*)$",
            width="medium",
        )
    }
    if extra:
        config.update(extra)
    return config


def get_requested_variant() -> str | None:
    return decode_variant_key(st.query_params.get("variant_key"))


def extract_selected_variant(event) -> str | None:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None

    point = points[0]
    custom_data = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
    if custom_data is not None:
        try:
            return str(custom_data[0])
        except (TypeError, IndexError):
            return str(custom_data)
    return None


def plot_variant_chart(fig, key: str) -> None:
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        on_select="rerun",
        selection_mode="points",
    )
    selected_variant = extract_selected_variant(event)
    if selected_variant:
        st.query_params["variant_key"] = encode_variant_key(selected_variant)
        st.rerun()


def sidebar_data_source() -> pd.DataFrame | None:
    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    fallback_path = st.sidebar.text_input("CSV file path", value=str(DEFAULT_CSV_PATH))

    try:
        if uploaded is not None:
            return cached_load_from_upload(uploaded)
        if fallback_path and Path(fallback_path).exists():
            return cached_load_from_path(fallback_path, file_revision(fallback_path))
        st.sidebar.warning("File not found. Upload a CSV or provide a valid path.")
    except Exception as exc:
        st.sidebar.error(f"Could not load data: {exc}")
    return None


def sidebar_navigation() -> str:
    st.sidebar.header("Navigation")
    group = st.sidebar.radio("Area", list(NAV_GROUPS.keys()), index=0)
    return st.sidebar.radio("View", NAV_GROUPS[group], index=0)


def multiselect_filter(df: pd.DataFrame, column: str, label: str) -> list[str]:
    if column not in df.columns:
        return []
    options = cached_unique_values(df, column)
    return st.sidebar.multiselect(label, options=options)


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    active_filters: dict[str, object] = {}
    filtered = df.copy()

    with st.sidebar.expander("Filters", expanded=True):
        if "Article variant" in df.columns:
            all_variants = cached_unique_values(df, "Article variant")
            variant_query = st.text_input(
                "Article variant",
                placeholder="Enter part of the variant code...",
            ).strip()
            variant_options = all_variants
            if variant_query:
                query = variant_query.lower()
                variant_options = [variant for variant in all_variants if query in variant.lower()]
                st.caption(
                    f"Found {len(variant_options)} matches. "
                    f"The picker shows up to {VARIANT_SEARCH_LIMIT}."
                )
            visible_variant_options = variant_options[:VARIANT_SEARCH_LIMIT]
            selected_variants = st.multiselect(
                "Select variants",
                options=visible_variant_options,
                placeholder="Enter part of the code first...",
            )
            if selected_variants:
                filtered = filtered[filtered["Article variant"].astype(str).isin(selected_variants)]
                active_filters["Article variant"] = selected_variants

        for column, label in [
            ("Country", "Country"),
            ("Category", "Category"),
            ("Article type", "Article type"),
            ("Season", "Season"),
            ("Gender", "Gender"),
            ("Estimated return rate status", "Estimation status"),
            ("Article visibility", "Visibility"),
        ]:
            selected = multiselect_filter(df, column, label)
            if selected:
                filtered = filtered[filtered[column].astype(str).isin(selected)]
                active_filters[label] = selected

        max_sold = int(max(df["Sold articles"].max(), 1))
        min_sold = st.slider("Minimum sold articles per row", 0, max_sold, 0)
        if min_sold > 0:
            filtered = filtered[filtered["Sold articles"].fillna(0) >= min_sold]
            active_filters["Min. sold"] = min_sold

    return filtered, active_filters


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_filter_context(df: pd.DataFrame, active_filters: dict[str, object]) -> None:
    summary = cached_kpi_summary(df)
    if active_filters:
        filters = []
        for key, value in active_filters.items():
            if isinstance(value, list):
                preview = ", ".join(map(str, value[:3]))
                suffix = f" +{len(value) - 3}" if len(value) > 3 else ""
                filters.append(f"{key}: {preview}{suffix}")
            else:
                filters.append(f"{key}: {value}")
        filter_text = " | ".join(filters)
    else:
        filter_text = "No active filters"

    st.markdown(
        f"""
        <div class="context-bar">
          <span class="context-pill">Filters: {html.escape(filter_text)}</span>
          <span class="context-pill">Rows: {format_number(len(df))}</span>
          <span class="context-pill">Sold: {format_number(summary["sold"])}</span>
          <span class="context-pill">Returned: {format_number(summary["returned"])}</span>
          <span class="context-pill">Weighted RR: {format_percent(summary["return_rate"])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame, baseline_df: pd.DataFrame | None = None) -> None:
    summary = cached_kpi_summary(df)
    baseline = cached_kpi_summary(baseline_df)["return_rate"] if baseline_df is not None and not baseline_df.empty else None
    delta = format_pp(summary["return_rate"] - baseline) if baseline is not None else "no benchmark"
    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi"><span>Sold articles</span><strong>{format_number(summary["sold"])}</strong><small>Volume after filters.</small></div>
          <div class="kpi"><span>Returned articles</span><strong>{format_number(summary["returned"])}</strong><small>Number of returned items.</small></div>
          <div class="kpi"><span>Weighted return rate</span><strong>{format_percent(summary["return_rate"])}</strong><small>{delta} vs full dataset.</small></div>
          <div class="kpi"><span>Average return rate</span><strong>{format_percent(summary["average_return_rate"])}</strong><small>Simple row average.</small></div>
          <div class="kpi"><span>NMV</span><strong>{format_number(summary["nmv"])}</strong><small>Sales value in the data.</small></div>
          <div class="kpi"><span>Variants</span><strong>{format_number(summary["variants"])}</strong><small>Unique Article variants.</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classify_problem(reason: str) -> str:
    text = str(reason).lower()
    if "too small" in text or "too big" in text:
        return "Size"
    if "described" in text:
        return "Description"
    if "damaged" in text or "wrong item" in text:
        return "Quality/ops"
    if "late" in text:
        return "Delivery"
    if "expensive" in text:
        return "Price"
    if "no details" in text:
        return "Unknown"
    return "Preference"


def priority_level_from_rank(index: int, total: int) -> str:
    if total <= 0:
        return "Low"
    pct = (index + 1) / total
    if pct <= 0.2:
        return "High"
    if pct <= 0.55:
        return "Medium"
    return "Low"


def enrich_actions(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return actions
    enriched = actions.copy().reset_index(drop=True)
    enriched["priority"] = [priority_level_from_rank(i, len(enriched)) for i in range(len(enriched))]
    enriched["problem_type"] = enriched["dominant_reason"].map(classify_problem)
    enriched["confidence"] = enriched["confidence_flag"].map(lambda value: "OK" if value == "OK" else "Use caution")
    enriched["impact"] = enriched["returned"].map(lambda value: "High" if value >= enriched["returned"].quantile(0.75) else "Medium")
    return enriched


def badge(label: str, kind: str = "info") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(str(label))}</span>'


def priority_badge(priority: str) -> str:
    kind = {"High": "high", "Medium": "medium", "Low": "low"}.get(priority, "info")
    return badge(priority, kind)


def render_guidance(title: str, text: str) -> None:
    with st.expander(title, expanded=False):
        st.write(text)


def render_top_action_cards(actions: pd.DataFrame, limit: int = 5) -> None:
    if actions.empty:
        st.info("No actions for the current filters.")
        return
    cards = []
    for _, row in actions.head(limit).iterrows():
        reason = html.escape(str(row["dominant_reason"]))
        returned = format_number(row["returned"])
        return_rate = format_percent(row["return_rate"])
        cards.append(
            '<div class="action-card">'
            f'{priority_badge(row["priority"])}'
            f'{badge(row["problem_type"], "info")}'
            f'<b>{variant_anchor(row["Article variant"])}</b>'
            f'<span>{reason} | {returned} returns | RR {return_rate}</span>'
            '</div>'
        )
    st.markdown(f'<div class="action-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_insight_cards(df: pd.DataFrame) -> None:
    reasons = cached_reason_summary(df)
    countries = cached_aggregate_by(df, "Country")
    types = cached_aggregate_by(df, "Article type")
    if reasons.empty or countries.empty or types.empty:
        return
    top_reason = reasons.iloc[0]
    top_country = countries.iloc[0]
    top_type = types.iloc[0]
    st.markdown(
        f"""
        <div class="insight-grid">
          <div class="insight-card"><b>Top reason</b><span>{html.escape(str(top_reason["reason"]))} | {format_number(top_reason["estimated_returns"])} est. items</span></div>
          <div class="insight-card"><b>Top market</b><span>{html.escape(str(top_country["Country"]))} | RR {format_percent(top_country["return_rate"])}</span></div>
          <div class="insight-card"><b>Top article type</b><span>{html.escape(str(top_type["Article type"]))} | {format_number(top_type["returned"])} returns</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pdf_export(
    df: pd.DataFrame,
    key: str = "filtered_report",
    filename_prefix: str = "returns_report",
    generate_label: str = "Generate PDF for current filters",
    download_label: str = "Download generated PDF",
) -> None:
    state_key = f"{key}_pdf_bytes"
    signature_key = f"{key}_pdf_signature"
    filename_key = f"{key}_pdf_filename"

    if state_key in st.session_state:
        current_signature = dataframe_signature(df)
        if st.session_state.get(signature_key) != current_signature:
            st.session_state.pop(state_key, None)
            st.session_state.pop(signature_key, None)
            st.session_state.pop(filename_key, None)

    if st.button(generate_label, key=f"{key}_generate_pdf", type="primary", width="stretch"):
        with st.spinner("Generating PDF..."):
            st.session_state[state_key] = cached_build_pdf_report(df)
            st.session_state[signature_key] = dataframe_signature(df)
            st.session_state[filename_key] = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

    if state_key in st.session_state:
        st.download_button(
            download_label,
            data=st.session_state[state_key],
            file_name=st.session_state[filename_key],
            mime="application/pdf",
            width="stretch",
        )


def render_executive_summary(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">What to fix first</div>', unsafe_allow_html=True)
    actions = enrich_actions(cached_action_list(df, min_sold=30))
    render_top_action_cards(actions)

    sim = cached_simulate_reason_reduction(df, ("Item is too small", "Item is too big"), 10)
    pareto = cached_pareto_products(df)
    breakpoints = cached_pareto_breakpoints(pareto)
    pareto_80 = breakpoints[breakpoints["return_share_threshold"].eq(80)].head(1)
    pareto_text = "no data"
    if not pareto_80.empty:
        row = pareto_80.iloc[0]
        pareto_text = f'{format_number(row["variants_needed"])} variants ({format_percent(row["variant_share"])})'

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Size effect -10%", f'-{format_number(sim["reduced_returns"])} returns', f'-{abs(sim["return_rate_delta"]):.1f} p.p.')
    col_b.metric("Pareto 80% of returns", pareto_text)
    col_c.metric("Action list", f'{format_number(len(actions))} variants', "min. 30 sold")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(reason_bar(cached_reason_summary(df)), width="stretch")
    with right:
        st.plotly_chart(country_bar(cached_aggregate_by(df, "Country").head(14)), width="stretch")

    with st.expander("Details: top action list", expanded=False):
        st.dataframe(
            with_variant_links(actions.head(50)),
            width="stretch",
            hide_index=True,
            column_config=variant_column_config(action_column_config()),
        )


def action_column_config() -> dict:
    return {
        "priority": st.column_config.TextColumn("priority", help="High = highest priority in the current dataset."),
        "problem_type": st.column_config.TextColumn("problem type", help="Problem type inferred from the dominant return reason."),
        "confidence": st.column_config.TextColumn("confidence", help="Use caution = low volume or unstable status."),
        "return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f", help="Returns / sales * 100."),
        "category_return_rate": st.column_config.NumberColumn("category RR (%)", format="%.1f"),
        "gap_vs_category": st.column_config.NumberColumn("gap vs category", format="%.1f"),
        "dominant_reason_returns": st.column_config.NumberColumn("dominant reason returns", format="%.0f"),
        "dominant_reason_share": st.column_config.NumberColumn("dominant reason share (%)", format="%.1f"),
        "priority_score": st.column_config.NumberColumn("priority score", format="%.0f", help="Excess return volume + dominant reason strength + gap vs category."),
    }


def watchlist_column_config() -> dict:
    config = variant_column_config(
        {
            "priority": st.column_config.SelectboxColumn("priority", options=["High", "Medium", "Low"]),
            "status": st.column_config.SelectboxColumn(
                "status",
                options=["New", "In review", "Action planned", "Done", "Ignored"],
            ),
            "owner": st.column_config.TextColumn("owner"),
            "due_date": st.column_config.TextColumn("due date"),
            "problem_type": st.column_config.TextColumn("problem type"),
            "return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f"),
            "returned": st.column_config.NumberColumn("returned", format="%.0f"),
            "notes": st.column_config.TextColumn("notes", width="large"),
        }
    )
    return config


def get_watchlist() -> pd.DataFrame:
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = pd.DataFrame(columns=WATCHLIST_COLUMNS)
    watchlist = st.session_state["watchlist"].copy()
    for column in WATCHLIST_COLUMNS:
        if column not in watchlist.columns:
            watchlist[column] = ""
    return watchlist[WATCHLIST_COLUMNS]


def save_watchlist(watchlist: pd.DataFrame) -> None:
    clean = watchlist.copy()
    if "Article variant" in clean.columns:
        clean = clean[clean["Article variant"].astype(str).str.strip().ne("")]
        clean = clean.drop_duplicates("Article variant", keep="last")
    st.session_state["watchlist"] = clean[WATCHLIST_COLUMNS]


def add_variants_to_watchlist(actions: pd.DataFrame, variants: list[str]) -> int:
    if not variants:
        return 0
    current = get_watchlist()
    existing = set(current["Article variant"].astype(str))
    rows = []
    for _, row in actions[actions["Article variant"].astype(str).isin(variants)].iterrows():
        variant = str(row["Article variant"])
        if variant in existing:
            continue
        rows.append(
            {
                "Article variant": variant,
                "priority": row.get("priority", "Medium"),
                "status": "New",
                "owner": "",
                "due_date": "",
                "problem_type": row.get("problem_type", ""),
                "dominant_reason": row.get("dominant_reason", ""),
                "return_rate": row.get("return_rate", 0.0),
                "returned": row.get("returned", 0.0),
                "recommended_action": row.get("recommended_action", ""),
                "notes": "",
            }
        )
    if not rows:
        return 0
    save_watchlist(pd.concat([current, pd.DataFrame(rows)], ignore_index=True))
    return len(rows)


def render_watchlist_editor() -> None:
    watchlist = get_watchlist()
    if watchlist.empty:
        st.info("The watchlist is empty. Add variants from the Action list view.")
        return

    edited = st.data_editor(
        with_variant_links(watchlist),
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config=watchlist_column_config(),
        key="watchlist_editor",
    )
    edited = edited.copy()
    if "Article variant" in edited.columns:
        edited["Article variant"] = edited["Article variant"].astype(str).str.split("#").str[-1]
    save_watchlist(edited)

    st.download_button(
        "Download watchlist CSV",
        data=dataframe_to_csv_bytes(get_watchlist()),
        file_name=f"returns_watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        width="stretch",
    )


def render_overview(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read Overview",
        "This view answers where the highest return volume appears and which dimensions have the highest return rate. The bubble chart shows the relationship between sales volume and return rate.",
    )
    reason_df = cached_reason_summary(df)
    countries = cached_aggregate_by(df, "Country").head(14)
    article_types = cached_aggregate_by(df, "Article type").head(20)
    category_type = cached_aggregate_by(df, ("Category", "Article type"))

    left, right = st.columns((1.1, 1))
    with left:
        st.plotly_chart(reason_bar(reason_df), width="stretch")
    with right:
        st.plotly_chart(country_bar(countries), width="stretch")

    left, right = st.columns((1, 1))
    with left:
        st.plotly_chart(return_rate_scatter(article_types, "Article type", "Return rate vs volume: article type"), width="stretch")
    with right:
        st.plotly_chart(treemap(category_type), width="stretch")


def render_reasons(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read return reasons",
        "Reasons are counted as estimated items: Returned articles * reason %. This is better than comparing percentages alone because it accounts for the scale of the problem.",
    )
    dimension = st.selectbox("Return reason analysis dimension", ["Country", "Article type", "Category", "Season", "Gender"], index=0)
    reason_dim = cached_reason_by_dimension(df, dimension)

    left, right = st.columns((1, 1))
    with left:
        st.plotly_chart(stacked_reasons(reason_dim, dimension), width="stretch")
    with right:
        st.plotly_chart(reason_heatmap(reason_dim, dimension), width="stretch")

    with st.expander("Details: reasons table", expanded=False):
        pivot = (
            reason_dim.pivot_table(index=dimension, columns="reason", values="estimated_returns", aggfunc="sum")
            .fillna(0)
            .round(0)
            .sort_index()
        )
        st.dataframe(pivot, width="stretch")


def render_size(df: pd.DataFrame) -> None:
    required = {"Estimated returns - Item is too big", "Estimated returns - Item is too small"}
    if not required.issubset(df.columns):
        st.info("The data does not contain the full set of sizing columns.")
        return

    render_guidance(
        "How to read sizing",
        "A 'too small' skew suggests undersized sizing or weak fit communication. A 'too big' skew suggests oversized sizing or unclear cut communication.",
    )
    dimension = st.selectbox("Sizing issue dimension", ["Article type", "Country", "Category", "Gender", "Season"])
    st.plotly_chart(size_reason_chart(df, dimension), width="stretch")

    ranking = cached_product_ranking(df, min_sold=10).copy()
    ranking["estimated_size_returns"] = (
        df.groupby("Article variant")["Estimated returns - Item is too big"].sum()
        + df.groupby("Article variant")["Estimated returns - Item is too small"].sum()
    ).reindex(ranking["Article variant"]).values
    ranking = ranking.sort_values("estimated_size_returns", ascending=False).head(80)

    with st.expander("Details: products with sizing issues", expanded=False):
        display = with_variant_links(ranking[[
            "Article variant", "Zalando article variant", "Category", "Article type", "sold", "returned", "return_rate", "estimated_size_returns"
        ]])
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config=variant_column_config({
                "return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f"),
                "estimated_size_returns": st.column_config.NumberColumn("est. size returns", format="%.0f"),
            }),
        )


def render_products(df: pd.DataFrame) -> None:
    min_sold = st.slider("Minimum variant sales", 1, 500, 30)
    ranking = cached_product_ranking(df, min_sold=min_sold).head(250)
    plot_variant_chart(return_rate_scatter(ranking.head(80), "Article variant", "Variants: return rate vs return volume"), key="products_variant_scatter")

    with st.expander("Details: variant ranking", expanded=False):
        st.dataframe(
            with_variant_links(ranking),
            width="stretch",
            hide_index=True,
            column_config=variant_column_config({
                "sold": st.column_config.NumberColumn("sold", format="%.0f"),
                "returned": st.column_config.NumberColumn("returned", format="%.0f"),
                "nmv": st.column_config.NumberColumn("NMV", format="%.0f"),
                "return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f"),
                "return_gap_vs_dataset": st.column_config.NumberColumn("gap vs dataset", format="%.1f"),
            }),
        )


def render_action_list(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read Action list",
        "This is the app's decision center. Priority combines return volume, deviation from category, and the strength of the dominant reason. Click a variant in the table or a point on the chart to open the variant analysis.",
    )
    min_sold = st.slider("Minimum sales for action list", 1, 500, 30)
    actions = enrich_actions(cached_action_list(df, min_sold=min_sold))
    if actions.empty:
        st.info("No products meet the minimum sales threshold.")
        return

    render_top_action_cards(actions)
    plot_variant_chart(action_list_chart(actions), key="action_list_variant_scatter")

    with st.expander("Add to watchlist", expanded=False):
        selected_for_watchlist = st.multiselect(
            "Variants to track",
            options=actions["Article variant"].astype(str).head(200).tolist(),
            placeholder="Select variants from the action list...",
        )
        if st.button("Add selected variants", width="stretch"):
            added = add_variants_to_watchlist(actions, selected_for_watchlist)
            if added:
                st.success(f"Added {added} variants to the watchlist.")
            else:
                st.info("No new variants were added.")

    st.download_button(
        "Download action list CSV",
        data=dataframe_to_csv_bytes(actions),
        file_name=f"returns_action_list_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Details: full action list", expanded=True):
        st.dataframe(
            with_variant_links(actions.head(200)),
            width="stretch",
            hide_index=True,
            column_config=variant_column_config(action_column_config()),
        )


def render_watchlist(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read the watchlist",
        "The watchlist is a working tracker for variants that need a decision. Status, owner, due date, and notes are stored in the current app session.",
    )
    watchlist = get_watchlist()
    if not watchlist.empty:
        variants = watchlist["Article variant"].astype(str).tolist()
        tracked_df = df[df["Article variant"].astype(str).isin(variants)]
        if not tracked_df.empty:
            render_kpis(tracked_df, df)
    render_watchlist_editor()


def render_anomalies(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read anomalies",
        "This view detects variants that stand out from the category or full dataset, have high return volume, or show a strong concentration in one reason. Unstable data lowers the score but remains visible as a flag.",
    )
    min_sold = st.slider("Minimum sales for anomaly detection", 1, 500, 30)
    anomalies = cached_product_anomalies(df, min_sold=min_sold)
    if anomalies.empty:
        st.info("No anomalies for the current filters and sales threshold.")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Anomalies", format_number(len(anomalies)))
    col_b.metric("Returns in anomalies", format_number(anomalies["returned"].sum()))
    col_c.metric("Avg. gap vs category", format_pp(anomalies["gap_vs_category"].mean()))

    plot_variant_chart(
        return_rate_scatter(anomalies.head(80), "Article variant", "Anomalies: return rate vs volume"),
        key="anomalies_variant_scatter",
    )
    st.download_button(
        "Download anomalies CSV",
        data=dataframe_to_csv_bytes(anomalies),
        file_name=f"returns_anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        width="stretch",
    )
    st.dataframe(
        with_variant_links(anomalies.head(250)),
        width="stretch",
        hide_index=True,
        column_config=variant_column_config(
            {
                "return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f"),
                "gap_vs_category": st.column_config.NumberColumn("gap vs category", format="%.1f"),
                "gap_vs_dataset": st.column_config.NumberColumn("gap vs dataset", format="%.1f"),
                "dominant_reason_share": st.column_config.NumberColumn("dominant reason share (%)", format="%.1f"),
                "anomaly_score": st.column_config.NumberColumn("anomaly score", format="%.0f"),
            }
        ),
    )


def render_benchmarks(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read benchmarks",
        "A positive gap means the variant has a higher return rate than the benchmark. Variants with a high gap and high return volume are the most suspicious.",
    )
    min_sold = st.slider("Minimum sales for benchmarks", 1, 500, 30)
    products = cached_benchmark_products(df, min_sold=min_sold)
    if products.empty:
        st.info("No products meet the minimum sales threshold.")
        return

    gap_col = st.radio(
        "Product benchmark",
        ["gap_vs_category", "gap_vs_type", "gap_vs_dataset"],
        horizontal=True,
        format_func={"gap_vs_category": "vs category", "gap_vs_type": "vs article type", "gap_vs_dataset": "vs full dataset"}.get,
    )
    plot_variant_chart(benchmark_gap_chart(products, gap_col), key=f"benchmark_{gap_col}")

    col_a, col_b = st.columns(2)
    with col_a:
        dimension = st.selectbox("Dimension benchmark", ["Country", "Category", "Article type", "Season"])
        dim_df = cached_aggregate_by(df, dimension)
        dim_df["gap_vs_dataset"] = dim_df["return_rate"] - cached_kpi_summary(df)["return_rate"]
        st.dataframe(
            dim_df.head(80),
            width="stretch",
            hide_index=True,
            column_config={
                "return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f"),
                "gap_vs_dataset": st.column_config.NumberColumn("gap vs dataset", format="%.1f"),
            },
        )
    with col_b:
        st.dataframe(
            with_variant_links(products.head(80)),
            width="stretch",
            hide_index=True,
            column_config=variant_column_config(action_column_config()),
        )


def mode_value(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns or df[column].dropna().empty:
        return "Unknown"
    mode = df[column].mode()
    return str(mode.iat[0]) if not mode.empty else "Unknown"


def render_product_profile(df: pd.DataFrame) -> None:
    variants = cached_unique_values(df, "Article variant")
    if not variants:
        st.info("No variants in the current filter.")
        return
    selected = st.selectbox("Select Article variant", variants)
    st.link_button("Open as variant page", variant_url(selected), width="stretch")
    render_variant_analysis(df, selected, full_page=False)


def render_variant_analysis(df: pd.DataFrame, article_variant: str, full_page: bool) -> None:
    profile = cached_product_profile(df, article_variant)
    if not profile:
        st.error(f"Variant not found: {article_variant}")
        return

    summary = profile["summary"]
    raw = profile["raw"]
    if full_page:
        st.link_button("Back to dashboard", "./", width="content")
        render_hero(article_variant, "Variant mini-report: KPIs, benchmarks, return reasons, countries, seasons, and source data.")

    meta = f'{mode_value(raw, "Category")} | {mode_value(raw, "Article type")} | {mode_value(raw, "Gender")} | {mode_value(raw, "Season")}'
    st.markdown(f'<div class="context-bar"><span class="context-pill">{html.escape(meta)}</span></div>', unsafe_allow_html=True)

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Sold", format_number(summary["sold"]))
    col_b.metric("Returned", format_number(summary["returned"]))
    col_c.metric("Return rate", format_percent(summary["return_rate"]))
    col_d.metric("Vs category", format_pp(summary["return_rate"] - summary["category_return_rate"]))
    col_e.metric("Vs dataset", format_pp(summary["return_rate"] - summary["dataset_return_rate"]))

    st.info(f"Dominant reason: {summary['dominant_reason']} | Recommendation: {summary['recommended_action']}")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Dominant reason share", format_percent(summary["dominant_reason_share"]))
    col_b.metric("Est. dominant reason returns", format_number(summary["dominant_reason_returns"]))
    col_c.metric("Sizing balance", format_pp(summary["size_balance"]))

    if full_page:
        variant_key = encode_variant_key(article_variant)
        render_pdf_export(
            raw,
            key=f"variant_{variant_key}",
            filename_prefix=f"variant_{variant_key}",
            generate_label="Generate variant PDF",
            download_label="Download variant PDF",
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(reason_bar(profile["reasons"]), width="stretch", key=f"variant_reasons_{encode_variant_key(article_variant)}")
    with right:
        st.plotly_chart(country_bar(profile["countries"]), width="stretch", key=f"variant_countries_{encode_variant_key(article_variant)}")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Benchmarks</div>', unsafe_allow_html=True)
        benchmark_rows = pd.DataFrame([
            {"benchmark": "Variant", "return_rate": summary["return_rate"]},
            {"benchmark": "Category", "return_rate": summary["category_return_rate"]},
            {"benchmark": "Article type", "return_rate": summary["type_return_rate"]},
            {"benchmark": "Dataset", "return_rate": summary["dataset_return_rate"]},
        ])
        st.dataframe(
            benchmark_rows,
            width="stretch",
            hide_index=True,
            column_config={"return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f")},
        )
    with right:
        st.markdown('<div class="section-title">Variant seasons</div>', unsafe_allow_html=True)
        st.dataframe(
            profile["seasons"],
            width="stretch",
            hide_index=True,
            column_config={"return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f")},
        )

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Reasons vs category</div>', unsafe_allow_html=True)
        st.dataframe(
            profile["reason_gap_vs_category"].head(10),
            width="stretch",
            hide_index=True,
            column_config={
                "share_of_returns": st.column_config.NumberColumn("variant share (%)", format="%.1f"),
                "category_share_of_returns": st.column_config.NumberColumn("category share (%)", format="%.1f"),
                "gap_vs_category": st.column_config.NumberColumn("gap vs category", format="%.1f"),
                "estimated_returns": st.column_config.NumberColumn("est. returns", format="%.0f"),
            },
        )
    with right:
        st.markdown('<div class="section-title">Similar better variants</div>', unsafe_allow_html=True)
        similar = profile["similar_products"]
        if similar.empty:
            st.info("No similar variants with a lower return rate in the current data range.")
        else:
            st.dataframe(
                with_variant_links(similar),
                width="stretch",
                hide_index=True,
                column_config=variant_column_config(
                    {
                        "return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f"),
                        "return_rate_advantage": st.column_config.NumberColumn("RR advantage", format="%.1f"),
                        "returned_delta_if_like_peer": st.column_config.NumberColumn("potential delta", format="%.0f"),
                    }
                ),
            )

    with st.expander("Variant source data", expanded=full_page):
        raw_columns = [
            "Article variant", "Zalando article variant", "Country", "Category", "Article type", "Season",
            "Sold articles", "Returned articles", "Return rate (%)", "Estimated return rate status", "Size-related return rate (%)",
        ] + reason_columns(raw)
        raw_columns = [column for column in raw_columns if column in raw.columns]
        st.dataframe(
            with_variant_links(raw[raw_columns]),
            width="stretch",
            hide_index=True,
            column_config=variant_column_config(),
        )


def render_variant_page(df: pd.DataFrame, article_variant: str) -> None:
    render_variant_analysis(df, article_variant, full_page=True)


def render_segmentation(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read segmentation",
        "Dashed lines split variants by median sales and weighted return rate. The key segment is high volume / high returns.",
    )
    min_sold = st.slider("Minimum sales for segmentation", 1, 500, 10)
    segments = cached_product_segments(df, min_sold=min_sold)
    if segments.empty:
        st.info("No products meet the minimum sales threshold.")
        return
    plot_variant_chart(segmentation_chart(segments), key="segmentation_variant_scatter")

    segment_summary = (
        segments.groupby("segment", dropna=False)
        .agg(variants=("Article variant", "nunique"), sold=("sold", "sum"), returned=("returned", "sum"))
        .reset_index()
    )
    segment_summary["return_rate"] = 100 * segment_summary["returned"] / segment_summary["sold"]
    st.dataframe(
        segment_summary,
        width="stretch",
        hide_index=True,
        column_config={"return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f")},
    )

    with st.expander("Details: variants in segments", expanded=False):
        st.dataframe(
            with_variant_links(segments.sort_values(["segment", "returned"], ascending=[True, False]).head(250)),
            width="stretch",
            hide_index=True,
            column_config=variant_column_config({"return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f")}),
        )


def render_pareto(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read Pareto",
        "Pareto shows whether the returns problem is concentrated in a small group of variants. The fewer variants cover 80% of returns, the more worthwhile it is to prioritize specific products.",
    )
    pareto = cached_pareto_products(df)
    st.plotly_chart(pareto_chart(pareto), width="stretch")
    breakpoints = cached_pareto_breakpoints(pareto)
    st.dataframe(
        breakpoints,
        width="stretch",
        hide_index=True,
        column_config={
            "variant_share": st.column_config.NumberColumn("variant share (%)", format="%.1f"),
            "returns_covered": st.column_config.NumberColumn("returns covered", format="%.0f"),
        },
    )
    with st.expander("Details: top Pareto variants", expanded=False):
        st.dataframe(
            with_variant_links(pareto.head(100)),
            width="stretch",
            hide_index=True,
            column_config=variant_column_config({
                "return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f"),
                "cumulative_return_share": st.column_config.NumberColumn("cumulative return share (%)", format="%.1f"),
                "cumulative_variant_share": st.column_config.NumberColumn("cumulative variant share (%)", format="%.1f"),
            }),
        )


def render_quality(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read data quality",
        "Low volume and unstable status do not mean the product has no problem. They mean decisions should be confirmed with additional data.",
    )
    validation_summary, validation_issues = cached_validation_report(df)
    st.markdown('<div class="section-title">Import validation</div>', unsafe_allow_html=True)
    if validation_issues.empty:
        st.success("No validation issues found in the current dataset.")
    else:
        col_a, col_b, col_c = st.columns(3)
        severity_counts = validation_issues["severity"].value_counts()
        col_a.metric("Errors", format_number(severity_counts.get("Error", 0)))
        col_b.metric("Warnings", format_number(severity_counts.get("Warning", 0)))
        col_c.metric("Info", format_number(severity_counts.get("Info", 0)))
        st.dataframe(
            validation_issues,
            width="stretch",
            hide_index=True,
            column_config={"share": st.column_config.NumberColumn("share (%)", format="%.1f")},
        )

    st.markdown('<div class="section-title">Data quality risks</div>', unsafe_allow_html=True)
    summary, risky_rows = cached_data_quality_report(df)
    st.plotly_chart(quality_bar(summary), width="stretch")
    with st.expander("Details: rows requiring caution", expanded=False):
        columns = [
            "Article variant", "Country", "Category", "Article type", "Sold articles", "Returned articles",
            "Return rate (%)", "Estimated return rate status", "Size-related return rate status", "low_volume", "unstable_status",
        ]
        columns = [column for column in columns if column in risky_rows.columns]
        st.dataframe(
            with_variant_links(risky_rows[columns].head(300)),
            width="stretch",
            hide_index=True,
            column_config=variant_column_config(),
        )


def render_seasons(df: pd.DataFrame) -> None:
    season_df = cached_season_article_type_analysis(df)
    st.plotly_chart(season_heatmap(season_df), width="stretch")
    with st.expander("Details: seasons and article types", expanded=False):
        st.dataframe(
            season_df.sort_values("returned", ascending=False),
            width="stretch",
            hide_index=True,
            column_config={"return_rate": st.column_config.NumberColumn("return rate (%)", format="%.1f")},
        )


def render_simulation(df: pd.DataFrame) -> None:
    render_guidance(
        "How to read simulation",
        "The simulation shows the potential effect of reducing selected return reasons by a chosen percentage. It is a directional estimate, not a financial forecast.",
    )
    reason_options = cached_reason_summary(df)["reason"].tolist()
    default_reasons = [reason for reason in ["Item is too small", "Item is too big"] if reason in reason_options]
    selected_reasons = st.multiselect("Reasons to reduce", options=reason_options, default=default_reasons or reason_options[:1])
    reduction_pct = st.slider("Assumed reduction of selected reasons", 0, 80, 10, step=5)
    result = cached_simulate_reason_reduction(df, tuple(selected_reasons), reduction_pct)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Current return rate", format_percent(result["current_return_rate"]))
    col_b.metric("After simulation", format_percent(result["new_return_rate"]))
    col_c.metric("Change", f'-{abs(result["return_rate_delta"]):.1f} p.p.')
    st.plotly_chart(simulation_chart(result), width="stretch")


def render_data(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Filtered data</div>', unsafe_allow_html=True)
    visible_columns = [
        "Article variant", "Zalando article variant", "Brand", "Country", "Category", "Article type", "Season",
        "Sold articles", "Returned articles", "Return rate (%)", "Estimated return rate status", "Size-related return rate (%)",
    ] + reason_columns(df)
    visible_columns = [column for column in visible_columns if column in df.columns]
    visible_df = df[visible_columns].copy()
    st.download_button(
        "Download full data CSV",
        data=dataframe_to_csv_bytes(visible_df),
        file_name=f"returns_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        width="stretch",
    )
    if len(visible_df) > TABLE_PREVIEW_ROWS:
        st.caption(
            f"Preview shows the first {TABLE_PREVIEW_ROWS} of {len(visible_df)} rows. "
            "Download the full result as CSV."
        )
    st.dataframe(
        with_variant_links(visible_df.head(TABLE_PREVIEW_ROWS)),
        width="stretch",
        hide_index=True,
        column_config=variant_column_config(),
    )


def render_selected_view(view: str, df: pd.DataFrame) -> None:
    if view == "Executive summary":
        render_executive_summary(df)
    elif view == "Overview":
        render_overview(df)
    elif view == "Action list":
        render_action_list(df)
    elif view == "Watchlist":
        render_watchlist(df)
    elif view == "Anomalies":
        render_anomalies(df)
    elif view == "Benchmarks":
        render_benchmarks(df)
    elif view == "Product profile":
        render_product_profile(df)
    elif view == "Return reasons":
        render_reasons(df)
    elif view == "Sizing":
        render_size(df)
    elif view == "Segmentation":
        render_segmentation(df)
    elif view == "Pareto":
        render_pareto(df)
    elif view == "Data quality":
        render_quality(df)
    elif view == "Seasons":
        render_seasons(df)
    elif view == "Simulation":
        render_simulation(df)
    elif view == "Products":
        render_products(df)
    else:
        render_data(df)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    df = sidebar_data_source()
    if df is None:
        st.stop()

    requested_variant = get_requested_variant()
    if requested_variant:
        render_variant_page(df, requested_variant)
        return

    view = sidebar_navigation()
    filtered, active_filters = apply_filters(df)
    render_hero(
        "Returns Analysis",
        "A dashboard for finding where returns happen, which reasons dominate, and which variants need action.",
    )

    if filtered.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    render_filter_context(filtered, active_filters)
    render_kpis(filtered, df)
    if view in {"Executive summary", "Overview"}:
        render_insight_cards(filtered)
    render_pdf_export(filtered)
    render_selected_view(view, filtered)


if __name__ == "__main__":
    main()
