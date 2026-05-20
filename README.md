# Returns Analysis Dashboard

Interactive Streamlit dashboard for browsing, filtering, and visualizing return reasons from a zDirect/Zalando CSV file.

## Features

- KPIs: sales, returns, weighted return rate, simple average return rate, and NMV.
- Interactive Plotly charts with hover details, zoom, and filtering.
- Analysis of return reasons, countries, categories, article types, and seasons.
- Product rankings for a selected return reason, with category and full-dataset benchmarks.
- Pricing analysis with average net price, returned NMV, and high price / high return detection.
- Optional sidebar switch to exclude the `No details` return reason and recalculate metrics without it.
- `N/A` article variants are split by Zalando article variant instead of being grouped together.
- Action list with priorities, problem type, and recommended action.
- Variant benchmarks vs category, article type, and the full dataset.
- Clickable `Article variant` codes that open a dedicated variant page.
- Lifecycle and new product early warning reports using `Days online` or `Date first on offer`.
- Country playbook with dominant reasons, size skew, risky variants, and recommended focus.
- Size & fit intelligence with too-big / too-small skew and variant-level fit risk scoring.
- Predictive return risk view with expected next-30-day returns, risk drivers, and recommended actions.
- Preventable returns report that groups fixable reasons into size, description, visual/material, quality, fulfillment, and price areas.
- Size guidance report with size-up / size-down / fit-copy recommendations and confidence levels.
- Trend report for comparing current filtered data with a previous-period CSV baseline.
- Forecast report with configurable horizon, expected returns, returned NMV, and risk score.
- Product page audit pack for PDP, fit, quality, price, fulfillment, and market checks.
- Quality/Supplier, Country, Pricing risk, New product early warning, trend, preventable returns, size guidance, and forecast report views with CSV export.
- Executive cockpit with top fixes, pricing risks, new product alerts, and narrative insight cards.
- Analysis presets, resettable filters, data-retention context, and a higher-confidence filter.
- Compare mode for countries, categories, article types, seasons, genders, and variants.
- Risk-index columns and compact product audit layout for faster visual scanning.
- Modern visual system with semantic risk colors, compact bento-style cards, hover microinteractions, and light/dark mode.
- Quick-scan table mode with optional advanced columns for deeper analysis.
- Plotly charts inherit the selected app theme for consistent light and dark presentations.
- Segmentation, Pareto, data quality analysis, and return reduction simulation.
- PDF export for the currently filtered report and CSV export for tables.
- Parquet cache and calculation cache for faster work on larger files.

## Local Run

1. Install Python 3.11 or newer.
2. Clone the repository or download the project.
3. In the project directory, create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

4. Start the app:

```powershell
streamlit run app.py
```

On Windows, you can also use:

```powershell
.\run_app.bat
```

## Data

The app does not include a CSV data file because sales and returns data usually should not be stored in a public repository.

You have three options for loading data:

1. Upload a CSV file in the app sidebar.
2. Place the file at `data/returns.csv`.
3. Set the `RETURNS_CSV_PATH` environment variable:

```powershell
$env:RETURNS_CSV_PATH="D:\path\to\file.csv"
streamlit run app.py
```
