# Returns Analysis Dashboard

Interactive Streamlit dashboard for browsing, filtering, and visualizing return reasons from a zDirect/Zalando CSV file.

## Features

- KPIs: sales, returns, weighted return rate, simple average return rate, and NMV.
- Interactive Plotly charts with hover details, zoom, and filtering.
- Analysis of return reasons, countries, categories, article types, and seasons.
- Action list with priorities, problem type, and recommended action.
- Variant benchmarks vs category, article type, and the full dataset.
- Clickable `Article variant` codes that open a dedicated variant page.
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
