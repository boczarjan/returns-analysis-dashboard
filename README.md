# Returns Analysis Dashboard

Interaktywny dashboard Streamlit do przeglądania, filtrowania i wizualizacji powodów zwrotów z pliku CSV zDirect/Zalando.

## Funkcje

- KPI: sprzedaż, zwroty, ważony return rate, zwykła średnia return rate i NMV.
- Interaktywne wykresy Plotly z hoverami, zoomem i filtrowaniem.
- Analiza powodów zwrotów, krajów, kategorii, typów artykułów i sezonów.
- Action list z priorytetami, typem problemu i rekomendowaną akcją.
- Benchmarki wariantu vs kategoria, typ artykułu i cały dataset.
- Klikalne kody `Article variant` prowadzące do osobnej strony wariantu.
- Segmentacja, Pareto, analiza jakości danych i symulacja redukcji zwrotów.
- Eksport aktualnie wyfiltrowanego raportu do PDF oraz eksport tabel do CSV.
- Cache Parquet i cache obliczeń dla szybszego działania na większych plikach.

## Uruchomienie lokalne

1. Zainstaluj Pythona 3.11 lub nowszego.
2. Sklonuj repozytorium albo pobierz projekt.
3. W katalogu projektu utwórz środowisko i zainstaluj zależności:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

4. Uruchom aplikację:

```powershell
streamlit run app.py
```

Na Windows możesz też użyć pliku:

```powershell
.\run_app.bat
```

## Dane

Aplikacja nie zawiera pliku CSV z danymi, ponieważ dane sprzedażowe i zwrotowe zwykle nie powinny trafiać do publicznego repozytorium.

Masz trzy opcje wczytania danych:

1. Wgraj plik CSV w panelu bocznym aplikacji.
2. Umieść plik jako `data/returns.csv`.
3. Ustaw zmienną środowiskową `RETURNS_CSV_PATH`:

```powershell
$env:RETURNS_CSV_PATH="D:\sciezka\do\pliku.csv"
streamlit run app.py
```
