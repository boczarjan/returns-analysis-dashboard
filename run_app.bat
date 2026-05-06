@echo off
setlocal

cd /d "%~dp0"

if not defined RETURNS_CSV_PATH (
    if exist "%~dp0data\returns.csv" (
        set "RETURNS_CSV_PATH=%~dp0data\returns.csv"
    )
)

echo Starting Returns Analysis Dashboard...
echo.
echo App URL: http://localhost:8501
echo Data file: %RETURNS_CSV_PATH%
echo.

python -m streamlit run app.py --server.port 8501 --server.headless true

echo.
echo Application stopped or failed to start.
pause
