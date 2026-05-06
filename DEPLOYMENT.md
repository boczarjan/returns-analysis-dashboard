# Publikacja aplikacji

Ten projekt jest aplikacją Streamlit. Repozytorium GitHub przechowuje kod, a sama aplikacja musi działać na hostingu obsługującym proces Pythona.

## GitHub

Jeśli projekt nie jest jeszcze repozytorium Git:

```powershell
git init -b main
git add .
git commit -m "Initial returns dashboard"
```

Następnie utwórz puste repozytorium na GitHubie i podłącz je lokalnie:

```powershell
git remote add origin https://github.com/TWOJ_LOGIN/returns-analysis-dashboard.git
git push -u origin main
```

## Streamlit Community Cloud

To najprostszy hosting dla tej aplikacji.

1. Wejdź na https://streamlit.io/cloud.
2. Zaloguj się kontem GitHub.
3. Kliknij `New app`.
4. Wybierz repozytorium.
5. Jako `Main file path` ustaw `app.py`.
6. Kliknij `Deploy`.

Po wdrożeniu użytkownik może wgrać CSV przez panel boczny aplikacji. Nie trzeba dodawać pliku CSV do repozytorium.

## Render, Railway albo VPS

Na hostingu, który uruchamia proces Pythona, użyj komendy startowej:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Jeżeli hosting nie ustawia zmiennej `$PORT`, użyj na przykład portu `8501`.

## Dane i sekrety

- Nie wrzucaj prawdziwych plików CSV do publicznego repozytorium.
- Lokalny folder `data/` jest ignorowany przez Git poza plikiem `.gitkeep`.
- Cache `.returns_cache/`, logi i `.streamlit/secrets.toml` są ignorowane.
- Jeśli chcesz wskazać plik danych bez uploadu, ustaw zmienną środowiskową `RETURNS_CSV_PATH`.
