# App Deployment

This project is a Streamlit app. The GitHub repository stores the code, while the app itself must run on hosting that supports a Python process.

## GitHub

If the project is not yet a Git repository:

```powershell
git init -b main
git add .
git commit -m "Initial returns dashboard"
```

Then create an empty repository on GitHub and connect it locally:

```powershell
git remote add origin https://github.com/YOUR_LOGIN/returns-analysis-dashboard.git
git push -u origin main
```

## Streamlit Community Cloud

This is the simplest hosting option for this app.

1. Go to https://streamlit.io/cloud.
2. Sign in with GitHub.
3. Click `New app`.
4. Select the repository.
5. Set `Main file path` to `app.py`.
6. Click `Deploy`.

After deployment, users can upload a CSV through the app sidebar. You do not need to add the CSV file to the repository.

## Render, Railway, or VPS

On hosting that starts a Python process, use this start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

If the hosting platform does not set `$PORT`, use a port such as `8501`.

## Data and Secrets

- Do not commit real CSV files to a public repository.
- The local `data/` folder is ignored by Git except for `.gitkeep`.
- `.returns_cache/`, logs, and `.streamlit/secrets.toml` are ignored.
- To point the app to a data file without upload, set the `RETURNS_CSV_PATH` environment variable.
