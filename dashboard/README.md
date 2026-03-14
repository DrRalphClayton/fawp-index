# FAWP Dashboard

Streamlit app for the [fawp-index](https://github.com/DrRalphClayton/fawp-index) package.

## Run locally

```bash
pip install "fawp-index[dashboard]"
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → set **Main file path** to `dashboard/app.py`
4. Deploy

The `requirements.txt` and `.streamlit/config.toml` in this folder are
picked up automatically by Streamlit Cloud.

## Live demo

[fawp-scanner.info](https://fawp-scanner.info)
