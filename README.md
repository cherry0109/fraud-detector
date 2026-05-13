# Real-Time Healthcare Claims Fraud Detector

An end-to-end machine learning pipeline that ingests insurance claims, scores them for fraud using XGBoost, and generates plain-English investigator summaries with Azure OpenAI GPT-4o.

## Live Demo

Deployed on Streamlit Community Cloud — submit a claim and get a fraud score + AI explanation in real time.

## Features

- Detects 5 fraud patterns: upcoding, phantom billing, duplicate claims, unbundling, provider rings
- XGBoost classifier trained on 2,000 synthetic claims (10.2% fraud rate)
- Azure OpenAI GPT-4o generates a 3-sentence investigator summary for every HIGH/MEDIUM risk claim
- FastAPI REST endpoint for single and batch scoring
- Streamlit dashboard with claim submission, fraud case browser, and model evaluation

## Tech Stack

Python · XGBoost · Azure OpenAI · FastAPI · Streamlit · Pandas · scikit-learn

## Running Locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Streamlit Cloud Secrets

If you have Azure OpenAI access, add these in the Streamlit Cloud Secrets panel:

```toml
AZURE_OPENAI_KEY = "your-key"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
```

Without them, the app runs in demo mode with realistic hardcoded explanations.

## API (FastAPI)

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
# Swagger docs: http://localhost:8001/docs
```
