# Azure Deployment Notes - Streamlit Version

## Recommended Azure service

Use **Azure App Service for Linux** for quick practice.

## Startup command

In Azure App Service > Configuration > General settings > Startup Command:

```bash
bash azure/startup.sh
```

## Local run

```bash
pip install -r requirements.txt
python src/train_model.py
python src/batch_score.py
python src/monitor_model.py
streamlit run streamlit_app.py
```

## What gets displayed

The app reads:

```text
data/scored/fraud_scored_output.csv
reports/batch_summary.json
reports/model_monitoring_report.json
```

## Production improvement ideas

For real deployment, store scored results in Azure SQL, Azure Blob, ADLS, or Databricks Delta instead of local CSV.
Then update `streamlit_app.py` to read from that source.
