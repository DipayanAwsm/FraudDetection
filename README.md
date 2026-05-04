# Fraud Detection MLOps Practice Project

This is a complete practice project for a Fraud Detection Batch Scoring MLOps pipeline with a Streamlit dashboard frontend.

## What is included

```text
1. Initial synthetic fraud training data
2. Model training pipeline
3. Batch scoring pipeline
4. Model monitoring report
5. Streamlit dashboard frontend
6. Azure deployment notes
7. GitHub Actions scheduled batch example
```

## Project flow

```text
Raw Claims Data
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Fraud Model Training
   ↓
Batch Scoring
   ↓
Scored Output CSV
   ↓
Streamlit Dashboard
   ↓
Monitoring Reports
```

## Run locally

```bash
pip install -r requirements.txt
python src/train_model.py
python src/batch_score.py
python src/monitor_model.py
streamlit run streamlit_app.py
```

## Important files

```text
data/raw/initial_claims_training_data.csv       Training data
data/raw/new_claims_for_batch_scoring.csv       New claims for batch scoring
models/fraud_model.pkl                          Trained model
src/train_model.py                              Model training
src/batch_score.py                              Batch scoring job
src/monitor_model.py                            Monitoring report
streamlit_app.py                                Streamlit dashboard
azure/startup.sh                                Azure App Service startup script
```

## Dashboard views

```text
Total claims scored
High / Medium / Low risk cases
Fraud probability by region
Fraud risk by product
Top suspicious claims
Batch summary
Model monitoring report
```

## Azure quick deployment

Use Azure App Service for Linux.

Startup command:

```bash
bash azure/startup.sh
```

For real production, replace local CSV storage with Azure SQL, Azure Blob, ADLS, or Databricks Delta tables.
