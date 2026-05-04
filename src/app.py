from fastapi import FastAPI
import pandas as pd
from pathlib import Path

app = FastAPI(title="Fraud Detection Batch Scoring API")

@app.get("/")
def root():
    return {"message": "Fraud Detection MLOps project is running"}

@app.get("/scored-summary")
def scored_summary():
    path = Path("data/scored/fraud_scored_output.csv")
    if not path.exists():
        return {"error": "No scored output found. Run python src/train_model.py and python src/batch_score.py first."}
    df = pd.read_csv(path)
    return {
        "rows": int(len(df)),
        "high_risk": int((df["risk_level"] == "High").sum()),
        "medium_risk": int((df["risk_level"] == "Medium").sum()),
        "low_risk": int((df["risk_level"] == "Low").sum()),
        "avg_fraud_probability": float(df["fraud_probability"].mean())
    }
