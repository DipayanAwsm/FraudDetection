import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from datetime import datetime
import pandas as pd
import joblib

from src.utils import load_config, ensure_parent, save_json
from src.data_validation import validate_batch_data


def risk_level(prob, high_threshold, medium_threshold):
    if prob >= high_threshold:
        return "High"
    if prob >= medium_threshold:
        return "Medium"
    return "Low"


def main():
    cfg = load_config()
    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    feature_cols = numeric + categorical

    df = pd.read_csv(cfg["paths"]["batch_input"])
    validation_summary = validate_batch_data(df, feature_cols)

    model = joblib.load(cfg["paths"]["model_path"])
    probs = model.predict_proba(df[feature_cols])[:, 1]

    scored = df.copy()
    scored["fraud_probability"] = probs.round(4)
    scored["fraud_flag"] = scored["fraud_probability"].apply(
        lambda x: "Fraud Risk" if x >= cfg["model"]["fraud_threshold"] else "Normal"
    )
    scored["risk_level"] = scored["fraud_probability"].apply(
        lambda x: risk_level(x, cfg["model"]["fraud_threshold"], cfg["model"]["medium_risk_threshold"])
    )
    scored["score_date"] = datetime.today().strftime("%Y-%m-%d")
    scored["model_version"] = "fraud_model_v1"

    ensure_parent(cfg["paths"]["scored_output"])
    scored.to_csv(cfg["paths"]["scored_output"], index=False)

    summary = {
        "run_time": datetime.now().isoformat(),
        "input_rows": int(len(df)),
        "output_rows": int(len(scored)),
        "high_risk_count": int((scored["risk_level"] == "High").sum()),
        "medium_risk_count": int((scored["risk_level"] == "Medium").sum()),
        "low_risk_count": int((scored["risk_level"] == "Low").sum()),
        "average_fraud_probability": float(scored["fraud_probability"].mean()),
        "validation_summary": validation_summary
    }
    save_json(summary, cfg["paths"]["batch_summary"])

    print("Batch scoring completed.")
    print(summary)


if __name__ == "__main__":
    main()
