import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.utils import load_config, save_json


def main():
    cfg = load_config()
    scored = pd.read_csv(cfg["paths"]["scored_output"])

    report = {
        "rows_scored": int(len(scored)),
        "avg_fraud_probability": float(scored["fraud_probability"].mean()),
        "max_fraud_probability": float(scored["fraud_probability"].max()),
        "high_risk_rate": float((scored["risk_level"] == "High").mean()),
        "medium_risk_rate": float((scored["risk_level"] == "Medium").mean()),
        "low_risk_rate": float((scored["risk_level"] == "Low").mean()),
        "records_by_region": scored.groupby("region").size().to_dict(),
        "high_risk_by_region": scored[scored["risk_level"] == "High"].groupby("region").size().to_dict(),
        "records_by_product": scored.groupby("product_type").size().to_dict(),
        "recommended_action": "Review high-risk claims first. Investigate drift if high_risk_rate changes sharply over time."
    }

    save_json(report, cfg["paths"]["monitoring_report"])
    print("Monitoring report generated.")
    print(report)


if __name__ == "__main__":
    main()
