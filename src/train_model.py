import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

from src.utils import load_config, ensure_parent, save_json
from src.feature_engineering import build_preprocessor
from src.data_validation import validate_columns


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["training_data"])

    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    target = cfg["target"]
    feature_cols = numeric + categorical

    validate_columns(df, feature_cols + [target])

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=cfg["model"]["random_state"], stratify=y
    )

    preprocessor = build_preprocessor(numeric, categorical)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=cfg["model"]["random_state"],
        class_weight="balanced"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ])

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= cfg["model"]["fraud_threshold"]).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "threshold": cfg["model"]["fraud_threshold"],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": feature_cols
    }

    ensure_parent(cfg["paths"]["model_path"])
    joblib.dump(pipeline, cfg["paths"]["model_path"])
    save_json(metrics, cfg["paths"]["metrics_report"])

    print("Model trained and saved.")
    print(metrics)


if __name__ == "__main__":
    main()
