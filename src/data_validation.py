import pandas as pd


def validate_columns(df: pd.DataFrame, required_columns: list):
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def validate_batch_data(df: pd.DataFrame, feature_columns: list):
    validate_columns(df, feature_columns)
    checks = {
        "row_count": int(len(df)),
        "duplicate_claim_ids": int(df["claim_id"].duplicated().sum()) if "claim_id" in df.columns else None,
        "missing_values_total": int(df[feature_columns].isna().sum().sum()),
    }
    if checks["row_count"] == 0:
        raise ValueError("Input data has zero rows.")
    if checks["missing_values_total"] > 0:
        raise ValueError(f"Input data contains missing values: {checks['missing_values_total']}")
    return checks
