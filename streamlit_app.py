"""
Streamlit dashboard for Fraud Detection Batch Scoring results.
Run:
    streamlit run streamlit_app.py
"""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCORED_PATH = ROOT / "data" / "scored" / "fraud_scored_output.csv"
BATCH_REPORT_PATH = ROOT / "reports" / "batch_summary.json"
MONITOR_REPORT_PATH = ROOT / "reports" / "model_monitoring_report.json"

st.set_page_config(
    page_title="Fraud Detection MLOps Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Fraud Detection MLOps Dashboard")
st.caption("Batch scoring results, fraud risk monitoring, and model operations summary")


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_scored_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "score_date" in df.columns:
        df["score_date"] = pd.to_datetime(df["score_date"], errors="coerce")
    return df


if not SCORED_PATH.exists():
    st.error("No scored output found. Run: python src/train_model.py and python src/batch_score.py")
    st.stop()

raw_df = load_scored_data(str(SCORED_PATH))
batch_report = load_json(BATCH_REPORT_PATH)
monitor_report = load_json(MONITOR_REPORT_PATH)

with st.sidebar:
    st.header("Filters")
    regions = sorted(raw_df["region"].dropna().unique().tolist()) if "region" in raw_df else []
    products = sorted(raw_df["product_type"].dropna().unique().tolist()) if "product_type" in raw_df else []
    risk_levels = ["High", "Medium", "Low"]

    selected_regions = st.multiselect("Region", regions, default=regions)
    selected_products = st.multiselect("Product Type", products, default=products)
    selected_risks = st.multiselect("Risk Level", risk_levels, default=risk_levels)
    min_probability = st.slider("Minimum fraud probability", 0.0, 1.0, 0.0, 0.01)

    st.divider()
    st.write("Model version:")
    if "model_version" in raw_df.columns:
        st.code(str(raw_df["model_version"].dropna().mode().iloc[0]))


df = raw_df.copy()
if selected_regions and "region" in df.columns:
    df = df[df["region"].isin(selected_regions)]
if selected_products and "product_type" in df.columns:
    df = df[df["product_type"].isin(selected_products)]
if selected_risks and "risk_level" in df.columns:
    df = df[df["risk_level"].isin(selected_risks)]
if "fraud_probability" in df.columns:
    df = df[df["fraud_probability"] >= min_probability]

# KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
total_claims = len(df)
high_risk = int((df["risk_level"] == "High").sum()) if "risk_level" in df else 0
medium_risk = int((df["risk_level"] == "Medium").sum()) if "risk_level" in df else 0
low_risk = int((df["risk_level"] == "Low").sum()) if "risk_level" in df else 0
avg_prob = float(df["fraud_probability"].mean()) if total_claims and "fraud_probability" in df else 0.0

c1.metric("Claims Scored", f"{total_claims:,}")
c2.metric("High Risk", f"{high_risk:,}")
c3.metric("Medium Risk", f"{medium_risk:,}")
c4.metric("Low Risk", f"{low_risk:,}")
c5.metric("Avg Fraud Probability", f"{avg_prob:.2%}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Risk Level Distribution")
    if "risk_level" in df.columns and not df.empty:
        risk_counts = df["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        st.bar_chart(risk_counts)
    else:
        st.info("No data available for selected filters.")

with right:
    st.subheader("Fraud Probability by Region")
    if {"region", "fraud_probability"}.issubset(df.columns) and not df.empty:
        region_avg = df.groupby("region")["fraud_probability"].mean().sort_values(ascending=False)
        st.bar_chart(region_avg)
    else:
        st.info("No region/probability data available.")

left2, right2 = st.columns(2)

with left2:
    st.subheader("Average Risk by Product")
    if {"product_type", "fraud_probability"}.issubset(df.columns) and not df.empty:
        product_avg = df.groupby("product_type")["fraud_probability"].mean().sort_values(ascending=False)
        st.bar_chart(product_avg)
    else:
        st.info("No product/probability data available.")

with right2:
    st.subheader("Fraud Probability Trend")
    if {"score_date", "fraud_probability"}.issubset(df.columns) and not df.empty:
        trend = df.groupby(df["score_date"].dt.date)["fraud_probability"].mean()
        st.line_chart(trend)
    else:
        st.info("No score date/probability data available.")

st.divider()

st.subheader("Top Suspicious Claims")
show_cols = [
    "claim_id", "customer_id", "claim_amount", "region", "product_type",
    "fraud_probability", "fraud_flag", "risk_level", "score_date", "model_version"
]
show_cols = [c for c in show_cols if c in df.columns]

top_n = st.slider("Number of suspicious claims to display", 5, 100, 20, 5)
if not df.empty:
    top_df = df.sort_values("fraud_probability", ascending=False).head(top_n)
    st.dataframe(
        top_df[show_cols],
        use_container_width=True,
        hide_index=True,
    )
    csv = top_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered suspicious claims CSV",
        data=csv,
        file_name="filtered_suspicious_claims.csv",
        mime="text/csv",
    )
else:
    st.warning("No records match the selected filters.")

st.divider()

st.subheader("MLOps Monitoring Summary")
mc1, mc2 = st.columns(2)
with mc1:
    st.write("Batch Report")
    st.json(batch_report if batch_report else {"message": "Run python src/batch_score.py to generate batch summary."})
with mc2:
    st.write("Model Monitoring Report")
    st.json(monitor_report if monitor_report else {"message": "Run python src/monitor_model.py to generate monitoring report."})

st.caption("Owner view: monitor batch health, scored records, fraud distribution, drift indicators, and dashboard-ready outputs.")
