# Streamlit Frontend

This folder documents the dashboard frontend. The runnable app is at the project root:

```bash
streamlit run streamlit_app.py
```

Dashboard sections:

1. KPI cards: total claims, high/medium/low risk, average fraud probability
2. Risk distribution chart
3. Fraud probability by region
4. Average fraud risk by product
5. Score trend
6. Top suspicious claims table
7. Batch and model monitoring JSON reports

Before launching, generate model outputs:

```bash
python src/train_model.py
python src/batch_score.py
python src/monitor_model.py
streamlit run streamlit_app.py
```
