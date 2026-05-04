#!/bin/bash
python src/train_model.py
python src/batch_score.py
python src/monitor_model.py
streamlit run streamlit_app.py --server.port=${PORT:-8000} --server.address=0.0.0.0
