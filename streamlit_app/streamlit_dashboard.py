"""
Streamlit Dashboard for Predictive Maintenance System

Interactive dashboard for:
- Real-time RUL predictions
- Equipment monitoring
- Risk analysis
- Historical trends
- Model performance
"""
"""
Streamlit Dashboard for Predictive Maintenance System

Interactive dashboard for:
- Real-time RUL predictions
- Equipment monitoring
- Risk analysis
- Historical trends
- Model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import requests

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/saved_models/xgboost_model.pkl')
        scaler = joblib.load('models/saved_models/scaler.pkl')
        feature_columns = joblib.load('models/saved_models/feature_columns.pkl')
        return model, scaler, feature_columns, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, False
    
@st.cache_data
def load_sample_data():
    try:
        data = pd.read_csv('data/processed/train_processed.csv')
        return data
    except Exception as e:
        st.warning(f"Sample data not available: {e}")
        return None


def determine_risk_level(rul):
    if rul < 10:
        return "Critical"
    elif rul < 30:
        return "High"
    elif rul < 60:
        return "Medium"
    else:
        return "Low"
    
def get_risk_color(risk_level):
    colors = {
        "Critical": "#FF4B4B",
        "High": "#FFA500",
        "Medium": "#FFD700",
        "Low": "#32CD32"
    }
    return colors.get(risk_level, "#808080")


def predict_rul(model, scaler, feature_columns, features):
    try:
        X = pd.DataFrame([features])
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]

        X_scaled = scaler.transform(X)
        rul = float(model.predict(X_scaled)[0])
        return max(0, rul)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None