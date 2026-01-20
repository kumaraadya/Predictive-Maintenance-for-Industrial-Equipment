"""
FastAPI Backend for Predictive Maintenance System

This API provides endpoints for:
- Health checks
- Model predictions
- Batch predictions
- Model information
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = FastAPI(
    title="Predictive Maintenance API",
    description="REST API for predicting equipment Remaining Useful Life (RUL)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None
feature_columns = None
MODEL_PATH = "../models/saved_models/xgboost_model.pkl"
SCALER_PATH = "../models/saved_models/scaler.pkl"
FEATURES_PATH = "../models/saved_models/feature_columns.pkl"

class SensorReading(BaseModel):
    unit_id: int = Field(..., description="Engine/Unit ID", ge=1)
    time_cycles: int = Field(..., description="Current operational cycle", ge=1)
    op_setting_1: float = Field(..., description="Operational setting 1")
    op_setting_2: float = Field(..., description="Operational setting 2")
    op_setting_3: float = Field(..., description="Operational setting 3")
    sensor_readings: Dict[str, float] = Field(..., description="Sensor measurements")
    
    class Config:
        schema_extra = {
            "example": {
                "unit_id": 1,
                "time_cycles": 50,
                "op_setting_1": 0.0023,
                "op_setting_2": 0.0003,
                "op_setting_3": 100.0,
                "sensor_readings": {
                    "sensor_1": 518.67,
                    "sensor_2": 643.02,
                    "sensor_3": 1585.29,
                    "sensor_4": 1398.21,
                    "sensor_7": 554.85,
                    # ... other sensors
                }
            }
        }

class PredictionResponse(BaseModel):
    unit_id: int
    time_cycles: int
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life (cycles)")
    confidence_interval: tuple[float, float] = Field(..., description="95% confidence interval")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    recommendations: List[str] = Field(..., description="Maintenance recommendations")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionRequest(BaseModel):
    readings: List[SensorReading]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str

class ModelInfo(BaseModel):
    model_type: str
    features_count: int
    model_version: str
    training_date: str
    performance_metrics: Dict[str, float]

@app.on_event("startup")
async def load_models():
    global model, scaler, feature_columns
    
    try:
        print("Loading models...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("API will run but predictions will fail")


def determine_risk_level(rul: float) -> str:
    if rul < 10:
        return "Critical"
    elif rul < 30:
        return "High"
    elif rul < 60:
        return "Medium"
    else:
        return "Low"
    
def get_recommendations(rul: float, risk_level: str) -> List[str]:
    recommendations = []
    
    if risk_level == "Critical":
        recommendations = [
            "IMMEDIATE ACTION REQUIRED",
            "Schedule emergency maintenance within 24 hours",
            "Consider temporary equipment shutdown",
            "Prepare replacement parts",
            "Alert maintenance team and management"
        ]
    elif risk_level == "High":
        recommendations = [
            "Schedule maintenance within 1 week",
            "Increase monitoring frequency",
            "Order replacement parts",
            "Prepare maintenance crew"
        ]
    elif risk_level == "Medium":
        recommendations = [
            "Schedule maintenance within 2-3 weeks",
            "Continue regular monitoring",
            "Review maintenance history",
            "Check parts inventory"
        ]
    else:
        recommendations = [
            "Equipment operating normally",
            "Continue routine maintenance schedule",
            "Regular monitoring recommended"
        ]
    
    return recommendations