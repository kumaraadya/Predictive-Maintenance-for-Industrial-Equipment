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
MODEL_PATH = "models/saved_models/xgboost_model.pkl"
SCALER_PATH = "models/saved_models/scaler.pkl"
FEATURES_PATH = "models/saved_models/feature_columns.pkl"

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

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = pd.read_csv("../models/saved_models/final_model_comparison.csv")
        xgb_results = results[results['model'] == 'XGBoost'].iloc[0]
        
        performance_metrics = {
            "rmse": float(xgb_results['rmse']),
            "mae": float(xgb_results['mae']),
            "r2_score": float(xgb_results['r2']),
            "asymmetric_score": float(xgb_results['asymmetric_score'])
        }
    except:
        performance_metrics = {"status": "metrics not available"}
    
    return ModelInfo(
        model_type="XGBoost",
        features_count=len(feature_columns) if feature_columns else 0,
        model_version="1.0.0",
        training_date="2025-01-05",
        performance_metrics=performance_metrics
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_rul(reading: SensorReading):
    if model is None or scaler is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        feature_dict = {
            'time_cycles': reading.time_cycles,
            'op_setting_1': reading.op_setting_1,
            'op_setting_2': reading.op_setting_2,
            'op_setting_3': reading.op_setting_3,
        }
        feature_dict.update(reading.sensor_readings)

        X = pd.DataFrame([feature_dict])

        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[feature_columns]

        X_scaled = scaler.transform(X)

        rul_pred = float(model.predict(X_scaled)[0])

        confidence_interval = (max(0, rul_pred - 10), rul_pred + 10)

        risk_level = determine_risk_level(rul_pred)

        recommendations = get_recommendations(rul_pred, risk_level)
        
        return PredictionResponse(
            unit_id=reading.unit_id,
            time_cycles=reading.time_cycles,
            predicted_rul=round(rul_pred, 2),
            confidence_interval=confidence_interval,
            risk_level=risk_level,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    risk_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    total_rul = 0
    
    for reading in request.readings:
        try:
            pred = await predict_rul(reading)
            predictions.append(pred)
            risk_counts[pred.risk_level] += 1
            total_rul += pred.predicted_rul
        except Exception as e:
            print(f"Error predicting for unit {reading.unit_id}: {e}")
            continue
    
    summary = {
        "total_predictions": len(predictions),
        "average_rul": round(total_rul / len(predictions), 2) if predictions else 0,
        "risk_distribution": risk_counts,
        "critical_units": [p.unit_id for p in predictions if p.risk_level == "CRITICAL"],
        "high_risk_units": [p.unit_id for p in predictions if p.risk_level == "HIGH"]
    }
    
    return BatchPredictionResponse(
        predictions=predictions,
        summary=summary
    )

@app.get("/statistics", tags=["Analytics"])
async def get_statistics():
    return {
        "message": "Statistics endpoint",
        "note": "Connect to database for real statistics",
        "example_stats": {
            "total_predictions_today": 150,
            "average_rul": 65.5,
            "critical_alerts": 3,
            "high_risk_alerts": 8
        }
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "path": str(request.url)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


