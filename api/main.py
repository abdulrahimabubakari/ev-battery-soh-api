# EV Battery SoH Prediction API v1.1
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List
import pickle
import numpy as np
import pandas as pd
import time
import os

# ── Load model on startup ──────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'lightgbm_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

FEATURE_COLS = [
    'discharge_duration', 'voltage_mean', 'voltage_min', 'voltage_drop',
    'voltage_std', 'current_mean', 'temp_mean', 'temp_max', 'temp_rise',
    'temp_std', 'energy_discharged', 'voltage_slope',
    'time_below_3_5v', 'time_above_35c', 'cycle_number'
]
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="EV Battery SoH Prediction API",
    description="Predicts battery State-of-Health using physics-informed ML features",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://battery-soh-dashboard.onrender.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────
class CycleFeatures(BaseModel):
    discharge_duration: float = Field(..., description="Discharge duration in seconds")
    voltage_mean: float       = Field(..., description="Mean discharge voltage (V)")
    voltage_min: float        = Field(..., description="Minimum voltage (V)")
    voltage_drop: float       = Field(..., description="Voltage drop start-to-end (V)")
    voltage_std: float        = Field(..., description="Voltage standard deviation")
    current_mean: float       = Field(..., description="Mean absolute current (A)")
    temp_mean: float          = Field(..., description="Mean temperature (°C)")
    temp_max: float           = Field(..., description="Max temperature (°C)")
    temp_rise: float          = Field(..., description="Temperature rise (°C)")
    temp_std: float           = Field(..., description="Temperature std deviation")
    energy_discharged: float  = Field(..., description="Energy discharged (Wh approx)")
    voltage_slope: float      = Field(..., description="Voltage degradation slope")
    time_below_3_5v: float    = Field(..., ge=0, le=1, description="Fraction of time below 3.5V")
    time_above_35c: float     = Field(..., ge=0, le=1, description="Fraction of time above 35°C")
    cycle_number: int         = Field(..., ge=1, description="Cycle number in battery life")

class PredictionResponse(BaseModel):
    soh_percent: float
    status: str
    inference_time_ms: float

class BatchRequest(BaseModel):
    cycles: List[CycleFeatures]

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_cycles: int
    inference_time_ms: float

# ── Helper ─────────────────────────────────────────────────
def get_status(soh: float) -> str:
    if soh >= 90:
        return "HEALTHY"
    elif soh >= 80:
        return "DEGRADED"
    else:
        return "END_OF_LIFE"

# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": "LightGBM", "version": "1.0.0"}

@app.get("/model/info")
async def model_info():
    return {
        "model_type": "LightGBM",
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "best_iteration": int(model.best_iteration_),
        "val_rmse_percent": 4.80,
        "val_r2": 0.89
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(cycle: CycleFeatures):
    start = time.perf_counter()
    
    features = np.array([[getattr(cycle, col) for col in FEATURE_COLS]])
    soh = float(model.predict(features)[0])
    soh = max(0.0, min(100.0, soh))  # clip to valid range
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return PredictionResponse(
        soh_percent=round(soh, 2),
        status=get_status(soh),
        inference_time_ms=round(elapsed_ms, 3)
    )

@app.post("/batch-predict", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    start = time.perf_counter()
    
    if len(request.cycles) > 1000:
        raise HTTPException(status_code=400, detail="Max batch size is 1000")
    
    features = np.array([
        [getattr(c, col) for col in FEATURE_COLS]
        for c in request.cycles
    ])
    
    preds = model.predict(features)
    preds = np.clip(preds, 0, 100)
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    predictions = [
        PredictionResponse(
            soh_percent=round(float(p), 2),
            status=get_status(float(p)),
            inference_time_ms=round(elapsed_ms / len(preds), 3)
        )
        for p in preds
    ]
    
    return BatchResponse(
        predictions=predictions,
        total_cycles=len(predictions),
        inference_time_ms=round(elapsed_ms, 3)
    )
import io

@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    contents = await file.read()
    
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df.columns = df.columns.str.strip()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file")
    
    required = ['Voltage_measured', 'Current_measured', 
                'Temperature_measured', 'Time']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing}. Required: {required}"
        )
    
    df = df.dropna()
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="CSV needs at least 10 rows")
    
    start = time.perf_counter()
    
    v = df['Voltage_measured'].values
    i = df['Current_measured'].values
    t = df['Temperature_measured'].values
    time_arr = df['Time'].values

    dt = np.diff(time_arr)
    power = np.abs(i[:-1]) * v[:-1]

    features = {
        'discharge_duration': float(time_arr[-1] - time_arr[0]),
        'voltage_mean': float(np.mean(v)),
        'voltage_min': float(np.min(v)),
        'voltage_drop': float(v[0] - v[-1]),
        'voltage_std': float(np.std(v)),
        'current_mean': float(np.mean(np.abs(i))),
        'temp_mean': float(np.mean(t)),
        'temp_max': float(np.max(t)),
        'temp_rise': float(t[-1] - t[0]),
        'temp_std': float(np.std(t)),
        'energy_discharged': float(np.sum(power * dt)),
        'voltage_slope': float(np.polyfit(time_arr, v, 1)[0]),
        'time_below_3_5v': float(np.sum(v < 3.5) / len(v)),
        'time_above_35c': float(np.sum(t > 35) / len(t)),
        'cycle_number': 1,
    }

    X = np.array([[features[col] for col in FEATURE_COLS]])
    soh = float(model.predict(X)[0])
    soh = max(0.0, min(100.0, soh))
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "soh_percent": round(soh, 2),
        "status": get_status(soh),
        "inference_time_ms": round(elapsed_ms, 3),
        "rows_processed": len(df),
        "features_extracted": features,
        "filename": file.filename
    }