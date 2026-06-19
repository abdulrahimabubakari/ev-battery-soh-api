import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "val_rmse_percent" in data
    assert data["n_features"] == 15

def test_predict_valid():
    payload = {
        "discharge_duration": 5500,
        "voltage_mean": 3.47,
        "voltage_min": 2.85,
        "voltage_drop": 1.35,
        "voltage_std": 0.18,
        "current_mean": 0.99,
        "temp_mean": 6.5,
        "temp_max": 9.2,
        "temp_rise": 2.1,
        "temp_std": 0.8,
        "energy_discharged": 18500,
        "voltage_slope": -0.00012,
        "time_below_3_5v": 0.42,
        "time_above_35c": 0.0,
        "cycle_number": 50
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["soh_percent"] <= 100
    assert data["status"] in ["HEALTHY", "DEGRADED", "END_OF_LIFE"]
    assert data["inference_time_ms"] >= 0

def test_predict_invalid_missing_field():
    r = client.post("/predict", json={"voltage_mean": 3.47})
    assert r.status_code == 422

def test_batch_predict():
    cycle = {
        "discharge_duration": 5500,
        "voltage_mean": 3.47,
        "voltage_min": 2.85,
        "voltage_drop": 1.35,
        "voltage_std": 0.18,
        "current_mean": 0.99,
        "temp_mean": 6.5,
        "temp_max": 9.2,
        "temp_rise": 2.1,
        "temp_std": 0.8,
        "energy_discharged": 18500,
        "voltage_slope": -0.00012,
        "time_below_3_5v": 0.42,
        "time_above_35c": 0.0,
        "cycle_number": 50
    }
    r = client.post("/batch-predict", json={"cycles": [cycle, cycle, cycle]})
    assert r.status_code == 200
    data = r.json()
    assert data["total_cycles"] == 3
    assert len(data["predictions"]) == 3

def test_soh_status_thresholds():
    """Test that status labels match SoH thresholds correctly."""
    from api.main import get_status
    assert get_status(95) == "HEALTHY"
    assert get_status(85) == "DEGRADED"
    assert get_status(70) == "END_OF_LIFE"