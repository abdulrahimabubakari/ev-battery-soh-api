from locust import HttpUser, task, between
import json

PAYLOAD = {
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

BATCH_PAYLOAD = {"cycles": [PAYLOAD] * 10}

class BatteryAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)
    host = "http://127.0.0.1:8000"

    @task(3)
    def predict_single(self):
        self.client.post("/predict", json=PAYLOAD)

    @task(1)
    def predict_batch(self):
        self.client.post("/batch-predict", json=BATCH_PAYLOAD)

    @task(1)
    def health_check(self):
        self.client.get("/health")