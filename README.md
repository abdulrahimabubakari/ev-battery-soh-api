\# EV Battery State-of-Health (SoH) Prediction API



A production-grade ML system for predicting lithium-ion battery degradation using physics-informed features and gradient boosting models.

## Live Demo
- **API:** https://battery-soh-predictor.onrender.com/docs
- **Health:** https://battery-soh-predictor.onrender.com/health

\## Results



\- \*\*RMSE: 4.8% SoH\*\* on held-out validation set

\- \*\*R²: 0.89\*\* predictive accuracy

\- \*\*P95 latency: 11ms\*\* at 50 concurrent users

\- \*\*151 requests/sec\*\* throughput

\- \*\*0% failure rate\*\* under load



\## Tech Stack



| Layer | Technology |

|---|---|

| ML Model | LightGBM (SoH regression) |

| Data | NASA Battery Dataset (7,565 cycles) |

| Features | 15 physics-informed degradation features |

| API | FastAPI + Uvicorn (async) |

| Experiment Tracking | MLflow |

| Frontend | React + Recharts |

| Testing | pytest (6/6) + Locust load testing |

| CI/CD | GitHub Actions |

| Deployment | Docker + docker-compose |



\## Quick Start



\### API

```bash

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

uvicorn api.main:app --reload --port 8000

```



\### Frontend

```bash

cd frontend

npm install

npm start

```



\### Tests

```bash

pytest tests/test\_api.py -v

```



\## API Endpoints



| Method | Endpoint | Description |

|---|---|---|

| GET | /health | Health check |

| GET | /model/info | Model metadata |

| POST | /predict | Single cycle SoH prediction |

| POST | /batch-predict | Bulk predictions (max 1000) |



\## Project Structure



```

ev-battery-soh-api/

├── api/              # FastAPI backend

├── src/              # ML training scripts

├── notebooks/        # EDA notebooks

├── frontend/         # React dashboard

├── tests/            # pytest + Locust

├── docker/           # Dockerfiles

└── .github/workflows # CI pipeline

```

 
