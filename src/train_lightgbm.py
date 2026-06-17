import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.lightgbm
import warnings
warnings.filterwarnings('ignore')

# ── Load features ──────────────────────────────────────────
df = pd.read_parquet('data/features.parquet')

FEATURE_COLS = [
    'discharge_duration', 'voltage_mean', 'voltage_min', 'voltage_drop',
    'voltage_std', 'current_mean', 'temp_mean', 'temp_max', 'temp_rise',
    'temp_std', 'energy_discharged', 'voltage_slope',
    'time_below_3_5v', 'time_above_35c', 'cycle_number'
]

X = df[FEATURE_COLS].values
y = df['SoH'].values

# ── Temporal split ─────────────────────────────────────────
df['split'] = df.groupby('battery_id')['cycle_number'].transform(
    lambda x: (x > x.quantile(0.8)).astype(int)
)
train_mask = df['split'] == 0
val_mask   = df['split'] == 1

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]

print(f"Train: {len(X_train)}  Val: {len(X_val)}")

# ── LightGBM ───────────────────────────────────────────────
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        __import__('lightgbm').early_stopping(50, verbose=False),
        __import__('lightgbm').log_evaluation(100)
    ]
)

preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
mae  = mean_absolute_error(y_val, preds)
r2   = r2_score(y_val, preds)

print(f"\n=== LightGBM Validation Results ===")
print(f"  RMSE : {rmse:.4f} % SoH")
print(f"  MAE  : {mae:.4f} % SoH")
print(f"  R²   : {r2:.4f}")
print(f"  Best iteration: {model.best_iteration_}")

# ── MLflow ─────────────────────────────────────────────────
mlflow.set_experiment("soh-prediction")

with mlflow.start_run(run_name="lightgbm-v1"):
    mlflow.log_params({
        "model": "LightGBM",
        "n_estimators": model.best_iteration_,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "split_strategy": "temporal-80-20"
    })
    mlflow.log_metrics({
        "val_rmse": rmse,
        "val_mae": mae,
        "val_r2": r2
    })
    mlflow.lightgbm.log_model(model, "lightgbm-model")
    print(" Model logged to MLflow")

# ── Feature importance ─────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
print("\n=== Feature Importance ===")
print(importance.sort_values(ascending=False))

# ── Save model ─────────────────────────────────────────────
import pickle
with open('src/lightgbm_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n Saved src/lightgbm_model.pkl")