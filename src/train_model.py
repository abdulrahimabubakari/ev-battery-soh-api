import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings('ignore')

# ── Load features ──────────────────────────────────────────
df = pd.read_parquet('data/features.parquet')

# Use only non-leaky features
FEATURE_COLS = [
    'discharge_duration', 'voltage_mean', 'voltage_min', 'voltage_drop',
    'voltage_std', 'current_mean', 'temp_mean', 'temp_max', 'temp_rise',
    'temp_std', 'energy_discharged', 'voltage_slope',
    'time_below_3_5v', 'time_above_35c', 'cycle_number'
]

X = df[FEATURE_COLS].values
y = df['SoH'].values
groups = df['battery_id'].values

# ── Train/val split: last 20% of cycles per battery = validation ──
df['split'] = df.groupby('battery_id')['cycle_number'].transform(
    lambda x: (x > x.quantile(0.8)).astype(int)
)

train_mask = df['split'] == 0
val_mask = df['split'] == 1

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

# ── XGBoost ────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
mae = mean_absolute_error(y_val, preds)
r2 = r2_score(y_val, preds)

print(f"\n=== Validation Results ===")
print(f"  RMSE : {rmse:.4f} % SoH")
print(f"  MAE  : {mae:.4f} % SoH")
print(f"  R²   : {r2:.4f}")

# ── MLflow ─────────────────────────────────────────────────
mlflow.set_experiment("soh-prediction")

with mlflow.start_run(run_name="xgboost-v2-temporal-split"):
    mlflow.log_params({
        "model": "XGBoost",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "split_strategy": "temporal-80-20"
    })
    mlflow.log_metrics({
        "val_rmse": rmse,
        "val_mae": mae,
        "val_r2": r2
    })
    model.fit(X, y)
    mlflow.xgboost.log_model(model, "xgboost-model")
    print("\n Model logged to MLflow")

# ── Feature importance ─────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
print("\n=== Feature Importance ===")
print(importance.sort_values(ascending=False).round(4))

# Save model features list
pd.Series(FEATURE_COLS).to_csv('src/feature_cols.csv', index=False)
print("\n Saved src/feature_cols.csv")