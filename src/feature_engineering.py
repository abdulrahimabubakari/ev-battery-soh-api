import pandas as pd
import numpy as np
import os
from pathlib import Path

def extract_features_from_cycle(filepath):
    """Extract physics-informed features from a single discharge cycle CSV."""
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        
        # Drop rows with nulls
        df = df.dropna()
        if len(df) < 10:
            return None

        v = df['Voltage_measured'].values
        i = df['Current_measured'].values
        t = df['Temperature_measured'].values
        time = df['Time'].values

        features = {}

        # 1. Discharge duration (seconds)
        features['discharge_duration'] = time[-1] - time[0]

        # 2. Average voltage
        features['voltage_mean'] = np.mean(v)

        # 3. Minimum voltage reached
        features['voltage_min'] = np.min(v)

        # 4. Voltage drop (start - end)
        features['voltage_drop'] = v[0] - v[-1]

        # 5. Voltage std (spread)
        features['voltage_std'] = np.std(v)

        # 6. Average current
        features['current_mean'] = np.mean(np.abs(i))

        # 7. Average temperature
        features['temp_mean'] = np.mean(t)

        # 8. Max temperature
        features['temp_max'] = np.max(t)

        # 9. Temperature rise
        features['temp_rise'] = t[-1] - t[0]

        # 10. Temperature std
        features['temp_std'] = np.std(t)

        # 11. Energy discharged (V * I * dt approximation)
        dt = np.diff(time)
        power = np.abs(i[:-1]) * v[:-1]
        features['energy_discharged'] = np.sum(power * dt)

        # 12. Voltage slope (linear degradation rate)
        if len(time) > 1:
            features['voltage_slope'] = np.polyfit(time, v, 1)[0]
        else:
            features['voltage_slope'] = 0

        # 13. Time spent below 3.5V (low voltage stress)
        features['time_below_3_5v'] = np.sum(v < 3.5) / len(v)

        # 14. Time spent above 35°C (thermal stress)
        features['time_above_35c'] = np.sum(t > 35) / len(t)

        return features

    except Exception as e:
        return None


def build_feature_matrix():
    """Build full feature matrix from all discharge cycles."""
    
    discharge_meta = pd.read_csv('data/discharge_meta.csv')
    
    all_features = []
    total = len(discharge_meta)
    
    print(f"Processing {total} discharge cycles...")
    
    for idx, row in discharge_meta.iterrows():
        filepath = Path('data') / row['filename']
        
        if not filepath.exists():
            continue
        
        features = extract_features_from_cycle(filepath)
        
        if features is not None:
	    features['cycle_index'] = idx
            features['battery_id'] = row['battery_id']
            features['filename'] = row['filename']
            features['capacity'] = row['Capacity']
            features['SoH'] = row['SoH']
            all_features.append(features)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total}...")
    
    df = pd.DataFrame(all_features)
    print(f"\n Feature matrix shape: {df.shape}")
    print(f"Features: {[c for c in df.columns if c not in ['battery_id','filename','capacity','SoH']]}")
    print(f"\nSoH range: {df['SoH'].min():.2f} to {df['SoH'].max():.2f}")
    print(f"Null values:\n{df.isnull().sum()}")
    
    # Save
    df.to_parquet('data/features.parquet', index=False)
    print("\n Saved data/features.parquet")
    return df


if __name__ == '__main__':
    df = build_feature_matrix()
    print("\nSample rows:")
    print(df.head())