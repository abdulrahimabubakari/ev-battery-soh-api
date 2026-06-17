import pandas as pd
import numpy as np

# Load metadata
meta = pd.read_csv('metadata.csv')

# Discharge cycles only
discharge = meta[meta['type'] == 'discharge'][['battery_id', 'filename', 'Capacity']].dropna()
discharge['Capacity'] = pd.to_numeric(discharge['Capacity'], errors='coerce')
discharge = discharge.dropna(subset=['Capacity'])

# Show all batteries and their cycle counts
print("=== Batteries in dataset ===")
print(discharge.groupby('battery_id')['Capacity'].agg(['count', 'max', 'min']).round(4))

# Compute SoH per battery
discharge = discharge.copy()
discharge['SoH'] = discharge.groupby('battery_id')['Capacity'].transform(lambda x: x / x.max() * 100)

print("\n=== SoH range per battery ===")
print(discharge.groupby('battery_id')['SoH'].agg(['max', 'min']).round(2))

# Save cleaned discharge metadata
discharge.to_csv('data/discharge_meta.csv', index=False)
print("\n Saved data/discharge_meta.csv")