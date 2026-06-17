import pandas as pd

df = pd.read_parquet('data/features.parquet')

# Add cycle number per battery (position in degradation sequence)
df['cycle_number'] = df.groupby('battery_id').cumcount() + 1

# Add rolling capacity mean (last 5 cycles trend)
df['capacity_rolling_mean'] = df.groupby('battery_id')['capacity'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

# Add capacity retention ratio (current / first cycle capacity)
df['capacity_retention'] = df.groupby('battery_id')['capacity'].transform(
    lambda x: x / x.iloc[0]
)

print(f"Updated shape: {df.shape}")
print(df[['battery_id', 'cycle_number', 'capacity_rolling_mean', 'capacity_retention', 'SoH']].head(10))

df.to_parquet('data/features.parquet', index=False)
print("\n✅ Updated features.parquet with cycle features")