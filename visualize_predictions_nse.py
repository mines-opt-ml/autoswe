import torch
import matplotlib.pyplot as plt
import pandas as pd
from modelzoo.LSTM import SWE_Net
from dataloader import SWEDataLoader
import numpy as np
import yaml
from types import SimpleNamespace
import pickle
import os

def calculate_nse(y_true, y_pred):
    """Calculate Nash-Sutcliffe Efficiency"""
    mean_observed = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - mean_observed) ** 2)
    if denominator == 0:
        return np.nan
    nse = 1 - (numerator / denominator)
    return nse

# Load configuration
with open("config.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.device = torch.device(cfg.device) 

# Load station metadata for coordinates
station_meta = pd.read_csv(cfg.meta_path)
station_meta["Station_clean"] = station_meta["Station Name"].str.lower()

# Load the trained model - need to check if station_stats are available
pickle_file = "backtransform_info.pkl"
if not os.path.exists(pickle_file):
    print("Backtransform info not found. Please run training first.")
    exit(1)

# Load cached data
with open(pickle_file, "rb") as f:
    train_loader, val_loader, _, bt_info = pickle.load(f)

# Get station stats for model initialization
station_stats = {}
if "station_stats" in bt_info:
    station_stats = bt_info["station_stats"]

model = SWE_Net(cfg, station_stats=station_stats).to(cfg.device)
model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))
model.eval()

# Collect predictions and targets by station
station_predictions = {}
station_targets = {}

print("Collecting predictions from validation data...")

with torch.no_grad():
    for batch in val_loader:
        X = batch["dynamic forcing"].to(cfg.device)
        y = batch["swe"].cpu().numpy()
        stations = batch["station"]
        
        pred = model(X, stations=stations).cpu().numpy()
        
        # Group by station
        for i, station in enumerate(stations):
            if station not in station_predictions:
                station_predictions[station] = []
                station_targets[station] = []
            
            station_predictions[station].extend(pred[i])
            station_targets[station].extend(y[i])

# Calculate NSE for each station
station_nse = {}
station_data = []

print("Calculating NSE for each station...")

for station in station_predictions.keys():
    if len(station_predictions[station]) > 10:  # Only calculate NSE if enough data points
        nse = calculate_nse(
            np.array(station_targets[station]), 
            np.array(station_predictions[station])
        )
        station_nse[station] = nse
        
        # Find station coordinates
        station_row = station_meta[station_meta["Station_clean"] == station.lower()]
        if not station_row.empty:
            station_data.append({
                'Station': station,
                'NSE': nse,
                'Latitude': station_row.iloc[0]['Latitude_x'],
                'Longitude': station_row.iloc[0]['Longitude_x'],
                'Elevation': station_row.iloc[0]['Elevation_x'],
                'State': station_row.iloc[0]['State Name_x']
            })

# Create DataFrame for plotting
station_df = pd.DataFrame(station_data)

print(f"Calculated NSE for {len(station_df)} stations")
print(f"NSE Statistics:")
print(f"  Mean: {station_df['NSE'].mean():.3f}")
print(f"  Median: {station_df['NSE'].median():.3f}")
print(f"  Min: {station_df['NSE'].min():.3f}")
print(f"  Max: {station_df['NSE'].max():.3f}")

# Create the map visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: NSE values on map
scatter = ax1.scatter(station_df['Longitude'], station_df['Latitude'], 
                     c=station_df['NSE'], cmap='RdYlBu', 
                     s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

ax1.set_xlim(-125, -65)  # US longitude bounds
ax1.set_ylim(25, 50)     # US latitude bounds
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('Nash-Sutcliffe Efficiency by SNOTEL Station')
ax1.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Nash-Sutcliffe Efficiency')

# Plot 2: NSE histogram
ax2.hist(station_df['NSE'], bins=30, alpha=0.7, edgecolor='black')
ax2.axvline(station_df['NSE'].mean(), color='red', linestyle='--', 
           label=f'Mean: {station_df["NSE"].mean():.3f}')
ax2.axvline(station_df['NSE'].median(), color='orange', linestyle='--', 
           label=f'Median: {station_df["NSE"].median():.3f}')
ax2.set_xlabel('Nash-Sutcliffe Efficiency')
ax2.set_ylabel('Number of Stations')
ax2.set_title('Distribution of NSE Values')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/nse_spatial_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save detailed results
station_df.to_csv('results/station_nse_results.csv', index=False)

# Print top and bottom performing stations
print("\nTop 10 performing stations (highest NSE):")
top_stations = station_df.nlargest(10, 'NSE')[['Station', 'NSE', 'State', 'Elevation']]
print(top_stations.to_string(index=False))

print("\nBottom 10 performing stations (lowest NSE):")
bottom_stations = station_df.nsmallest(10, 'NSE')[['Station', 'NSE', 'State', 'Elevation']]
print(bottom_stations.to_string(index=False))
