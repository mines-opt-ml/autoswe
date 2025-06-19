import torch
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

from modelzoo.NeuralNet import SWE_Net
from dataloader import SWEDataLoader
from config import SWEConfig

#Config file
cfg = SWEConfig(
    swe_path="data/SWE_data_CONUS.csv",
    meta_path="data/Snotel_Locations_Filtered_v3.csv",
    sample_date="1/5/2000",
    device=torch.device("cpu")  # use "cuda" if available
)

#model weights - dim=7 to reflect parameters (lat, long, slope, elevation, aspect, day,)
model = SWE_Net(input_dim=7, hidden_dims=cfg.hidden_dims).to(cfg.device)
model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))
model.eval()

# --- Load Validation Data ---
loader = SWEDataLoader(cfg)
_, val_loader, _, val_df = loader.prepare()

# --- Run Predictions ---
predictions = []
targets = []

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(cfg.device)
        pred = model(batch_x).cpu().squeeze()
        predictions.extend(pred.numpy())
        targets.extend(batch_y.squeeze().numpy())

val_df = val_df.copy()
val_df["Predicted_SWE"] = predictions
val_df["True_SWE"] = targets

# --- Plot: Scatter True vs Predicted ---
plt.figure(figsize=(8, 6))
plt.scatter(val_df["True_SWE"], val_df["Predicted_SWE"], alpha=0.6)
plt.plot([0, max(val_df["True_SWE"])], [0, max(val_df["True_SWE"])], 'r--')
plt.xlabel("True SWE (in)")
plt.ylabel("Predicted SWE (in)")
plt.title(f"SWE Prediction Accuracy\n{cfg.sample_date}")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot: Map View of Errors ---
gdf = gpd.GeoDataFrame(
    val_df,
    geometry=gpd.points_from_xy(val_df["Lon"], val_df["Lat"]),
    crs="EPSG:4326"
)

gdf["Abs_Error"] = (gdf["True_SWE"] - gdf["Predicted_SWE"]).abs()

fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(column="Abs_Error", cmap="coolwarm", legend=True, ax=ax, markersize=50)
plt.title(f"SWE Absolute Error Map\n{cfg.sample_date}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()