import torch
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from modelzoo.NeuralNet import SWE_Net
from dataloader import SWEDataLoader
from utils.utils import back_transform_predictions
import numpy as np
import geoplot as gplt
import geoplot.crs as gcrs
import yaml
from types import SimpleNamespace

with open("config.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.device = torch.device(cfg.device) 

model = SWE_Net(input_dim=14, hidden_dims=cfg.hidden_dims).to(cfg.device)
model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))
model.eval()

loader = SWEDataLoader(cfg)
_, val_loader, transformer, val_df, spatial_obj = loader.prepare()

predictions = []
targets = []

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(cfg.device)
        pred = model(batch_x).cpu().squeeze()
        predictions.extend(pred.numpy())
        targets.extend(batch_y.squeeze().numpy())

y_pred_spatial = back_transform_predictions(
    y_pred_decor=np.array(predictions),
    transformer=transformer,
    spatial_obj=spatial_obj
)
val_df = val_df.copy()
val_df["Predicted_SWE"] = y_pred_spatial
val_df["True_SWE"] = targets

plt.figure(figsize=(8, 6))
plt.scatter(val_df["True_SWE"], val_df["Predicted_SWE"], alpha=0.6)
plt.plot([0, max(val_df["True_SWE"])], [0, max(val_df["True_SWE"])], 'r--')
plt.xlabel("True SWE (in)")
plt.ylabel("Predicted SWE (in)")
plt.title(f"SWE Prediction Accuracy\n{cfg.sample_date}")
plt.grid(True)
plt.tight_layout()
plt.show()

usa = gpd.read_file(gplt.datasets.get_path("contiguous_usa"))

gdf = gpd.GeoDataFrame(
    val_df,
    geometry=gpd.points_from_xy(val_df["Longitude_x"], val_df["Latitude_x"]),
    crs="EPSG:4326"
)

gdf["Abs_Error"] = (gdf["True_SWE"] - gdf["Predicted_SWE"]).abs()

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': gcrs.PlateCarree()})
gplt.polyplot(usa, ax=ax, facecolor='lightgray', edgecolor='black', linewidth=1)
gdf.to_crs(epsg=4326).plot(
    ax=ax,
    column="Abs_Error",
    cmap="coolwarm",
    legend=True,
    markersize=50,
    alpha=0.9
)
plt.title(f"SWE Absolute Error Map\n{cfg.sample_date}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
cbar = ax.get_figure().get_axes()[-1]
cbar.set_ylabel("Absolute Error (in)")
plt.tight_layout()
minx, miny, maxx, maxy = gdf.total_bounds
x_margin = (maxx - minx) * 0.05  
y_margin = (maxy - miny) * 0.05  
ax.set_xlim(minx - x_margin, maxx + x_margin)
ax.set_ylim(miny - y_margin, maxy + y_margin)
#plt.show()