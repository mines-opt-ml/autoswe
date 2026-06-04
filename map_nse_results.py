import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
from types import SimpleNamespace
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
from geopandas.datasets import get_path
import cartopy.crs as ccrs

with open("config.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)

cfg = SimpleNamespace(**cfg_dict)

station_metrics = pd.read_csv("results/test_station_metrics.csv")
station_meta = pd.read_csv(cfg.meta_path)

station_meta["Station_clean"] = station_meta["Station Name"].str.lower()

station_data = []

for _, row in station_metrics.iterrows():
    station = row["station"]

    station_row = station_meta[
        station_meta["Station_clean"] == station.lower()
    ]

    if not station_row.empty:
        station_data.append({
            "Station": station,
            "NSE": row["nse"],
            "RMSE": row["rmse"],
            "N_predictions": row["n_predictions"],
            "Latitude": station_row.iloc[0]["Latitude_x"],
            "Longitude": station_row.iloc[0]["Longitude_x"],
            "Elevation": station_row.iloc[0]["Elevation_x"],
            "State": station_row.iloc[0]["State Name_x"]
        })

station_df = pd.DataFrame(station_data)

print(f"Loaded NSE data for {len(station_df)} stations")
print("NSE Statistics:")
print(f"  Mean:   {station_df['NSE'].mean():.3f}")
print(f"  Median: {station_df['NSE'].median():.3f}")
print(f"  Min:    {station_df['NSE'].min():.3f}")
print(f"  Max:    {station_df['NSE'].max():.3f}")


gdf = gpd.GeoDataFrame(
    station_df,
    geometry=gpd.points_from_xy(
        station_df["Longitude"],
        station_df["Latitude"]
    ),
    crs="EPSG:4326"
)


usa = gpd.read_file(get_path("naturalearth_lowres"), engine="pyogrio")
usa = usa[usa["continent"] == "North America"]

with open("us_states.json", "r") as f:
    states_geojson = json.load(f)

states = gpd.GeoDataFrame.from_features(
    states_geojson["features"],
    crs="EPSG:4326"
)

fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(1, 1, 1, projection=gcrs.PlateCarree())

gplt.polyplot(
    usa,
    ax=ax,
    facecolor="lightgray",
    edgecolor="black",
    linewidth=0.5
)

states.boundary.plot(
    ax=ax,
    linewidth=0.7,
    color="black",
    alpha=0.8
)

scatter = ax.scatter(
    gdf["Longitude"],
    gdf["Latitude"],
    c=gdf["NSE"],
    cmap="turbo",
    s=22,
    edgecolors="black",
    linewidths=0.25,
    alpha=0.95,
    vmin=-1.0,
    vmax=1.0,
    transform=ccrs.PlateCarree()
)

cbar = plt.colorbar(
    scatter,
    ax=ax,
    shrink=0.75,
    pad=0.02
)

cbar.set_label("Nash-Sutcliffe Efficiency (NSE)", fontsize=11)

ax.set_title(
    "Nash-Sutcliffe Efficiency by SNOTEL Station",
    fontsize=14
)

ax.set_xlim(-125, -102)
ax.set_ylim(31, 50)

plt.tight_layout()
plt.savefig("results/nse_spatial_map.png", dpi=300, bbox_inches="tight")
plt.show()

station_df.to_csv("results/station_metrics_with_coords.csv", index=False)

print("\nTop 10 performing stations (highest NSE):")
top_stations = station_df.nlargest(10, "NSE")[
    ["Station", "NSE", "State", "Elevation"]
]
print(top_stations.to_string(index=False))

print("\nBottom 10 performing stations (lowest NSE):")
bottom_stations = station_df.nsmallest(10, "NSE")[
    ["Station", "NSE", "State", "Elevation"]
]
print(bottom_stations.to_string(index=False))

print("\nAverage NSE by State:")
state_nse = (
    station_df
    .groupby("State")["NSE"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

state_nse.columns = ["Mean_NSE", "Station_Count"]
print(state_nse.to_string())

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import yaml
# from types import SimpleNamespace
# import geoplot as gplt
# import geoplot.crs as gcrs
# import geopandas as gpd
# from geopandas.datasets import get_path
# import json

# with open("config.yaml", "r") as f:
#     cfg_dict = yaml.safe_load(f)
# cfg = SimpleNamespace(**cfg_dict)

# station_metrics = pd.read_csv("results/test_station_metrics.csv")

# station_meta = pd.read_csv(cfg.meta_path)
# station_meta["Station_clean"] = station_meta["Station Name"].str.lower()

# station_data = []
# for _, row in station_metrics.iterrows():
#     station = row['station']
    
#     station_row = station_meta[station_meta["Station_clean"] == station.lower()]
#     if not station_row.empty:
#         station_data.append({
#             'Station': station,
#             'NSE': row['nse'],
#             'RMSE': row['rmse'],
#             'N_predictions': row['n_predictions'],
#             'Latitude': station_row.iloc[0]['Latitude_x'],
#             'Longitude': station_row.iloc[0]['Longitude_x'],
#             'Elevation': station_row.iloc[0]['Elevation_x'],
#             'State': station_row.iloc[0]['State Name_x']
#         })

# station_df = pd.DataFrame(station_data)

# def categorize_nse(nse):
#     if nse <= 0:
#         return 'NSE≤0'
#     elif nse <= 0.30:
#         return '0<NSE≤0.30'
#     elif nse <= 0.50:
#         return '0.30<NSE≤0.50'
#     elif nse <= 0.75:
#         return '0.50<NSE≤0.75'
#     else:
#         return '0.75<NSE≤1'

# station_df['NSE_Category'] = station_df['NSE'].apply(categorize_nse)

# nse_colors = {
#     'NSE≤0': '#C4B569',          
#     '0<NSE≤0.30': '#FF8C00',       
#     '0.30<NSE≤0.50': '#87CEEB', 
#     '0.50<NSE≤0.75': '#4682B4',   
#     '0.75<NSE≤1': '#191970'       
# }

# print(f"Loaded NSE data for {len(station_df)} stations")
# print(f"NSE Statistics:")
# print(f"  Mean: {station_df['NSE'].mean():.3f}")
# print(f"  Median: {station_df['NSE'].median():.3f}")
# print(f"  Min: {station_df['NSE'].min():.3f}")
# print(f"  Max: {station_df['NSE'].max():.3f}")

# fig = plt.figure(figsize=(12, 8))
# ax1 = plt.subplot(1, 1, 1, projection=gcrs.PlateCarree())
# #ax2 = plt.subplot(1, 2, 2)

# usa = gpd.read_file(get_path("naturalearth_lowres"), engine="pyogrio")
# usa = usa[usa["continent"] == "North America"]
# gplt.polyplot(usa, ax=ax1, facecolor='lightgray', edgecolor='black', linewidth=0.5)
# with open("us_states.json", "r") as f:
#     states_geojson = json.load(f)
# states = gpd.GeoDataFrame.from_features(states_geojson["features"], crs="EPSG:4326")
# states.boundary.plot(ax=ax1, linewidth=0.5, color="black", alpha=0.7)
# gdf = gpd.GeoDataFrame(
#     station_df,
#     geometry=gpd.points_from_xy(station_df['Longitude'], station_df['Latitude']),
#     crs="EPSG:4326"
# )

# for category, color in nse_colors.items():
#     category_data = gdf[gdf['NSE_Category'] == category]
#     if not category_data.empty:
#         gplt.pointplot(category_data, ax=ax1, color=color, s=5, alpha=0.9, 
#                       edgecolor='black', linewidth=0.1, label=category)

# ax1.set_title('Nash-Sutcliffe Efficiency by SNOTEL Station', fontsize=14)

# ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

# ax1.set_xlim(-125, -95)
# ax1.set_ylim(30, 50)

# # ax2.hist(station_df['NSE'], bins=30, alpha=0.7, edgecolor='black')
# # ax2.axvline(station_df['NSE'].mean(), color='red', linestyle='--', 
# #            label=f'Mean: {station_df["NSE"].mean():.3f}')
# # ax2.axvline(station_df['NSE'].median(), color='orange', linestyle='--', 
# #            label=f'Median: {station_df["NSE"].median():.3f}')
# # ax2.set_xlabel('Nash-Sutcliffe Efficiency')
# # ax2.set_ylabel('Number of Stations')
# # ax2.set_title('Distribution of NSE Values')
# # ax2.legend()
# # ax2.grid(True, alpha=0.3)

# # plt.tight_layout()
# # plt.savefig('results/nse_spatial_map.png', dpi=300, bbox_inches='tight')
# # plt.show()

# # station_df.to_csv('results/station_metrics_with_coords.csv', index=False)

# # print("\nTop 10 performing stations (highest NSE):")
# # top_stations = station_df.nlargest(10, 'NSE')[['Station', 'NSE', 'State', 'Elevation']]
# # print(top_stations.to_string(index=False))

# # print("\nBottom 10 performing stations (lowest NSE):")
# # bottom_stations = station_df.nsmallest(10, 'NSE')[['Station', 'NSE', 'State', 'Elevation']]
# # print(bottom_stations.to_string(index=False))

# # print("\nAverage NSE by State:")
# # state_nse = station_df.groupby('State')['NSE'].agg(['mean', 'count']).sort_values('mean', ascending=False)
# # state_nse.columns = ['Mean_NSE', 'Station_Count']
# # print(state_nse.to_string())

# # def create_nse_distribution_chart():
# #     category_counts = station_df['NSE_Category'].value_counts()
# #     categories = ['0.75<NSE≤1', '0.50<NSE≤0.75', '0.30<NSE≤0.50', '0<NSE≤0.30', 'NSE≤0']
# #     counts = [category_counts.get(cat, 0) for cat in categories]
# #     colors_list = [nse_colors[cat] for cat in categories]
    
# #     fig, ax = plt.subplots(figsize=(12, 3))
# #     left = 0
# #     for i, (count, color, category) in enumerate(zip(counts, colors_list, categories)):
# #         ax.barh('LR', count, left=left, color=color, label=category, height=0.5)
# #         left += count
    
# #     ax.set_xlabel('Station Count', fontsize=12)
# #     ax.set_xlim(0, sum(counts))
# #     ax.set_ylim(-0.5, 0.5)
# #     ax.set_yticks([0])
# #     ax.set_yticklabels(['LR'])
    
# #     cumsum = np.cumsum([0] + counts[:-1])
# #     for pos in cumsum[1:]:
# #         ax.axvline(x=pos, color='white', linewidth=2)
    
# #     nse_05_position = counts[0] + counts[1]  
# #     ax.axvline(x=nse_05_position, color='red', linewidth=2, linestyle='--', alpha=0.8)
    
# #     ax.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=5, 
# #               frameon=False, fontsize=10)
    
# #     ax.set_title('Distribution of Stations by NSE Performance Category', 
# #                  fontsize=14, pad=20)
    
# #     plt.tight_layout()
# #     plt.subplots_adjust(bottom=0.25)  
# #     plt.savefig('results/nse_distribution_chart.png', dpi=300, bbox_inches='tight')
# #     plt.show()
    
# #     total_stations = sum(counts)
# #     print(f"\nNSE Distribution Summary ({total_stations} total stations):")
# #     for category, count in zip(categories, counts):
# #         percentage = (count/total_stations)*100 if total_stations > 0 else 0
# #         print(f"{category:15}: {count:3d} stations ({percentage:5.1f}%)")

# # create_nse_distribution_chart()
