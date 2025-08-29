import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from types import SimpleNamespace
import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd

# Load configuration for metadata path
with open("config.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)

# Load the station metrics that train.py already calculated
station_metrics = pd.read_csv("results/station_metrics.csv")

# Load station metadata for coordinates
station_meta = pd.read_csv(cfg.meta_path)
station_meta["Station_clean"] = station_meta["Station Name"].str.lower()

# Merge metrics with coordinates
station_data = []
for _, row in station_metrics.iterrows():
    station = row['station']
    
    # Find station coordinates
    station_row = station_meta[station_meta["Station_clean"] == station.lower()]
    if not station_row.empty:
        station_data.append({
            'Station': station,
            'NSE': row['nse'],
            'RMSE': row['rmse'],
            'N_predictions': row['n_predictions'],
            'Mean_Actual': row['mean_actual'],
            'Mean_Predicted': row['mean_predicted'],
            'Latitude': station_row.iloc[0]['Latitude_x'],
            'Longitude': station_row.iloc[0]['Longitude_x'],
            'Elevation': station_row.iloc[0]['Elevation_x'],
            'State': station_row.iloc[0]['State Name_x']
        })

# Create DataFrame for plotting
station_df = pd.DataFrame(station_data)

# Create NSE categories for discrete coloring
def categorize_nse(nse):
    if nse <= 0:
        return 'NSE≤0'
    elif nse <= 0.30:
        return '0<NSE≤0.30'
    elif nse <= 0.50:
        return '0.30<NSE≤0.50'
    elif nse <= 0.75:
        return '0.50<NSE≤0.75'
    else:
        return '0.75<NSE≤1'

station_df['NSE_Category'] = station_df['NSE'].apply(categorize_nse)

# Define colors to match your image
nse_colors = {
    'NSE≤0': '#C4B569',           # Exact tan color from your image
    '0<NSE≤0.30': '#FF8C00',       
    '0.30<NSE≤0.50': '#87CEEB', 
    '0.50<NSE≤0.75': '#4682B4',   
    '0.75<NSE≤1': '#191970'       
}

print(f"Loaded NSE data for {len(station_df)} stations")
print(f"NSE Statistics:")
print(f"  Mean: {station_df['NSE'].mean():.3f}")
print(f"  Median: {station_df['NSE'].median():.3f}")
print(f"  Min: {station_df['NSE'].min():.3f}")
print(f"  Max: {station_df['NSE'].max():.3f}")

# Create the map visualization with US outline
fig = plt.figure(figsize=(20, 8))

# Create subplots with different projections
ax1 = plt.subplot(1, 2, 1, projection=gcrs.PlateCarree())
ax2 = plt.subplot(1, 2, 2)

# Load US states shapefile from geoplot
usa = gpd.read_file(gplt.datasets.get_path("contiguous_usa"))

# Plot 1: NSE values on US map
gplt.polyplot(usa, ax=ax1, facecolor='lightgray', edgecolor='black', linewidth=0.5)

# Create GeoDataFrame for the stations
gdf = gpd.GeoDataFrame(
    station_df,
    geometry=gpd.points_from_xy(station_df['Longitude'], station_df['Latitude']),
    crs="EPSG:4326"
)

# Plot stations colored by NSE category
for category, color in nse_colors.items():
    category_data = gdf[gdf['NSE_Category'] == category]
    if not category_data.empty:
        gplt.pointplot(category_data, ax=ax1, color=color, s=5, alpha=0.9, 
                      edgecolor='black', linewidth=0.1, label=category)

ax1.set_title('Nash-Sutcliffe Efficiency by SNOTEL Station', fontsize=14)

# Add legend for NSE categories
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

# Set map extent to focus on western US (where most SNOTEL stations are)
ax1.set_xlim(-125, -95)
ax1.set_ylim(30, 50)

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
plt.savefig('results/nse_spatial_map.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the merged data with coordinates
station_df.to_csv('results/station_metrics_with_coords.csv', index=False)

# Print top and bottom performing stations
print("\nTop 10 performing stations (highest NSE):")
top_stations = station_df.nlargest(10, 'NSE')[['Station', 'NSE', 'State', 'Elevation']]
print(top_stations.to_string(index=False))

print("\nBottom 10 performing stations (lowest NSE):")
bottom_stations = station_df.nsmallest(10, 'NSE')[['Station', 'NSE', 'State', 'Elevation']]
print(bottom_stations.to_string(index=False))

# Additional analysis: NSE by state
print("\nAverage NSE by State:")
state_nse = station_df.groupby('State')['NSE'].agg(['mean', 'count']).sort_values('mean', ascending=False)
state_nse.columns = ['Mean_NSE', 'Station_Count']
print(state_nse.to_string())

# Create horizontal stacked bar chart for NSE distribution
def create_nse_distribution_chart():
    # Count stations in each NSE category
    category_counts = station_df['NSE_Category'].value_counts()
    
    # Ensure all categories are present (even if count is 0) - ordered from high to low NSE (left to right)
    categories = ['0.75<NSE≤1', '0.50<NSE≤0.75', '0.30<NSE≤0.50', '0<NSE≤0.30', 'NSE≤0']
    counts = [category_counts.get(cat, 0) for cat in categories]
    colors_list = [nse_colors[cat] for cat in categories]
    
    # Create figure for stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Create horizontal stacked bar
    left = 0
    for i, (count, color, category) in enumerate(zip(counts, colors_list, categories)):
        ax.barh('LR', count, left=left, color=color, label=category, height=0.5)
        left += count
    
    # Customize the plot
    ax.set_xlabel('Station Count', fontsize=12)
    ax.set_xlim(0, sum(counts))
    ax.set_ylim(-0.5, 0.5)
    
    # Remove y-axis ticks since we only have one bar
    ax.set_yticks([0])
    ax.set_yticklabels(['LR'])
    
    # Add vertical lines to separate categories
    cumsum = np.cumsum([0] + counts[:-1])
    for pos in cumsum[1:]:
        ax.axvline(x=pos, color='white', linewidth=2)
    
    # Add red dashed line at NSE = 0.5 threshold (right of 0.50<NSE≤0.75 category)
    # The categories are ordered: ['0.75<NSE≤1', '0.50<NSE≤0.75', '0.30<NSE≤0.50', '0<NSE≤0.30', 'NSE≤0']
    # So the 0.5 line goes after the first two categories (0.75<NSE≤1 + 0.50<NSE≤0.75)
    nse_05_position = counts[0] + counts[1]  # Sum of first two categories
    ax.axvline(x=nse_05_position, color='red', linewidth=2, linestyle='--', alpha=0.8)
    
    # Add legend below the plot with more space
    ax.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=5, 
              frameon=False, fontsize=10)
    
    # Add title
    ax.set_title('Distribution of Stations by NSE Performance Category', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Add extra space at bottom for legend
    plt.savefig('results/nse_distribution_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print the counts
    total_stations = sum(counts)
    print(f"\nNSE Distribution Summary ({total_stations} total stations):")
    for category, count in zip(categories, counts):
        percentage = (count/total_stations)*100 if total_stations > 0 else 0
        print(f"{category:15}: {count:3d} stations ({percentage:5.1f}%)")

# Create the distribution chart
create_nse_distribution_chart()
