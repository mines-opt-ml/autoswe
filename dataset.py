from typing import Dict
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.preprocess import preprocess


class SWEStationDataset(Dataset):
    """
    The dataset class should handle loading the csv files and processing the data.
    Each item in the dataset should be (X: torch.Tensor[num_features, window_length], y: torch.Tensor[, window_length])
    where X is the features and y is the target SWE value.
    Windoow_length is the length of a snow year.
    """

    def __init__(self, cfg):
        super(SWEStationDataset, self).__init__()
        self.cfg = cfg
        self.dynamic_forcing_and_swe, self.snotel_attributes = preprocess(cfg)
        
        self.beginning_of_snow_year = self.cfg.beginning_of_snow_year
        self.end_of_snow_year = self.cfg.end_of_snow_year
        self.beginning_year = self.cfg.beginning_year
        self.end_year = self.cfg.end_year
        self.dynamic_forcing_and_swe, self.snotel_attributes = preprocess(cfg)
        
        self.dynamic_forcing_and_swe = self.dynamic_forcing_and_swe[
            (self.dynamic_forcing_and_swe['Date'] >= f"{cfg.beginning_year}-{cfg.beginning_of_snow_year}") &
            (self.dynamic_forcing_and_swe['Date'] <= f"{cfg.end_year}-12-31")
        ]
        
        self.dynamic_forcing_and_swe["Date"] = pd.to_datetime(self.dynamic_forcing_and_swe["Date"])
        self.lookup_table = []
        years = pd.to_datetime(self.dynamic_forcing_and_swe["Date"]).dt.year.unique()
        
        for station in self.snotel_attributes["Station"]:
            for year in years:
                start_date = f"{year}-{cfg.beginning_of_snow_year}"
                end_of_snow_year_date = f"{year+1}-{cfg.end_of_snow_year}"
                
                station_data = self.dynamic_forcing_and_swe[
                    (self.dynamic_forcing_and_swe["Station"] == station) & 
                    (self.dynamic_forcing_and_swe["Date"] >= start_date) & 
                    (self.dynamic_forcing_and_swe["Date"] <= end_of_snow_year_date)
                ]
                
                if len(station_data) > 0:
                    self.lookup_table.append({
                        "Station": station,
                        "Year": year,
                        "StartDate": pd.Timestamp(start_date),
                        "EndOfSnowYearDate": pd.Timestamp(end_of_snow_year_date)
                    })
        
        
        self.dynamic_forcing_and_swe.set_index(['Station', 'Date'], inplace=True)
        
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Elevation": df["Elevation_x"],
                "Slope": df["Slope_tif1_x"],
                "Aspect": df["Aspect_tif_x"],
                "Latitude": df["Latitude_x"],
                "Longitude": df["Longitude_x"],
                "DayOfYear": pd.to_datetime(df["Date"]).dt.dayofyear,
                "Tmax": df["Tmax"],
                "Tmin": df["Tmin"],
                "Precip": df["Precip"],
                "Tobs": df["Tobs"],
                "TB_19": df["TB_19"],
                "TB_37": df["TB_37"],
                "TB_diff": df["TB_diff"],
            }
        )

    def _create_lookup_table(self):
        lookup = []
        stations = self.dynamic_forcing_and_swe["Station"].drop_duplicates().values

        for year in range(self.beginning_year, self.end_year - 1):
            for station in stations:
                lookup.append(
                    {
                        "Station": station,
                        "Year": year,
                        "StartDate": pd.to_datetime(f"{year}-{self.beginning_of_snow_year}"),
                        "EndOfSnowYearDate": pd.to_datetime(f"{year + 1}-{self.end_of_snow_year}"),
                    }
                )
        return lookup

    def __len__(self):
        return len(self.lookup_table)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        station = self.lookup_table[idx]["Station"]
        start_date = self.lookup_table[idx]["StartDate"]
        end_of_snow_year_date = self.lookup_table[idx]["EndOfSnowYearDate"]
        year = self.lookup_table[idx]["Year"]

        sample = {}
        mask = (
            (self.dynamic_forcing_and_swe["Station"] == station)
            & (self.dynamic_forcing_and_swe["Date"] >= pd.Timestamp(start_date))
            & (self.dynamic_forcing_and_swe["Date"] <= pd.Timestamp(end_of_snow_year_date))
        )

        data = self.dynamic_forcing_and_swe[mask].copy()  

        if len(data) == 0:
            raise ValueError(f"No data found for station {station} between {start_date} and {end_of_snow_year_date}")

        features = pd.DataFrame(
            {
                "Elevation": data["Elevation_x"],
                "Slope": data["Slope_tif1_x"],
                "Aspect": data["Aspect_tif_x"],
                "Latitude": data["Latitude_x"],
                "Longitude": data["Longitude_x"],
                "DayOfYear": pd.to_datetime(data["Date"]).dt.dayofyear,
                "Tmax": data["Tmax"],
                "Tmin": data["Tmin"],
                "Precip": data["Precip"],
                "Tobs": data["Tobs"],
                "TB_19": data["TB_19"],
                "TB_37": data["TB_37"],
                "TB_diff": data["TB_diff"],
            }
        )

        sample["dynamic forcing"] = torch.tensor(features.values, dtype=torch.float32)
        sample["swe"] = torch.tensor(data["SWE"].values, dtype=torch.float32)
        sample["dates"] = data["Date"].dt.strftime("%Y-%m-%d").values
        station_mask = self.snotel_attributes["Station"] == station
        attrs = self.snotel_attributes.loc[station_mask, ["Elevation", "Slope", "Aspect"]].values
        sample["snotel attributes"] = torch.tensor(attrs, dtype=torch.float32)
        sample["year"] = year
        sample["station"] = station

        return sample

    def get(self, station: str, year: int) -> Dict[str, torch.Tensor]:
        """
        Get data for a specific station and year.

        Args:
            station: Station identifier
            year: Year to retrieve data for

        Returns:
            Dictionary containing dynamic forcing, SWE, and station attributes
        """
        for idx, entry in enumerate(self.lookup_table):
            if entry["Station"] == station and entry["Year"] == year:
                return self.__getitem__(idx)

        raise ValueError(f"No data found for station {station} and year {year}")