from typing import Dict, Tuple

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
        # self.X = torch.tensor(X, dtype=torch.float32)
        # self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.beginning_of_snow_year = self.cfg.beginning_of_snow_year
        self.peak_swe_date = self.cfg.peak_swe_date
        self.beginning_year = self.cfg.beginning_year
        self.end_year = self.cfg.end_year
        self.dynamic_forcing_and_swe, self.snotel_attributes = preprocess(self.cfg)
        self.dynamic_forcing_and_swe["Date"] = pd.to_datetime(self.dynamic_forcing_and_swe["Date"])
        self.lookup_table = self._create_lookup_table()
        self._get_start_and_end_dates()  # Todo: implement this method. Should read these dates from the config

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

    # Add stuff from dataloader
    def prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dynamic_forcing_and_swe, snotel_attributes = preprocess(self.cfg)
        # daily = merged[merged["Date"] == self.cfg.sample_date].copy()
        # train_df, val_df = train_test_split(daily, test_size=0.2, random_state=42)

        # self.X = build_features(daily)
        # self.y = daily["SWE"].values

    def _create_lookup_table(self):
        lookup = []
        for year in range(self.beginning_year, self.end_year - 1):
            for station in self.dynamic_forcing_and_swe["Station"].unique():
                lookup.append(
                    {
                        "Station": station,
                        "Year": year,
                        "StartDate": pd.to_datetime(f"{year}-{self.beginning_of_snow_year}"),
                        "PeakSWEDate": pd.to_datetime(f"{year}-{self.peak_swe_date}"),
                    }
                )
        return lookup

    # and also slice time series into water year

    def __len__(self):
        return len(self.lookup_table)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        station = self.lookup_table[idx]["Station"]
        year = self.lookup_table[idx]["Year"]
        start_date = self.lookup_table[idx]["StartDate"]
        peak_swe_date = self.lookup_table[idx]["PeakSWEDate"]

        sample = {}

        # Filter by station and date range
        mask = (
            (self.dynamic_forcing_and_swe["Station"] == station)
            & (self.dynamic_forcing_and_swe["Date"] >= start_date)
            & (self.dynamic_forcing_and_swe["Date"] <= peak_swe_date)
        )

        sample["dynamic forcing"] = self.dynamic_forcing_and_swe[mask].drop(columns=["SWE"])
        sample["swe"] = self.dynamic_forcing_and_swe[mask]["SWE"].values
        # TODO: These should be torch.Tensor
        sample["snotel attributes"] = self.snotel_attributes[self.snotel_attributes["Station"] == station]
        # should return time series of all forcings for a given snotel from beginning of snow year to peak swe date
        return sample

    def _get_(self):
        raise NotImplementedError
