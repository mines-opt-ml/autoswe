from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.normalizer import Normalizer
from utils.preprocess import preprocess
from utils.snowyear_DOY_conversion import compute_day_of_snow_year


class SWEStationDataset(Dataset):
    """
    Dataset class for SWE prediction.
    Each item = (X: torch.Tensor[num_features, window_length],
                 y: torch.Tensor[window_length])
    Window_length = length of a snow year.
    """

    def __init__(self, cfg):
        super(SWEStationDataset, self).__init__()
        self.cfg = cfg

        self.beginning_of_snow_year = self.cfg.beginning_of_snow_year
        self.end_of_snow_year = self.cfg.end_of_snow_year
        self.global_start_year = min(self.cfg.train_start_year, self.cfg.val_start_year, self.cfg.test_start_year)
        self.global_end_year = max(self.cfg.train_end_year, self.cfg.val_end_year, self.cfg.test_end_year)
        self.dynamic_forcing_and_swe, self.snotel_attributes = preprocess(cfg)

        self.dynamic_forcing_and_swe = self.dynamic_forcing_and_swe.apply(
            lambda col: col.astype(float) if col.dtype in ["int64", "int32"] else col
        )

        self.obs_swe = self.dynamic_forcing_and_swe[["Station", "Date", "SWE"]].copy()

        years = pd.to_datetime(self.dynamic_forcing_and_swe["Date"]).dt.year
        train_mask = (years >= cfg.train_start_year) & (years <= cfg.train_end_year)

        self.station_normalizers = {}
        self.swe_normalizers = {}
        normalization_method = getattr(cfg, "normalization", "zscore")

        for station in self.dynamic_forcing_and_swe["Station"].unique():
            station_mask = (self.dynamic_forcing_and_swe["Station"] == station) & train_mask

            train_inputs = (
                self.dynamic_forcing_and_swe.loc[station_mask].drop(columns=["SWE"]).select_dtypes(include=[float, int])
            )
            input_norm = Normalizer(method=normalization_method)
            if len(train_inputs) > 0:
                input_norm.fit(train_inputs)
            self.station_normalizers[station] = input_norm

            train_targets = self.dynamic_forcing_and_swe.loc[station_mask, ["SWE"]]
            swe_norm = Normalizer(method=normalization_method)
            if len(train_targets) > 0:
                swe_norm.fit(train_targets)
            self.swe_normalizers[station] = swe_norm

        inputs_normed = []
        for station, norm in self.station_normalizers.items():
            station_mask = self.dynamic_forcing_and_swe["Station"] == station
            inputs = (
                self.dynamic_forcing_and_swe.loc[station_mask]
                .drop(columns=["SWE"])
                .select_dtypes(include=[float, int])
                .astype(float)
            )
            normed = norm.transform(inputs)
            normed["Station"] = station
            normed["Date"] = self.dynamic_forcing_and_swe.loc[station_mask, "Date"].values
            inputs_normed.append(normed)

        inputs_normed_df = pd.concat(inputs_normed, axis=0)
        self.dynamic_forcing_and_swe.update(inputs_normed_df)

        swe_normed = []
        for station, norm in self.swe_normalizers.items():
            station_mask = self.dynamic_forcing_and_swe["Station"] == station
            swe = self.dynamic_forcing_and_swe.loc[station_mask, ["SWE"]].astype(float)
            normed = norm.transform(swe)
            normed["Station"] = station
            normed["Date"] = self.dynamic_forcing_and_swe.loc[station_mask, "Date"].values
            swe_normed.append(normed)

        swe_normed_df = pd.concat(swe_normed, axis=0)
        self.dynamic_forcing_and_swe.update(swe_normed_df)

        start = pd.to_datetime(f"{self.global_start_year}-{self.beginning_of_snow_year}")
        end = pd.to_datetime(f"{self.global_end_year}-12-31")
        self.dynamic_forcing_and_swe = self.dynamic_forcing_and_swe[
            (self.dynamic_forcing_and_swe["Date"] >= start) & (self.dynamic_forcing_and_swe["Date"] <= end)
        ].copy()

        valid_stations = self.dynamic_forcing_and_swe["Station"].unique()
        self.snotel_attributes = self.snotel_attributes[self.snotel_attributes["Station"].isin(valid_stations)].copy()

        self.dynamic_forcing_and_swe["Date"] = pd.to_datetime(self.dynamic_forcing_and_swe["Date"])
        self.lookup_table = self._create_lookup_table()


    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct feature DataFrame from raw data.
        Static station attributes are always included.
        Meteorological forcings are controlled via cfg.dynamic_features.
        """
        dynamic_cols = getattr(
            self.cfg,
            "dynamic_features",
            ["Tmax", "Tmin", "Precip", "Tobs", "TB_19", "TB_37", "TB_diff"],  
        )

        feat_dict = {
            "Elevation": df["Elevation_x"],
            "Slope": df["Slope_tif1_x"],
            "Aspect": df["Aspect_tif_x"],
            "Latitude": df["Latitude_x"],
            "Longitude": df["Longitude_x"],
            "DayOfYear": df["Date"].apply(lambda d: compute_day_of_snow_year(d, self.beginning_of_snow_year)),
        }

        for c in dynamic_cols:
            feat_dict[c] = df[c]

        lag = int(getattr(self.cfg, "lag_days", 0))
        if lag > 0:
            feat_dict[f"SWE_lag_{lag}"] = df["SWE"].shift(lag).fillna(0.0)

        return pd.DataFrame(feat_dict)

    def _create_lookup_table(self):
        lookup = []
        valid_stations = self.dynamic_forcing_and_swe["Station"].unique()
        years = range(self.global_start_year, self.global_end_year + 1)

        for station in valid_stations:
            for year in years:
                start_date = pd.to_datetime(f"{year}-{self.beginning_of_snow_year}")
                end_of_snow_year_date = pd.to_datetime(f"{year + 1}-{self.end_of_snow_year}")

                window_data = self.dynamic_forcing_and_swe[
                    (self.dynamic_forcing_and_swe["Station"] == station)
                    & (self.dynamic_forcing_and_swe["Date"] >= start_date)
                    & (self.dynamic_forcing_and_swe["Date"] <= end_of_snow_year_date)
                ]

                if window_data.empty:
                    raise RuntimeError(
                        f"No data found for station {station} in water year {year} "
                        f"(expected {start_date.date()} to {end_of_snow_year_date.date()}). "
                        f"Check date types and snow-year slicing."
                    )

                lookup.append(
                    {
                        "Station": station,
                        "Year": year,
                        "StartDate": pd.Timestamp(start_date),
                        "EndOfSnowYearDate": pd.Timestamp(end_of_snow_year_date),
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

        mask = (
            (self.dynamic_forcing_and_swe["Station"] == station)
            & (self.dynamic_forcing_and_swe["Date"] >= pd.Timestamp(start_date))
            & (self.dynamic_forcing_and_swe["Date"] <= pd.Timestamp(end_of_snow_year_date))
        )

        data = self.dynamic_forcing_and_swe[mask].copy()
        if len(data) == 0:
            raise ValueError(f"No data found for station {station} between {start_date} and {end_of_snow_year_date}")

        features = self.build_features(data)

        sample = {
            "dynamic forcing": torch.tensor(features.values, dtype=torch.float32),
            "swe": torch.tensor(data["SWE"].values, dtype=torch.float32),
            "dates": data["Date"].dt.strftime("%Y-%m-%d").values,
            "snotel attributes": torch.tensor(
                self.snotel_attributes.loc[self.snotel_attributes["Station"] == station, ["Elevation", "Slope", "Aspect"]].values,
                dtype=torch.float32,
            ),
            "year": year,
            "station": station,
        }
        if "SWE_climo" in data.columns:
            sample["swe_climo"] = torch.tensor(data["SWE_climo"].values, dtype=torch.float32)

        return sample

    def get(self, station: str, year: int) -> Dict[str, torch.Tensor]:
        """
        Get data for a specific station and year.
        """
        for idx, entry in enumerate(self.lookup_table):
            if entry["Station"] == station and entry["Year"] == year:
                return self.__getitem__(idx)
        raise ValueError(f"No data found for station {station} and year {year}")