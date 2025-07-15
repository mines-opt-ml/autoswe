import os
from functools import reduce
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


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
        self.beginning_of_snow_year = pd.to_datetime(self.cfg.beginning_of_snow_year)
        self.peak_swe_date = pd.to_datetime(self.cfg.peak_swe_date)
        self.beginning_year = self.cfg.beginning_year
        self.end_year = self.cfg.end_year

        # Load data into a single pandas DataFrame
        swe_long = self._load_swe_values()
        meta = self._load_meta_data()
        dynamic_merged = self._load_dynamic_forcing_data()
        self.merged = self._merge_data(swe_long, dynamic_merged, meta)

    ### Loading functions
    def _load_swe_values(self) -> pd.DataFrame:
        swe = pd.read_csv(self.cfg.swe_path)
        swe["Date"] = pd.to_datetime(swe["Date"]).dt.strftime("%Y-%m-%d")
        swe_long = swe.melt(id_vars="Date", var_name="Station", value_name="SWE")
        swe_long["Station"] = swe_long["Station"].str.replace("_", " ").str.lower()
        return swe_long

    def _load_meta_data(self) -> pd.DataFrame:
        meta = pd.read_csv(self.cfg.meta_data_path)
        meta["Station_clean"] = meta["Station Name"].str.lower()
        return meta

    def _load_dynamic_forcing_data(self) -> pd.DataFrame:
        max_temp = self._melt_dynamic(self.cfg.max_temp_path, "Tmax")
        min_temp = self._melt_dynamic(self.cfg.min_temp_path, "Tmin")
        precip = self._melt_dynamic(self.cfg.precip_path, "Precip")
        obs_temp = self._melt_dynamic(self.cfg.obs_temp_path, "Tobs")
        tb19 = self._melt_dynamic(self.cfg.tb19_path, "TB_19")
        tb37 = self._melt_dynamic(self.cfg.tb37_path, "TB_37")
        tb_diff = self._melt_dynamic(self.cfg.tbdiff_path, "TB_diff")

        dynamic_dfs = [max_temp, min_temp, precip, obs_temp, tb19, tb37, tb_diff]
        dynamic_merged = reduce(lambda left, right: pd.merge(left, right, on=["Date", "Station"]), dynamic_dfs)
        return dynamic_merged

    def _merge_data(self, swe_long: pd.DataFrame, dynamic_merged: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(swe_long, dynamic_merged, on=["Date", "Station"], how="inner")
        merged = pd.merge(merged, meta, left_on="Station", right_on="Station_clean", how="inner")
        merged = merged.rename(columns={"Station_clean": "Station"})
        merged = merged.dropna()
        merged["Date"] = pd.to_datetime(merged["Date"])
        return merged

    def _melt_dynamic(self, path: os.PathLike, var_name: str) -> pd.DataFrame:
        # @Colin: what does this function do? Could we place it in a utlils folder
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")  # normalize date format
        df_long = df.melt(id_vars="Date", var_name="Station", value_name=var_name)
        df_long["Station"] = df_long["Station"].str.replace("_", " ").str.lower()
        return df_long

    ## Small utils
    def _get_snow_year(self, idx: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Get the start and end dates of the snow year for a given index.
        """
        year = self.beginning_year + idx
        if year > self.end_year:
            raise IndexError("Index out of range for the dataset.")
        start_date = pd.Timestamp(year=year, month=self.beginning_of_snow_year.month, day=self.beginning_of_snow_year.day)
        end_date = pd.Timestamp(year=year + 1, month=self.peak_swe_date.month, day=self.peak_swe_date.day)

        return start_date, end_date

    ## Required class methods for Dataset
    def __len__(self):
        return self.end_year - self.beginning_year + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_date, end_date = self._get_snow_year(idx)
        self.merged.set_index("Date")
        snow_year_merged_data = self.merged.loc[start_date:end_date]
        X = build_features(snow_year_merged_data)  # ToDo: need to implement build_features
        y = snow_year_merged_data["SWE"].values
        meta = snow_year_merged_data["Station"]
        return {
            "X": torch.tensor(X.values, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32).view(-1, 1),
            "meta": meta.tolist(),
        }
