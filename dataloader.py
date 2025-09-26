from typing import Tuple, Dict, List
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from dataset import SWEStationDataset
from utils.SpatialTransform import SpatialTransformer, precompute_weights, fast_transform_with_weights
import numpy as np
import pandas as pd

class SWEDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_years = range(cfg.train_start_year, cfg.train_end_year + 1)
        self.val_years = range(cfg.val_start_year, cfg.val_end_year + 1)
        self.test_years = range(cfg.test_start_year, cfg.test_end_year + 1)

    def prepare(self) -> Tuple[DataLoader, DataLoader, DataLoader, SpatialTransformer, dict]:
        """
        Create train,
          validation, and test data loaders with spatial decorrelation.
        """
        dataset = SWEStationDataset(self.cfg)
        
        orig_swe = dataset.dynamic_forcing_and_swe.reset_index()
        orig_swe = orig_swe[["Date", "Station", "SWE"]].copy()
        
        all_dates = orig_swe["Date"].unique()
        unique_dates = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='D')
        
        dataset.dynamic_forcing_and_swe = dataset.dynamic_forcing_and_swe.reset_index()
        
        MIN_COVERAGE = 95.0
        complete_stations = []
        
        for station in dataset.snotel_attributes["Station"].values:
            station_data = orig_swe[orig_swe["Station"] == station]
            station_dates = station_data["Date"].unique()
            coverage = len(station_dates) / len(unique_dates) * 100
            if coverage >= MIN_COVERAGE:
                complete_stations.append(station)
                
        dataset.snotel_attributes = dataset.snotel_attributes[
            dataset.snotel_attributes["Station"].isin(complete_stations)
        ].copy()

        dataset.dynamic_forcing_and_swe = dataset.dynamic_forcing_and_swe[
            dataset.dynamic_forcing_and_swe["Station"].isin(complete_stations)
        ].copy()
        
        train_indices = [
            idx for idx, entry in enumerate(dataset.lookup_table)
            if entry["Year"] in self.train_years
        ]
        val_indices = [
            idx for idx, entry in enumerate(dataset.lookup_table)
            if entry["Year"] in self.val_years
        ]
        test_indices = [
            idx for idx, entry in enumerate(dataset.lookup_table)
            if entry["Year"] in self.test_years
        ]
        
        station_order = dataset.snotel_attributes["Station"].values
        station_index = {s: i for i, s in enumerate(station_order)}
        locs_all = dataset.snotel_attributes[["Latitude", "Longitude"]].values

        weights = precompute_weights(
            trainLocs=locs_all,
            M=self.cfg.M,
            range_param=self.cfg.range_param,
            smoothness=self.cfg.smoothness,
            nugget=self.cfg.nugget,
        )

        backtrans_cache = {}
        unique_dates = sorted(dataset.dynamic_forcing_and_swe["Date"].unique())

        for dt in unique_dates:
            swe_dt = (
                pd.DataFrame({"Station": station_order})
                .merge(orig_swe[orig_swe["Date"] == dt][["Station", "SWE"]], on="Station", how="left")
            )
            if swe_dt["SWE"].isna().any():
                swe_dt["SWE"] = swe_dt["SWE"].fillna(swe_dt["SWE"].mean())

            y_vec = swe_dt["SWE"].to_numpy(dtype=np.float32)

            offsets = np.empty(len(station_order), dtype=np.float32)
            for i, nbrs in enumerate(weights.nn_index):
                offsets[i] = np.dot(weights.A[i], y_vec[nbrs])

            backtrans_cache[pd.Timestamp(dt).strftime("%Y-%m-%d")] = offsets

            if pd.Timestamp(dt).year <= self.cfg.train_end_year:
                time_mask = dataset.dynamic_forcing_and_swe["Date"] == dt
                df_dt = dataset.dynamic_forcing_and_swe[time_mask].copy()
                aligned = pd.DataFrame({"Station": station_order}).merge(df_dt, on="Station", how="left")

                numeric_cols = aligned.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    if aligned[col].isna().any():
                        aligned[col] = aligned[col].fillna(aligned[col].mean())

                data_for_transform = aligned.copy()
                cols_to_decor = ["SWE", "Tmax", "Tmin", "Precip", "Tobs", "TB_19", "TB_37", "TB_diff"]

                transformed = fast_transform_with_weights(
                    trainData=data_for_transform,
                    target="SWE",
                    weights=weights,
                    cols_to_transform=cols_to_decor,
                    static_cols=None,
                    station_col="Station"
                )

                updated = dataset.dynamic_forcing_and_swe.loc[time_mask].merge(
                    transformed[["Station"] + cols_to_decor],
                    on="Station",
                    suffixes=("", "_decor"),
                    how="left"
                )
                for col in cols_to_decor:
                    dataset.dynamic_forcing_and_swe.loc[time_mask, col] = updated[f"{col}_decor"].values

        val_dates = [d for d in unique_dates if pd.Timestamp(d).year in self.val_years]
        for dt in val_dates:
            time_mask = dataset.dynamic_forcing_and_swe["Date"] == dt
            df_dt = dataset.dynamic_forcing_and_swe[time_mask].copy()
            aligned = pd.DataFrame({"Station": station_order}).merge(df_dt, on="Station", how="left")

            numeric_cols = aligned.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if aligned[col].isna().any():
                    aligned[col] = aligned[col].fillna(aligned[col].mean())

            data_for_transform = aligned.copy()
            cols_to_decor = ["SWE", "Tmax", "Tmin", "Precip", "Tobs", "TB_19", "TB_37", "TB_diff"]

            transformed = fast_transform_with_weights(
                trainData=data_for_transform,
                target="SWE",
                weights=weights,
                cols_to_transform=cols_to_decor,
                static_cols=None,
                station_col="Station"
            )

            updated = dataset.dynamic_forcing_and_swe.loc[time_mask].merge(
                transformed[["Station"] + cols_to_decor],
                on="Station",
                suffixes=("", "_decor"),
                how="left"
            )
            for col in cols_to_decor:
                dataset.dynamic_forcing_and_swe.loc[time_mask, col] = updated[f"{col}_decor"].values

        test_dates = [d for d in unique_dates if pd.Timestamp(d).year in self.test_years]
        for dt in test_dates:
            time_mask = dataset.dynamic_forcing_and_swe["Date"] == dt
            df_dt = dataset.dynamic_forcing_and_swe[time_mask].copy()
            aligned = pd.DataFrame({"Station": station_order}).merge(df_dt, on="Station", how="left")

            numeric_cols = aligned.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if aligned[col].isna().any():
                    aligned[col] = aligned[col].fillna(aligned[col].mean())

            data_for_transform = aligned.copy()
            cols_to_decor = ["SWE", "Tmax", "Tmin", "Precip", "Tobs", "TB_19", "TB_37", "TB_diff"]

            transformed = fast_transform_with_weights(
                trainData=data_for_transform,
                target="SWE",
                weights=weights,
                cols_to_transform=cols_to_decor,
                static_cols=None,
                station_col="Station"
            )

            updated = dataset.dynamic_forcing_and_swe.loc[time_mask].merge(
                transformed[["Station"] + cols_to_decor],
                on="Station",
                suffixes=("", "_decor"),
                how="left"
            )
            for col in cols_to_decor:
                dataset.dynamic_forcing_and_swe.loc[time_mask, col] = updated[f"{col}_decor"].values

        if getattr(self.cfg, "anomaly_target", False):
            df_all = dataset.dynamic_forcing_and_swe.copy()
            df_all["DOY"] = pd.to_datetime(df_all["Date"]).dt.dayofyear
            train_mask = df_all["Date"].dt.year.isin(self.train_years)
            climo_model = (
                df_all.loc[train_mask]
                .groupby(["Station", "DOY"])["SWE"]
                .mean()
                .reset_index()
                .rename(columns={"SWE": "SWE_climo"})
            )
            station_train_mean = (
                df_all.loc[train_mask].groupby("Station")["SWE"].mean().rename("SWE_climo_fallback")
            )
            df_all = df_all.merge(climo_model, on=["Station", "DOY"], how="left")
            df_all = df_all.merge(station_train_mean, on="Station", how="left")
            df_all["SWE_climo"] = df_all["SWE_climo"].fillna(df_all["SWE_climo_fallback"]).fillna(0.0)
            df_all = df_all.drop(columns=["SWE_climo_fallback", "DOY"])
            dataset.dynamic_forcing_and_swe = df_all

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=self.cfg.shuffle,
            collate_fn=self.collate_fn
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.cfg.batch_size,
            collate_fn=self.collate_fn
        )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.cfg.batch_size,
            collate_fn=self.collate_fn
        )

        bt_info = {
            "weights": weights,
            "station_index": station_index,
            "backtrans_cache": backtrans_cache
        }
        
        return train_loader, val_loader, test_loader, SpatialTransformer(), bt_info

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        dynamic = [item['dynamic forcing'] for item in batch]
        swe = [item['swe'] for item in batch]
        has_climo = 'swe_climo' in batch[0]
        if has_climo:
            swe_clm = [item['swe_climo'] for item in batch]
        metadata = {
            key: [item[key] for item in batch]
            for key in batch[0].keys()
            if key not in ['dynamic forcing', 'swe', 'swe_climo']
        }
        dynamic_pad = pad_sequence(dynamic, batch_first=True)
        swe_pad = pad_sequence(swe, batch_first=True)
        if has_climo:
            swe_clm_pad = pad_sequence(swe_clm, batch_first=True)
        lengths = torch.tensor([len(x) for x in dynamic])
        max_len = lengths.max()
        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        out = {
            'dynamic forcing': dynamic_pad,
            'swe': swe_pad,
            'mask': mask,
            **metadata
        }
        if has_climo:
            out['swe_climo'] = swe_clm_pad
        return out