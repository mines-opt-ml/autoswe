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
        self.train_years = range(cfg.beginning_year, cfg.train_end_year + 1)
        self.val_years = range(cfg.val_start_year, cfg.end_year + 1)

    def prepare(self) -> Tuple[DataLoader, DataLoader, SpatialTransformer, dict]:
        """
        Create train and validation data loaders with spatial decorrelation.
        
        Returns:
            Tuple containing:
            - train_loader: DataLoader for training data 
            - val_loader: DataLoader for validation data 
            - transformer: SpatialTransformer object for back-transformation
            - spatial_obj: Dictionary with transformation information
        """
        dataset = SWEStationDataset(self.cfg)
        
        orig_swe = dataset.dynamic_forcing_and_swe.reset_index()
        orig_swe = orig_swe[["Date", "Station", "SWE"]].copy()
        
        all_dates = orig_swe["Date"].unique()
        unique_dates = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='D')
        
        dataset.dynamic_forcing_and_swe = dataset.dynamic_forcing_and_swe.reset_index()
        
        stations = orig_swe["Station"].unique()
        station_coverages = []
        
        for station in stations:
            station_dates = orig_swe[orig_swe["Station"] == station]["Date"].unique()
            coverage = len(station_dates) / len(unique_dates) * 100
            station_coverages.append({
                'station': station,
                'coverage': coverage
            })
                
        MIN_COVERAGE = 95.0
        complete_stations = []
        incomplete_info = {}
        
        for station in dataset.snotel_attributes["Station"].values:
            station_data = orig_swe[orig_swe["Station"] == station]
            station_dates = station_data["Date"].unique()
            coverage = len(station_dates) / len(unique_dates) * 100
            
            if coverage >= MIN_COVERAGE:
                complete_stations.append(station)
            else:
                missing_dates = set(unique_dates) - set(station_dates)
                incomplete_info[station] = {
                    "coverage": coverage,
                    "missing_count": len(missing_dates),
                    "sample_missing": list(missing_dates)[:3]
                }
                
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
                swe_dt = swe_dt.copy()
                swe_dt["SWE"] = swe_dt["SWE"].fillna(swe_dt["SWE"].mean())

            y_vec = swe_dt["SWE"].to_numpy(dtype=np.float32)

            offsets = np.empty(len(station_order), dtype=np.float32)
            for i, nbrs in enumerate(weights.nn_index):
                offsets[i] = np.dot(weights.A[i], y_vec[nbrs])
            backtrans_cache[pd.Timestamp(dt).strftime("%Y-%m-%d")] = offsets

            time_mask = dataset.dynamic_forcing_and_swe["Date"] == dt
            df_dt = dataset.dynamic_forcing_and_swe[time_mask].copy()
            df_num = df_dt.select_dtypes(include=[np.number]).copy()
            
            aligned = pd.DataFrame({"Station": station_order})
            aligned = aligned.merge(df_dt[["Station"]], on="Station", how="left")
            
            numeric_data = df_num.join(df_dt["Station"]).drop(columns=[])
            aligned = aligned.merge(numeric_data, on="Station", how="left")
            
            numeric_cols = df_num.columns
            for col in numeric_cols:
                if aligned[col].isna().any():
                    aligned[col] = aligned[col].fillna(aligned[col].mean())

            data_for_transform = aligned.drop(columns=["Station"])
            transformed = fast_transform_with_weights(
                trainData=data_for_transform, target="SWE", weights=weights
            )

            swe_trans = pd.DataFrame({"Station": station_order, "SWE": transformed["SWE"].values})
            rewritten = df_dt[["Station"]].merge(swe_trans, on="Station", how="left")["SWE"].values
            dataset.dynamic_forcing_and_swe.loc[time_mask, "SWE"] = rewritten

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

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

        bt_info = {
            "weights": weights,
            "station_index": station_index,
            "backtrans_cache": backtrans_cache
        }
        
        return train_loader, val_loader, SpatialTransformer(), bt_info

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to pad sequences in batch to same length.
        
        Args:
            batch: List of dictionaries containing tensors of varying sequence lengths

        Returns:
            Dictionary with padded tensors and mask indicating valid timesteps
        """
        dynamic = [item['dynamic forcing'] for item in batch]
        swe = [item['swe'] for item in batch]

        metadata = {
            key: [item[key] for item in batch]
            for key in batch[0].keys()
            if key not in ['dynamic forcing', 'swe']
        }

        dynamic_pad = pad_sequence(dynamic, batch_first=True)
        swe_pad = pad_sequence(swe, batch_first=True)

        lengths = torch.tensor([len(x) for x in dynamic])
        max_len = lengths.max()
        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        return {
            'dynamic forcing': dynamic_pad,
            'swe': swe_pad,
            'mask': mask,
            **metadata
        }